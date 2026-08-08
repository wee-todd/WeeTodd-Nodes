import Foundation
import Metal
import MetalPerformanceShadersGraph

struct Options {
  var rows = 7_689
  var warmup = 1
  var iterations = 5
}

let hidden = 5_376
let heads = 56
let headDim = 128
let inner = heads * headDim
let ffn = 14_336
let rotaryDim = 96
let qkvWidth = 3 * inner
let fc1Width = 2 * ffn

func fail(_ message: String) -> Never {
  FileHandle.standardError.write(Data("error: \(message)\n".utf8))
  exit(2)
}

func parseOptions() -> Options {
  var options = Options()
  let arguments = Array(CommandLine.arguments.dropFirst())
  var index = 0
  while index < arguments.count {
    guard index + 1 < arguments.count else { fail("missing value for \(arguments[index])") }
    let name = arguments[index]
    let value = arguments[index + 1]
    switch name {
    case "--rows": options.rows = Int(value) ?? 0
    case "--warmup": options.warmup = Int(value) ?? -1
    case "--iterations": options.iterations = Int(value) ?? 0
    default: fail("unknown argument \(name)")
    }
    index += 2
  }
  guard options.rows > 0 else { fail("rows must be positive") }
  guard options.warmup >= 0, options.iterations > 0 else {
    fail("warmup must be nonnegative and iterations must be positive")
  }
  return options
}

func checkedElements(_ values: Int...) -> Int {
  var result = 1
  for value in values {
    let (next, overflow) = result.multipliedReportingOverflow(by: value)
    if overflow { fail("tensor element count overflow") }
    result = next
  }
  return result
}

func checkedBytes(elements: Int, stride: Int) -> Int {
  let (bytes, overflow) = elements.multipliedReportingOverflow(by: stride)
  if overflow { fail("tensor byte count overflow") }
  return bytes
}

func shape(_ values: Int...) -> [NSNumber] {
  values.map(NSNumber.init(value:))
}

func makeUInt16Buffer(device: MTLDevice, elements: Int, bits: UInt16) -> MTLBuffer {
  let bytes = checkedBytes(elements: elements, stride: MemoryLayout<UInt16>.stride)
  guard let buffer = device.makeBuffer(length: bytes, options: .storageModeShared) else {
    fail("Metal BF16 buffer allocation failed")
  }
  let values = buffer.contents().bindMemory(to: UInt16.self, capacity: elements)
  values.update(repeating: bits, count: elements)
  return buffer
}

func makeIndexBuffer(device: MTLDevice, rows: Int) -> MTLBuffer {
  let bytes = checkedBytes(elements: rows, stride: MemoryLayout<Int32>.stride)
  guard let buffer = device.makeBuffer(length: bytes, options: .storageModeShared) else {
    fail("Metal index buffer allocation failed")
  }
  let values = buffer.contents().bindMemory(to: Int32.self, capacity: rows)
  for index in 0..<rows { values[index] = Int32(index % 3) }
  return buffer
}

func monotonicSeconds() -> Double {
  Double(DispatchTime.now().uptimeNanoseconds) / 1_000_000_000
}

func median(_ values: [Double]) -> Double {
  let sorted = values.sorted()
  let middle = sorted.count / 2
  return sorted.count % 2 == 0
    ? (sorted[middle - 1] + sorted[middle]) / 2
    : sorted[middle]
}

func bfloat16ToFloat(_ bits: UInt16) -> Float {
  Float(bitPattern: UInt32(bits) << 16)
}

let options = parseOptions()
guard let device = MTLCreateSystemDefaultDevice() else { fail("Metal device is unavailable") }
guard let commandQueue = device.makeCommandQueue() else { fail("Metal command queue creation failed") }
let graph = MPSGraph()
let bf16 = MPSDataType.bFloat16

var buffers: [MTLBuffer] = []
var feedTypes: [MPSGraphTensor: MPSGraphShapedType] = [:]
var feedData: [MPSGraphTensor: MPSGraphTensorData] = [:]

func addBF16Feed(_ name: String, _ tensorShape: [NSNumber], bits: UInt16) -> MPSGraphTensor {
  let elements = tensorShape.reduce(1) { $0 * $1.intValue }
  let buffer = makeUInt16Buffer(device: device, elements: elements, bits: bits)
  buffers.append(buffer)
  let tensor = graph.placeholder(shape: tensorShape, dataType: bf16, name: name)
  feedTypes[tensor] = MPSGraphShapedType(shape: tensorShape, dataType: bf16)
  feedData[tensor] = MPSGraphTensorData(buffer, shape: tensorShape, dataType: bf16)
  return tensor
}

let xShape = shape(1, options.rows, hidden)
let x = addBF16Feed("x", xShape, bits: 0x3c80)  // 0.015625
let norm1Weight = addBF16Feed("norm1_weight", shape(hidden), bits: 0x3f80)
let norm2Weight = addBF16Feed("norm2_weight", shape(hidden), bits: 0x3f80)
let qNormWeight = addBF16Feed("q_norm_weight", shape(headDim), bits: 0x3f80)
let kNormWeight = addBF16Feed("k_norm_weight", shape(headDim), bits: 0x3f80)
let qkvWeight = addBF16Feed("qkv_weight", shape(qkvWidth, hidden), bits: 0x3900)
let outWeight = addBF16Feed("out_weight", shape(hidden, inner), bits: 0x3900)
let fc1Weight = addBF16Feed("fc1_weight", shape(fc1Width, hidden), bits: 0x3900)
let fc2Weight = addBF16Feed("fc2_weight", shape(hidden, ffn), bits: 0x3900)
let shiftMSA = addBF16Feed("shift_msa", shape(3, hidden), bits: 0x0000)
let scaleMSA = addBF16Feed("scale_msa", shape(3, hidden), bits: 0x0000)
let gateMSA = addBF16Feed("gate_msa", shape(3, hidden), bits: 0x3f80)
let shiftMLP = addBF16Feed("shift_mlp", shape(3, hidden), bits: 0x0000)
let scaleMLP = addBF16Feed("scale_mlp", shape(3, hidden), bits: 0x0000)
let gateMLP = addBF16Feed("gate_mlp", shape(3, hidden), bits: 0x3f80)
let rotaryCos = addBF16Feed("rotary_cos", shape(options.rows, rotaryDim), bits: 0x3f80)
let rotarySin = addBF16Feed("rotary_sin", shape(options.rows, rotaryDim), bits: 0x0000)

let indexBuffer = makeIndexBuffer(device: device, rows: options.rows)
buffers.append(indexBuffer)
let indexShape = shape(options.rows)
let indices = graph.placeholder(shape: indexShape, dataType: .int32, name: "adaln_indices")
feedTypes[indices] = MPSGraphShapedType(shape: indexShape, dataType: .int32)
feedData[indices] = MPSGraphTensorData(indexBuffer, shape: indexShape, dataType: .int32)

func rmsNorm(
  _ input: MPSGraphTensor,
  weight: MPSGraphTensor,
  inputShape: [NSNumber],
  name: String
) -> MPSGraphTensor {
  let fp32 = graph.cast(input, to: .float32, name: "\(name)_fp32")
  let square = graph.square(with: fp32, name: "\(name)_square")
  let mean = graph.mean(of: square, axes: [-1], name: "\(name)_mean")
  let reducedShape = Array(inputShape.dropLast()) + [NSNumber(value: 1)]
  let meanExpanded = graph.reshape(mean, shape: reducedShape, name: "\(name)_mean_expanded")
  let epsilon = graph.constant(1e-5, dataType: .float32)
  let inverse = graph.reciprocalSquareRoot(
    graph.addition(meanExpanded, epsilon, name: nil), name: "\(name)_inverse")
  let normalized = graph.multiplication(fp32, inverse, name: "\(name)_normalized")
  let normalizedBF16 = graph.cast(normalized, to: bf16, name: "\(name)_bf16")
  return graph.multiplication(normalizedBF16, weight, name: name)
}

func linear(
  _ input: MPSGraphTensor,
  weight: MPSGraphTensor,
  name: String
) -> MPSGraphTensor {
  let transposed = graph.transpose(weight, permutation: [1, 0], name: "\(name)_weight_t")
  return graph.matrixMultiplication(primary: input, secondary: transposed, name: name)
}

func gatherRows(_ table: MPSGraphTensor, name: String) -> MPSGraphTensor {
  let gathered = graph.gather(
    withUpdatesTensor: table,
    indicesTensor: indices,
    axis: 0,
    batchDimensions: 0,
    name: name)
  return graph.reshape(gathered, shape: xShape, name: "\(name)_batched")
}

func modulate(
  _ input: MPSGraphTensor,
  shift: MPSGraphTensor,
  scale: MPSGraphTensor,
  name: String
) -> MPSGraphTensor {
  let one = graph.constant(1.0, dataType: bf16)
  let scaled = graph.multiplication(
    input,
    graph.addition(one, gatherRows(scale, name: "\(name)_scale"), name: nil),
    name: "\(name)_scaled")
  return graph.addition(scaled, gatherRows(shift, name: "\(name)_shift"), name: name)
}

func rotary(_ input: MPSGraphTensor, name: String) -> MPSGraphTensor {
  let rotatedInput = graph.sliceTensor(
    input, dimension: 3, start: 0, length: rotaryDim, name: "\(name)_input")
  let pass = graph.sliceTensor(
    input, dimension: 3, start: rotaryDim, length: headDim - rotaryDim, name: "\(name)_pass")
  let first = graph.sliceTensor(
    rotatedInput, dimension: 3, start: 0, length: rotaryDim / 2, name: "\(name)_first")
  let second = graph.sliceTensor(
    rotatedInput,
    dimension: 3,
    start: rotaryDim / 2,
    length: rotaryDim / 2,
    name: "\(name)_second")
  let rotateHalf = graph.concatTensors(
    [graph.negative(with: second, name: nil), first], dimension: 3, name: "\(name)_half")
  let cos = graph.reshape(
    rotaryCos, shape: shape(1, 1, options.rows, rotaryDim), name: "rotary_cos_broadcast")
  let sin = graph.reshape(
    rotarySin, shape: shape(1, 1, options.rows, rotaryDim), name: "rotary_sin_broadcast")
  let mixed = graph.addition(
    graph.multiplication(rotatedInput, cos, name: nil),
    graph.multiplication(rotateHalf, sin, name: nil),
    name: "\(name)_mixed")
  return graph.concatTensors([mixed, pass], dimension: 3, name: name)
}

let normalized1 = rmsNorm(x, weight: norm1Weight, inputShape: xShape, name: "norm1")
let attentionInput = modulate(normalized1, shift: shiftMSA, scale: scaleMSA, name: "adaln1")
let qkv = graph.reshape(
  linear(attentionInput, weight: qkvWeight, name: "qkv"),
  shape: shape(1, options.rows, heads, 3, headDim),
  name: "qkv_heads")
func qkvPart(_ part: Int, name: String) -> MPSGraphTensor {
  let sliced = graph.sliceTensor(qkv, dimension: 3, start: part, length: 1, name: name)
  return graph.reshape(sliced, shape: shape(1, options.rows, heads, headDim), name: "\(name)_squeezed")
}
let qRows = qkvPart(0, name: "q")
let kRows = qkvPart(1, name: "k")
let vRows = qkvPart(2, name: "v")
let qHeads = graph.transpose(qRows, permutation: [0, 2, 1, 3], name: "q_heads")
let kHeads = graph.transpose(kRows, permutation: [0, 2, 1, 3], name: "k_heads")
let vHeads = graph.transpose(vRows, permutation: [0, 2, 1, 3], name: "v_heads")
let headShape = shape(1, heads, options.rows, headDim)
let qNormalized = rmsNorm(qHeads, weight: qNormWeight, inputShape: headShape, name: "q_norm")
let kNormalized = rmsNorm(kHeads, weight: kNormWeight, inputShape: headShape, name: "k_norm")
let qRotary = rotary(qNormalized, name: "q_rotary")
let kRotary = rotary(kNormalized, name: "k_rotary")
let attended = graph.scaledDotProductAttention(
  query: qRotary,
  key: kRotary,
  value: vHeads,
  scale: Float(1.0 / sqrt(Double(headDim))),
  name: "sdpa")
let attendedRows = graph.reshape(
  graph.transpose(attended, permutation: [0, 2, 1, 3], name: "attention_rows"),
  shape: shape(1, options.rows, inner),
  name: "attention_flat")
let attentionOutput = linear(attendedRows, weight: outWeight, name: "out_proj")
let attentionGate = gatherRows(gateMSA, name: "attention_gate")
let attentionResidual = graph.addition(
  x,
  graph.multiplication(attentionGate, attentionOutput, name: nil),
  name: "attention_residual")

let normalized2 = rmsNorm(
  attentionResidual, weight: norm2Weight, inputShape: xShape, name: "norm2")
let mlpInput = modulate(normalized2, shift: shiftMLP, scale: scaleMLP, name: "adaln2")
let fc1 = linear(mlpInput, weight: fc1Weight, name: "fc1")
let gate = graph.sliceTensor(fc1, dimension: 2, start: 0, length: ffn, name: "swiglu_gate")
let value = graph.sliceTensor(fc1, dimension: 2, start: ffn, length: ffn, name: "swiglu_value")
let activatedGate = graph.multiplication(
  gate, graph.sigmoid(with: gate, name: nil), name: "silu")
let mlpHidden = graph.multiplication(activatedGate, value, name: "swiglu")
let mlpOutput = linear(mlpHidden, weight: fc2Weight, name: "fc2")
let mlpGate = gatherRows(gateMLP, name: "mlp_gate")
let output = graph.addition(
  attentionResidual,
  graph.multiplication(mlpGate, mlpOutput, name: nil),
  name: "block_output")

let outputElements = checkedElements(1, options.rows, hidden)
guard let outputBuffer = device.makeBuffer(
  length: checkedBytes(elements: outputElements, stride: MemoryLayout<UInt16>.stride),
  options: .storageModeShared)
else { fail("Metal output buffer allocation failed") }
buffers.append(outputBuffer)
let outputData = MPSGraphTensorData(outputBuffer, shape: xShape, dataType: bf16)

let descriptor = MPSGraphCompilationDescriptor()
descriptor.optimizationLevel = .level1
descriptor.waitForCompilationCompletion = true
let graphDevice = MPSGraphDevice(mtlDevice: device)
let compileStarted = monotonicSeconds()
let executable = graph.compile(
  with: graphDevice,
  feeds: feedTypes,
  targetTensors: [output],
  targetOperations: nil,
  compilationDescriptor: descriptor)
let compileSeconds = monotonicSeconds() - compileStarted

guard let orderedFeeds = executable.feedTensors?.map({ feedData[$0]! }) else {
  fail("MPSGraph executable did not expose feed order")
}

func execute() -> Double {
  let started = monotonicSeconds()
  _ = executable.run(
    with: commandQueue,
    inputs: orderedFeeds,
    results: [outputData],
    executionDescriptor: nil)
  return monotonicSeconds() - started
}

let firstExecutionSeconds = execute()
for _ in 0..<options.warmup { _ = execute() }
var samples: [Double] = []
for _ in 0..<options.iterations { samples.append(execute()) }
let medianSeconds = median(samples)
let outputBits = outputBuffer.contents().bindMemory(to: UInt16.self, capacity: outputElements)[0]
let explicitBufferBytes = buffers.reduce(0) { $0 + $1.length }
let result: [String: Any] = [
  "backend": "mpsgraph_block",
  "device": device.name,
  "dtype": "bf16",
  "rows": options.rows,
  "compile_seconds": compileSeconds,
  "first_execution_seconds": firstExecutionSeconds,
  "warm_median_seconds": medianSeconds,
  "warm_samples_seconds": samples,
  "output_sample": bfloat16ToFloat(outputBits),
  "explicit_buffer_bytes": explicitBufferBytes,
  "metal_current_allocated_bytes": device.currentAllocatedSize,
  "warmup": options.warmup,
  "iterations": options.iterations,
]
let json = try JSONSerialization.data(withJSONObject: result, options: [.prettyPrinted, .sortedKeys])
FileHandle.standardOutput.write(json)
FileHandle.standardOutput.write(Data("\n".utf8))
