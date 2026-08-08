import Foundation
import Metal
import MetalPerformanceShadersGraph

struct Options {
  var rows = 7_689
  var inputDim = 5_376
  var outputDim = 21_504
  var warmup = 2
  var iterations = 7
  var dtype = "bf16"
}

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
    case "--input-dim": options.inputDim = Int(value) ?? 0
    case "--output-dim": options.outputDim = Int(value) ?? 0
    case "--warmup": options.warmup = Int(value) ?? -1
    case "--iterations": options.iterations = Int(value) ?? 0
    case "--dtype": options.dtype = value
    default: fail("unknown argument \(name)")
    }
    index += 2
  }
  guard options.rows > 0, options.inputDim > 0, options.outputDim > 0 else {
    fail("matrix dimensions must be positive")
  }
  guard options.warmup >= 0, options.iterations > 0 else {
    fail("warmup must be nonnegative and iterations must be positive")
  }
  guard options.dtype == "bf16" || options.dtype == "fp16" else {
    fail("dtype must be bf16 or fp16")
  }
  return options
}

func checkedBytes(_ elements: Int) -> Int {
  let (bytes, overflow) = elements.multipliedReportingOverflow(by: MemoryLayout<UInt16>.stride)
  if overflow { fail("buffer byte count overflow") }
  return bytes
}

func checkedElements(_ left: Int, _ right: Int) -> Int {
  let (elements, overflow) = left.multipliedReportingOverflow(by: right)
  if overflow { fail("matrix element count overflow") }
  return elements
}

func makeFilledBuffer(device: MTLDevice, elements: Int, bits: UInt16) -> MTLBuffer {
  guard let buffer = device.makeBuffer(length: checkedBytes(elements), options: .storageModeShared)
  else { fail("Metal buffer allocation failed") }
  let values = buffer.contents().bindMemory(to: UInt16.self, capacity: elements)
  for index in 0..<elements { values[index] = bits }
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

let dataType: MPSDataType = options.dtype == "bf16" ? .bFloat16 : .float16
let sourceBits: UInt16 = options.dtype == "bf16"
  ? UInt16(0x3c80)
  : Float16(0.015625).bitPattern
let weightBits: UInt16 = options.dtype == "bf16"
  ? UInt16(0x3d00)
  : Float16(0.03125).bitPattern

let sourceElements = checkedElements(options.rows, options.inputDim)
let weightElements = checkedElements(options.outputDim, options.inputDim)
let outputElements = checkedElements(options.rows, options.outputDim)
let allocationBefore = device.currentAllocatedSize
let sourceBuffer = makeFilledBuffer(device: device, elements: sourceElements, bits: sourceBits)
let weightBuffer = makeFilledBuffer(device: device, elements: weightElements, bits: weightBits)
guard let outputBuffer = device.makeBuffer(
  length: checkedBytes(outputElements), options: .storageModeShared)
else { fail("Metal output buffer allocation failed") }

let graph = MPSGraph()
let sourceShape: [NSNumber] = [NSNumber(value: options.rows), NSNumber(value: options.inputDim)]
let weightShape: [NSNumber] = [NSNumber(value: options.outputDim), NSNumber(value: options.inputDim)]
let outputShape: [NSNumber] = [NSNumber(value: options.rows), NSNumber(value: options.outputDim)]
let source = graph.placeholder(shape: sourceShape, dataType: dataType, name: "source")
let weight = graph.placeholder(shape: weightShape, dataType: dataType, name: "weight")
let transposedWeight = graph.transpose(weight, permutation: [1, 0], name: "weight_transpose")
let output = graph.matrixMultiplication(primary: source, secondary: transposedWeight, name: "projection")

let sourceType = MPSGraphShapedType(shape: sourceShape, dataType: dataType)
let weightType = MPSGraphShapedType(shape: weightShape, dataType: dataType)
let descriptor = MPSGraphCompilationDescriptor()
descriptor.optimizationLevel = .level1
descriptor.waitForCompilationCompletion = true
let graphDevice = MPSGraphDevice(mtlDevice: device)
let compileStarted = monotonicSeconds()
let executable = graph.compile(
  with: graphDevice,
  feeds: [source: sourceType, weight: weightType],
  targetTensors: [output],
  targetOperations: nil,
  compilationDescriptor: descriptor)
let compileSeconds = monotonicSeconds() - compileStarted

let sourceData = MPSGraphTensorData(sourceBuffer, shape: sourceShape, dataType: dataType)
let weightData = MPSGraphTensorData(weightBuffer, shape: weightShape, dataType: dataType)
let outputData = MPSGraphTensorData(outputBuffer, shape: outputShape, dataType: dataType)

func execute() -> Double {
  let started = monotonicSeconds()
  _ = executable.run(
    with: commandQueue,
    inputs: [sourceData, weightData],
    results: [outputData],
    executionDescriptor: nil)
  return monotonicSeconds() - started
}

let firstExecutionSeconds = execute()
for _ in 0..<options.warmup { _ = execute() }
var samples: [Double] = []
for _ in 0..<options.iterations { samples.append(execute()) }

let outputBits = outputBuffer.contents().bindMemory(to: UInt16.self, capacity: outputElements)[0]
let outputSample = options.dtype == "bf16"
  ? bfloat16ToFloat(outputBits)
  : Float(Float16(bitPattern: outputBits))
let expectedSample = Float(options.inputDim) * 0.015625 * 0.03125
let operationCount = 2.0 * Double(options.rows) * Double(options.inputDim) * Double(options.outputDim)
let medianSeconds = median(samples)
let result: [String: Any] = [
  "backend": "mpsgraph",
  "device": device.name,
  "dtype": options.dtype,
  "rows": options.rows,
  "input_dim": options.inputDim,
  "output_dim": options.outputDim,
  "compile_seconds": compileSeconds,
  "first_execution_seconds": firstExecutionSeconds,
  "warm_median_seconds": medianSeconds,
  "warm_samples_seconds": samples,
  "warm_tflops": operationCount / medianSeconds / 1e12,
  "output_sample": outputSample,
  "expected_sample": expectedSample,
  "sample_absolute_error": abs(outputSample - expectedSample),
  "explicit_buffer_bytes": checkedBytes(sourceElements + weightElements + outputElements),
  "metal_allocated_delta_bytes": device.currentAllocatedSize - allocationBefore,
  "warmup": options.warmup,
  "iterations": options.iterations,
]
let json = try JSONSerialization.data(withJSONObject: result, options: [.prettyPrinted, .sortedKeys])
FileHandle.standardOutput.write(json)
FileHandle.standardOutput.write(Data("\n".utf8))
