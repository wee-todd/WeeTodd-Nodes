import Foundation
import Metal

struct Options {
  var rows = 7_689
  var inputDim = 5_376
  var outputDim = 21_504
  var tileM = 64
  var tileN = 64
  var simdgroups = 4
  var warmup = 2
  var iterations = 7
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
    case "--tile-m": options.tileM = Int(value) ?? 0
    case "--tile-n": options.tileN = Int(value) ?? 0
    case "--simdgroups": options.simdgroups = Int(value) ?? 0
    case "--warmup": options.warmup = Int(value) ?? -1
    case "--iterations": options.iterations = Int(value) ?? 0
    default: fail("unknown argument \(name)")
    }
    index += 2
  }
  guard options.rows > 0, options.inputDim > 0, options.outputDim > 0 else {
    fail("matrix dimensions must be positive")
  }
  guard options.tileM > 0, options.tileN > 0, options.simdgroups > 0 else {
    fail("tile dimensions and simdgroup count must be positive")
  }
  guard options.warmup >= 0, options.iterations > 0 else {
    fail("warmup must be nonnegative and iterations must be positive")
  }
  return options
}

func checkedElements(_ left: Int, _ right: Int) -> Int {
  let (elements, overflow) = left.multipliedReportingOverflow(by: right)
  if overflow { fail("matrix element count overflow") }
  return elements
}

func checkedBytes(_ elements: Int) -> Int {
  let (bytes, overflow) = elements.multipliedReportingOverflow(by: MemoryLayout<UInt16>.stride)
  if overflow { fail("buffer byte count overflow") }
  return bytes
}

func makeFilledBuffer(device: MTLDevice, elements: Int, bits: UInt16) -> MTLBuffer {
  guard let buffer = device.makeBuffer(length: checkedBytes(elements), options: .storageModeShared)
  else { fail("Metal buffer allocation failed") }
  buffer.contents().bindMemory(to: UInt16.self, capacity: elements).update(
    repeating: bits, count: elements)
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
guard #available(macOS 26.0, *) else { fail("MPP tensor operations require macOS 26 or newer") }
guard let device = MTLCreateSystemDefaultDevice() else { fail("Metal device is unavailable") }
guard let commandQueue = device.makeCommandQueue() else { fail("Metal command queue creation failed") }

let source = """
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

kernel void mpp_bf16_nt_matmul(
    device bfloat* source [[buffer(0)]],
    device bfloat* weight [[buffer(1)]],
    device bfloat* output [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]]) {
  auto A = tensor(source, dextents<int, 2>{int(K), int(M)}, array<int, 2>{1, int(K)});
  auto B = tensor(weight, dextents<int, 2>{int(K), int(N)}, array<int, 2>{1, int(K)});
  auto C = tensor(output, dextents<int, 2>{int(N), int(M)}, array<int, 2>{1, int(N)});

  constexpr auto descriptor = matmul2d_descriptor(
      \(options.tileM), \(options.tileN), static_cast<int>(dynamic_extent),
      false, true, false);
  matmul2d<descriptor, execution_simdgroups<\(options.simdgroups)>> operation;

  auto tileA = A.slice(0, tgid.y * \(options.tileM));
  auto tileB = B.slice(0, tgid.x * \(options.tileN));
  auto tileC = C.slice(tgid.x * \(options.tileN), tgid.y * \(options.tileM));
  auto result = operation.get_destination_cooperative_tensor<
      decltype(tileA), decltype(tileB), bfloat>();
  #pragma unroll
  for (ushort index = 0; index < result.get_capacity(); ++index) {
    result[index] = bfloat(0.0f);
  }
  operation.run(tileA, tileB, result);
  result.store(tileC);
}
"""

let compileOptions = MTLCompileOptions()
compileOptions.languageVersion = .version4_0
let compileStarted = monotonicSeconds()
let library: MTLLibrary
do {
  library = try device.makeLibrary(source: source, options: compileOptions)
} catch {
  fail("Metal library compilation failed: \(error)")
}
guard let function = library.makeFunction(name: "mpp_bf16_nt_matmul") else {
  fail("MPP benchmark kernel was not found")
}
let pipeline: MTLComputePipelineState
do {
  pipeline = try device.makeComputePipelineState(function: function)
} catch {
  fail("compute pipeline creation failed: \(error)")
}
let compileSeconds = monotonicSeconds() - compileStarted

let sourceElements = checkedElements(options.rows, options.inputDim)
let weightElements = checkedElements(options.outputDim, options.inputDim)
let outputElements = checkedElements(options.rows, options.outputDim)
let allocationBefore = device.currentAllocatedSize
let sourceBuffer = makeFilledBuffer(device: device, elements: sourceElements, bits: 0x3c80)
let weightBuffer = makeFilledBuffer(device: device, elements: weightElements, bits: 0x3d00)
guard let outputBuffer = device.makeBuffer(
  length: checkedBytes(outputElements), options: .storageModeShared)
else { fail("Metal output buffer allocation failed") }

var m = UInt32(options.rows)
var n = UInt32(options.outputDim)
var k = UInt32(options.inputDim)
let threadgroups = MTLSize(
  width: (options.outputDim + options.tileN - 1) / options.tileN,
  height: (options.rows + options.tileM - 1) / options.tileM,
  depth: 1)
let threads = MTLSize(
  width: pipeline.threadExecutionWidth * options.simdgroups, height: 1, depth: 1)

func execute() -> Double {
  memset(outputBuffer.contents(), 0, outputBuffer.length)
  guard let commandBuffer = commandQueue.makeCommandBuffer(),
    let encoder = commandBuffer.makeComputeCommandEncoder()
  else { fail("Metal command encoding failed") }
  encoder.setComputePipelineState(pipeline)
  encoder.setBuffer(sourceBuffer, offset: 0, index: 0)
  encoder.setBuffer(weightBuffer, offset: 0, index: 1)
  encoder.setBuffer(outputBuffer, offset: 0, index: 2)
  encoder.setBytes(&m, length: MemoryLayout<UInt32>.stride, index: 3)
  encoder.setBytes(&n, length: MemoryLayout<UInt32>.stride, index: 4)
  encoder.setBytes(&k, length: MemoryLayout<UInt32>.stride, index: 5)
  encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threads)
  encoder.endEncoding()
  let started = monotonicSeconds()
  commandBuffer.commit()
  commandBuffer.waitUntilCompleted()
  if let error = commandBuffer.error { fail("Metal execution failed: \(error)") }
  return monotonicSeconds() - started
}

let firstExecutionSeconds = execute()
for _ in 0..<options.warmup { _ = execute() }
var samples: [Double] = []
for _ in 0..<options.iterations { samples.append(execute()) }

let outputBits = outputBuffer.contents().bindMemory(to: UInt16.self, capacity: outputElements)[0]
let outputSample = bfloat16ToFloat(outputBits)
let expectedSample = Float(options.inputDim) * 0.015625 * 0.03125
let medianSeconds = median(samples)
let operationCount = 2.0 * Double(options.rows) * Double(options.inputDim) * Double(options.outputDim)
let result: [String: Any] = [
  "backend": "mpp",
  "device": device.name,
  "dtype": "bf16",
  "rows": options.rows,
  "input_dim": options.inputDim,
  "output_dim": options.outputDim,
  "tile_m": options.tileM,
  "tile_n": options.tileN,
  "simdgroups": options.simdgroups,
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
