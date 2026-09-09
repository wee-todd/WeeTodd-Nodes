import CoreGraphics
import Foundation
import GRPCImageServiceModels
import ImageIO
import NNC
import UniformTypeIdentifiers

// File names are generated here, never accepted from a remote response.
final class ArtifactWriter {
  let root: URL
  let requestID: String
  let operation: String
  let expectedFrames: Int
  let fps: Int
  let sampleRate: Int
  let requiresAudio: Bool
  private(set) var frameCount = 0
  private var dimensions: [Int]?
  var width: Int? { dimensions?[2] }
  var height: Int? { dimensions?[1] }
  private var sampleCount = 0
  private var channels = 0

  init(root: URL, requestID: String, operation: String, expectedFrames: Int,
       fps: Int, sampleRate: Int, requiresAudio: Bool) throws {
    guard root.isFileURL, !requestID.isEmpty, ["image", "video"].contains(operation),
      expectedFrames > 0, expectedFrames <= 100000, fps > 0,
      !requiresAudio || sampleRate > 0,
      !FileManager.default.fileExists(atPath: root.path) else { throw TransportError.invalidRequest }
    self.root = root; self.requestID = requestID; self.operation = operation
    self.expectedFrames = expectedFrames; self.fps = fps; self.sampleRate = sampleRate
    self.requiresAudio = requiresAudio
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
    try FileManager.default.createDirectory(at: root.appendingPathComponent("frames"),
                                             withIntermediateDirectories: false)
  }

  func image(_ tensor: Tensor<Float>) throws {
    let shape = Array(tensor.shape)
    guard shape.count == 4, shape[0] == 1, shape[1] > 0, shape[2] > 0, shape[3] == 3,
      shape[1] <= 16384, shape[2] <= 16384, frameCount < expectedFrames,
      dimensions == nil || dimensions == shape else { throw TransportError.invalidMedia }
    dimensions = shape
    let height = shape[1], width = shape[2]
    var bytes = [UInt8](repeating: 255, count: height * width * 4)
    for y in 0..<height { for x in 0..<width { for c in 0..<3 {
      let value = tensor[0,y,x,c]
      guard value.isFinite else { throw TransportError.invalidMedia }
      bytes[(y*width+x)*4+c] = UInt8(max(0, min(255, (Double(value)+1)*127.5)))
    } } }
    guard let provider = CGDataProvider(data: Data(bytes) as CFData),
      let image = CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32,
        bytesPerRow: width*4, space: CGColorSpace(name: CGColorSpace.sRGB)!,
        bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
        provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent),
      let destination = CGImageDestinationCreateWithURL(
        root.appendingPathComponent(String(format: "frames/%08d.png", frameCount)) as CFURL,
        UTType.png.identifier as CFString, 1, nil) else { throw TransportError.invalidMedia }
    CGImageDestinationAddImage(destination, image, nil)
    guard CGImageDestinationFinalize(destination) else { throw TransportError.invalidMedia }
    frameCount += 1
  }

  func audio(_ tensor: Tensor<Float>) throws {
    let shape = Array(tensor.shape)
    guard operation == "video", sampleRate > 0, sampleCount == 0,
      shape.count == 2, (1...2).contains(shape[0]), shape[1] > 0,
      shape[1] <= 100000000 else { throw TransportError.invalidMedia }
    channels = shape[0]; sampleCount = shape[1]
    let count = sampleCount * channels * 4
    guard count <= Int(UInt32.max)-36 else { throw TransportError.invalidMedia }
    var header = Data("RIFF".utf8)
    func append<T: FixedWidthInteger>(_ value: T) {
      var little = value.littleEndian
      withUnsafeBytes(of: &little) { header.append(contentsOf: $0) }
    }
    append(UInt32(count+36)); header.append(Data("WAVEfmt ".utf8)); append(UInt32(16))
    append(UInt16(3)); append(UInt16(channels)); append(UInt32(sampleRate))
    append(UInt32(sampleRate*channels*4)); append(UInt16(channels*4)); append(UInt16(32))
    header.append(Data("data".utf8)); append(UInt32(count))
    let file = root.appendingPathComponent("audio.wav")
    try header.write(to: file, options: .withoutOverwriting)
    let output = try FileHandle(forWritingTo: file)
    defer { try? output.close() }
    try output.seekToEnd()
    var buffer = Data()
    for i in 0..<sampleCount {
      for c in 0..<channels {
        let value = tensor[c,i]
        guard value.isFinite else { throw TransportError.invalidMedia }
        var bits = value.bitPattern.littleEndian
        withUnsafeBytes(of: &bits) { buffer.append(contentsOf: $0) }
      }
      if buffer.count >= 65536 { try output.write(contentsOf: buffer); buffer.removeAll(keepingCapacity: true) }
    }
    try output.write(contentsOf: buffer)
  }

  func finish(configuration: [String: Any]) throws -> [String: Any] {
    guard frameCount == expectedFrames, !requiresAudio || sampleCount > 0 else {
      throw TransportError.invalidMedia
    }
    if sampleCount > 0 {
      let discrepancy = abs(Double(sampleCount)/Double(sampleRate) - Double(frameCount)/Double(fps))
      guard discrepancy <= 1/Double(fps) + 0.01 else { throw TransportError.invalidMedia }
    }
    var manifest: [String: Any] = [
      "schema": "weetodd-drawthings-media-v1", "requestID": requestID,
      "operation": operation, "frameCount": frameCount, "configuration": configuration
    ]
    if operation == "image" {
      manifest["imagePaths"] = (0..<frameCount).map { String(format: "frames/%08d.png", $0) }
    } else {
      manifest["framesDirectory"] = "frames"
      manifest["fpsNumerator"] = fps; manifest["fpsDenominator"] = 1
      manifest["requiresAudio"] = requiresAudio
      if sampleCount > 0 {
        manifest["audioPath"] = "audio.wav"; manifest["sampleRate"] = sampleRate
        manifest["sampleCount"] = sampleCount; manifest["channels"] = channels
      }
    }
    try JSONSerialization.data(withJSONObject: manifest, options: [.sortedKeys])
      .write(to: root.appendingPathComponent("manifest.json"), options: .atomic)
    return manifest
  }
}

// A failed or incomplete tensor is fatal. Keep image and audio continuation buffers separate.
// Completed frames are written immediately, avoiding accumulation of the whole movie in RAM.
final class TensorStream {
  private let writer: ArtifactWriter
  private var imageChunk = Data()
  private var audioChunk = Data()
  private let maximumTensorBytes = 512 * 1024 * 1024
  init(writer: ArtifactWriter) { self.writer = writer }

  private func decode(_ chunks: [Data], pending: inout Data, more: Bool,
                      consume: (Tensor<Float>) throws -> Void) throws {
    guard !chunks.isEmpty else { return }
    guard !more || chunks.count == 1 else { throw TransportError.invalidMedia }
    for (index, chunk) in chunks.enumerated() {
      guard chunk.count + pending.count <= maximumTensorBytes else { throw TransportError.invalidMedia }
      var data = chunk
      if index == 0, !pending.isEmpty { data = pending + chunk; pending.removeAll() }
      if more { pending = data; continue }
      guard let decoded = Tensor<Float>(data: data, using: [.zip, .fpzip]) else {
        throw TransportError.invalidMedia
      }
      try consume(Tensor<Float>(from: decoded))
    }
  }

  func receive(_ response: ImageGenerationResponse) throws {
    guard response.chunkState == .lastChunk || response.chunkState == .moreChunks else {
      throw TransportError.invalidMedia
    }
    let more = response.chunkState == .moreChunks
    try decode(response.generatedImages, pending: &imageChunk, more: more, consume: writer.image)
    try decode(response.generatedAudio, pending: &audioChunk, more: more, consume: writer.audio)
  }

  func finish(configuration: [String: Any]) throws -> [String: Any] {
    guard imageChunk.isEmpty, audioChunk.isEmpty else { throw TransportError.invalidMedia }
    return try writer.finish(configuration: configuration)
  }
}
