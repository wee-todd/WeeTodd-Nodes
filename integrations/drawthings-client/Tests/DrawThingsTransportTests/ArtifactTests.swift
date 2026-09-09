import Foundation
import GRPCImageServiceModels
import ImageIO
import NNC
import XCTest
@testable import DrawThingsTransport

final class ArtifactTests: XCTestCase {
  func tensor(_ value: Float) -> Tensor<Float> {
    var tensor = Tensor<Float>(.CPU, .NHWC(1, 2, 2, 3))
    for y in 0..<2 { for x in 0..<2 { for c in 0..<3 { tensor[0,y,x,c] = value } } }
    return tensor
  }
  func testChunkedFramesAndAudioProduceMeasuredArtifacts() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: root) }
    let writer = try ArtifactWriter(root: root, requestID: "fixture", operation: "video",
                                    expectedFrames: 2, fps: 2, sampleRate: 8, requiresAudio: true)
    let stream = TensorStream(writer: writer)
    let frame = tensor(0.5).data(using: [.zip, .fpzip])
    let mid = frame.count / 2
    try stream.receive(ImageGenerationResponse.with {
      $0.generatedImages = [frame.prefix(mid)]; $0.chunkState = .moreChunks
    })
    try stream.receive(ImageGenerationResponse.with {
      $0.generatedImages = [frame.suffix(frame.count-mid), frame]; $0.chunkState = .lastChunk
    })
    var tone = Tensor<Float>(.CPU, .NC(2, 8))
    for c in 0..<2 { for i in 0..<8 { tone[c,i] = Float(i) / 8 } }
    let encoded = tone.data(using: [.zip, .fpzip]); let split = encoded.count / 2
    try stream.receive(ImageGenerationResponse.with {
      $0.generatedAudio = [encoded.prefix(split)]; $0.chunkState = .moreChunks
    })
    try stream.receive(ImageGenerationResponse.with {
      $0.generatedAudio = [encoded.suffix(encoded.count-split)]; $0.chunkState = .lastChunk
    })
    let manifest = try stream.finish(configuration: ["seed": 42])
    XCTAssertEqual(manifest["frameCount"] as? Int, 2)
    XCTAssertEqual(manifest["sampleCount"] as? Int, 8)
    let source = CGImageSourceCreateWithURL(root.appendingPathComponent("frames/00000000.png") as CFURL, nil)!
    XCTAssertEqual(CGImageSourceCreateImageAtIndex(source, 0, nil)?.width, 2)
    let wav = try Data(contentsOf: root.appendingPathComponent("audio.wav"))
    XCTAssertEqual(String(data: wav.prefix(4), encoding: .ascii), "RIFF")
    XCTAssertEqual(wav.count, 44 + 8 * 2 * 4)
  }
  func testMalformedAndIncompleteTensorsCannotComplete() throws {
    for malformed in [false, true] {
      let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
      defer { try? FileManager.default.removeItem(at: root) }
      let writer = try ArtifactWriter(root: root, requestID: "fixture", operation: "image",
                                      expectedFrames: 1, fps: 1, sampleRate: 0, requiresAudio: false)
      let stream = TensorStream(writer: writer)
      let frame = tensor(0.5).data(using: [.zip, .fpzip])
      if malformed {
        XCTAssertThrowsError(try stream.receive(ImageGenerationResponse.with { $0.generatedImages = [Data([1,2,3])] }))
      } else {
        try stream.receive(ImageGenerationResponse.with {
          $0.generatedImages = [frame.prefix(20)]; $0.chunkState = .moreChunks
        })
        XCTAssertThrowsError(try stream.finish(configuration: [:]))
      }
      XCTAssertFalse(FileManager.default.fileExists(atPath: root.appendingPathComponent("manifest.json").path))
    }
  }
  func testMissingRequiredAudioAndExtraFramesAreErrors() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: root) }
    let writer = try ArtifactWriter(root: root, requestID: "fixture", operation: "video",
                                    expectedFrames: 1, fps: 24, sampleRate: 48000, requiresAudio: true)
    try writer.image(tensor(0.5))
    XCTAssertThrowsError(try writer.finish(configuration: [:]))
    XCTAssertThrowsError(try writer.image(tensor(0.5)))
  }
  func testExistingOutputDirectoryCannotBeOverwritten() throws {
    XCTAssertThrowsError(try ArtifactWriter(root: FileManager.default.temporaryDirectory,
      requestID: "../escape", operation: "image", expectedFrames: 1, fps: 1, sampleRate: 0, requiresAudio: false))
  }
}
