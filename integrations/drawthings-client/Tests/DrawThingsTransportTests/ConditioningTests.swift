import CryptoKit
import Foundation
import GRPCImageServiceModels
import NNC
import XCTest
@testable import DrawThingsTransport

final class ConditioningTests: XCTestCase {
  func testWirePayloadKeepsFirstImageAndLastShuffleDistinct() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: root) }
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
    var inputs = [[String: Any]]()
    for (index, role) in ["first", "last"].enumerated() {
      let folder = root.appendingPathComponent(role)
      let writer = try ArtifactWriter(root: folder, requestID: role, operation: "image",
        expectedFrames: 1, fps: 1, sampleRate: 0, requiresAudio: false)
      var original = Tensor<Float>(.CPU, .NHWC(1, 64, 64, 3))
      for y in 0..<64 { for x in 0..<64 { for c in 0..<3 {
        original[0,y,x,c] = c == index * 2 ? 1 : -1
      } } }
      try writer.image(original)
      let url = folder.appendingPathComponent("frames/00000000.png")
      let hash = SHA256.hash(data: try Data(contentsOf: url)).map { String(format: "%02x", $0) }.joined()
      inputs.append(["role": role, "frameIndex": index == 0 ? 0 : 123, "strength": 1,
                     "path": url.path, "sha256": hash])
    }
    let request: [String: Any] = ["operation": "video", "configuration": ["numFrames": 124], "inputs": inputs]
    var payload = ImageGenerationRequest()
    try Conditioning.apply(request, to: &payload, width: 64, height: 64)
    let wire = try ImageGenerationRequest(serializedBytes: payload.serializedData())
    let first = try XCTUnwrap(Tensor<Float>(data: wire.image, using: [.zip, .fpzip]))
    XCTAssertEqual(wire.hints.count, 1)
    XCTAssertEqual(wire.hints[0].hintType, "shuffle")
    XCTAssertEqual(wire.hints[0].tensors.count, 1)
    XCTAssertEqual(wire.hints[0].tensors[0].weight, 1)
    let last = try XCTUnwrap(Tensor<Float>(data: wire.hints[0].tensors[0].tensor, using: [.zip, .fpzip]))
    XCTAssertEqual(first[0,0,0,0], 1, accuracy: 0.02)
    XCTAssertEqual(first[0,0,0,2], -1, accuracy: 0.02)
    XCTAssertEqual(last[0,0,0,2], 1, accuracy: 0.02)
    XCTAssertEqual(last[0,0,0,0], -1, accuracy: 0.02)
    let canvas: [String: Any] = ["role": "canvas", "path": inputs[0]["path"]!,
      "sha256": inputs[0]["sha256"]!, "strength": 1, "fit": "fit"]
    let reference: [String: Any] = ["role": "moodboard", "path": inputs[1]["path"]!,
      "sha256": inputs[1]["sha256"]!, "strength": 0.35, "fit": "fit"]
    let imageRequest: [String: Any] = ["operation": "image", "inputs": [canvas, reference, reference]]
    var imagePayload = ImageGenerationRequest()
    try Conditioning.apply(imageRequest, to: &imagePayload, width: 64, height: 64)
    XCTAssertFalse(imagePayload.image.isEmpty)
    XCTAssertEqual(imagePayload.hints[0].tensors.count, 2)
    XCTAssertEqual(imagePayload.hints[0].tensors[0].weight, 0.35, accuracy: 0.001)
    var boardOnly = ImageGenerationRequest()
    try Conditioning.apply(["operation": "image", "inputs": [reference]], to: &boardOnly, width: 64, height: 64)
    XCTAssertTrue(boardOnly.image.isEmpty)
    XCTAssertEqual(boardOnly.hints[0].tensors.count, 1)
    try Data("stale".utf8).write(to: URL(fileURLWithPath: inputs[1]["path"] as! String))
    XCTAssertThrowsError(try Conditioning.apply(request, to: &payload, width: 64, height: 64))
  }
  func testFirstLastContractUsesTheResolvedFinalFrame() throws {
    let first: [String: Any] = ["role": "first", "frameIndex": 0, "strength": 1,
      "path": "/first.png", "sha256": String(repeating: "a", count: 64)]
    var last = first; last["role"] = "last"; last["frameIndex"] = 123
    let request: [String: Any] = ["operation": "video", "configuration": ["numFrames": 124],
      "inputs": [first, last]]
    XCTAssertEqual(try Conditioning.inputs(request).count, 2)
    var invalid = request; last["frameIndex"] = 120; invalid["inputs"] = [first, last]
    XCTAssertThrowsError(try Conditioning.inputs(invalid))
    invalid["inputs"] = [first, first]
    XCTAssertThrowsError(try Conditioning.inputs(invalid))
    invalid["inputs"] = [last]
    XCTAssertThrowsError(try Conditioning.inputs(invalid))
  }
  func testFirstFramePreservesPixelsAndRejectsChangedFile() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: root) }
    let writer = try ArtifactWriter(root: root, requestID: "first", operation: "image",
      expectedFrames: 1, fps: 1, sampleRate: 0, requiresAudio: false)
    var original = Tensor<Float>(.CPU, .NHWC(1, 64, 64, 3))
    for y in 0..<64 { for x in 0..<64 {
      original[0,y,x,0] = y < 32 ? 1 : -1
      original[0,y,x,1] = -1
      original[0,y,x,2] = y < 32 ? -1 : 1
    } }
    try writer.image(original)
    let image = root.appendingPathComponent("frames/00000000.png")
    let digest = SHA256.hash(data: try Data(contentsOf: image)).map { String(format: "%02x", $0) }.joined()
    let input: [String: Any] = ["role": "first", "frameIndex": 0, "strength": 1,
      "path": image.path, "sha256": digest]
    let request: [String: Any] = ["operation": "video", "inputs": [input]]
    let encoded = try XCTUnwrap(Conditioning.image(request, width: 64, height: 64))
    let tensor = try XCTUnwrap(Tensor<Float>(data: encoded, using: [.zip, .fpzip]))
    XCTAssertEqual(tensor[0,0,0,0], 1, accuracy: 0.02)
    XCTAssertEqual(tensor[0,63,0,2], 1, accuracy: 0.02)
    try Data("changed".utf8).write(to: image)
    XCTAssertThrowsError(try Conditioning.image(request, width: 64, height: 64))
  }
  func testUnsupportedRolesAndLoRAAvailabilityCannotSilentlyFallback() throws {
    for role in ["reference", "last", "keyframe", "audioDriver", "control"] {
      XCTAssertThrowsError(try Conditioning.inputs(["operation": "video", "inputs": [["role": role]]]))
    }
    let request: [String: Any] = ["modelID": "video", "loras": [["modelID": "style", "weight": 0.6]]]
    let values = try Conditioning.loras(request)
    XCTAssertEqual(values.first?.weight, 0.6)
    XCTAssertThrowsError(try Conditioning.validateAvailability(request, catalog: ["files": ["style"]]))
    XCTAssertNoThrow(try Conditioning.validateAvailability(request, catalog: ["files": ["style"],
      "loras": [["id": "style", "compatibleModelIDs": ["video"]]]]))
    XCTAssertThrowsError(try Conditioning.loras(["loras": [["modelID": "style", "weight": true]]]))
    XCTAssertThrowsError(try Conditioning.loras(["loras": [["modelID": "style", "weight": 1], ["modelID": "style", "weight": 1]]]))
  }
}
