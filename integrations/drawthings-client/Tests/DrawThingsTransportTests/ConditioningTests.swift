import CryptoKit
import Foundation
import NNC
import XCTest
@testable import DrawThingsTransport

final class ConditioningTests: XCTestCase {
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
