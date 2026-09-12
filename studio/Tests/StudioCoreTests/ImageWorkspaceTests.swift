import XCTest
@testable import StudioCore

final class ImageWorkspaceTests: XCTestCase {
  func testZeroStrengthReferenceDoesNotRequireTheLinkedFile() throws {
    var draft = DrawThingsImageDraft(destination: ImageAssetDestination(scope: .global, projectID: UUID()))
    draft.moodboard = [ImageWorkspaceInput(path: "/missing-reference.png")]
    draft.moodboard[0].strength = 0
    XCTAssertEqual((try draft.request(id: "zero")["inputs"] as? [[String: Any]])?.count, 0)
  }
  func testRequestPreservesCanvasReferencesAndSkipsDisabledInputs() throws {
    let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try Data("image".utf8).write(to: url)
    defer { try? FileManager.default.removeItem(at: url) }
    var draft = DrawThingsImageDraft(destination: ImageAssetDestination(scope: .global, projectID: UUID()))
    draft.canvas = ImageWorkspaceInput(path: url.path)
    draft.moodboard = [ImageWorkspaceInput(path: url.path), ImageWorkspaceInput(path: url.path)]
    draft.moodboard[0].strength = 0.4
    draft.moodboard[1].enabled = false
    draft.strength = 0.25
    draft.sampler = 17
    draft.shift = 2
    draft.loras = [DrawThingsLoRA(modelID: "style", weight: 0.6)]
    let request = try draft.request(id: "test")
    let inputs = try XCTUnwrap(request["inputs"] as? [[String: Any]])
    XCTAssertEqual(inputs.compactMap { $0["role"] as? String }, ["canvas", "moodboard"])
    XCTAssertEqual(inputs[1]["strength"] as? Double, 0.4)
    XCTAssertEqual((inputs[0]["sha256"] as? String)?.count, 64)
    XCTAssertEqual((request["configuration"] as? [String: Any])?["strength"] as? Double, 0.25)
    draft.canvas?.enabled = false
    XCTAssertEqual((try draft.request(id: "test")["inputs"] as? [[String: Any]])?.count, 1)
    XCTAssertThrowsError(try {
      var invalid = draft; invalid.moodboard[0].strength = .nan
      _ = try invalid.request(id: "bad")
    }())
  }
}
