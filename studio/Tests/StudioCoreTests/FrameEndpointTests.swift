import XCTest
@testable import StudioCore

final class FrameEndpointTests: XCTestCase {
  func testEndpointCapabilitiesRespectEngineModelAndRecipe() {
    for engine in [Engine.h3, .ltx23, .ltx25] {
      let clip = Clip(engine: engine)
      XCTAssertTrue(clip.supportsEndpoint(.first))
      XCTAssertTrue(clip.supportsEndpoint(.last))
      XCTAssertFalse(clip.supportsEndpoint(.last, supportedTasks: ["t2v"]))
    }
    XCTAssertFalse(Clip(engine: .movie).supportsEndpoint(.first))
    var remote = Clip(engine: .drawThings)
    XCTAssertTrue(remote.supportsEndpoint(.first))
    XCTAssertTrue(remote.supportsEndpoint(.last))
    remote.drawThings = DrawThingsSelection(profileID: "local", modelID: "h3", modelFamily: "minimaxH3")
    XCTAssertTrue(remote.supportsEndpoint(.first))
    XCTAssertTrue(remote.supportsEndpoint(.last))
    remote.drawThings?.modelFamily = "ltx2_3"
    XCTAssertTrue(remote.supportsEndpoint(.first))
    XCTAssertFalse(remote.supportsEndpoint(.last))
  }

  func testDrawThingsEndpointsCanBeAuthoredBeforeChoosingModel() throws {
    var clip = Clip(engine: .drawThings)
    clip.generationSelection = GenerationSelection(task: "fflf", preset: .custom)
    let first = MediaAsset(name: "First", kind: .image, path: "/first.png")
    let last = MediaAsset(name: "Last", kind: .image, path: "/last.png")
    XCTAssertTrue(clip.canAssignDrawThingsInput(first, role: .first))
    try clip.assignEndpoint(first, role: .first, fps: 24)
    XCTAssertEqual(clip.inferredTask, "fflf")
    clip.drawThings = DrawThingsSelection(profileID: "local", modelID: "", modelFamily: "")
    try clip.assignEndpoint(last, role: .last, fps: 24)
    XCTAssertEqual(clip.inferredTask, "fflf")
    let authored = clip.attachments
    let pending = clip.drawThingsConditioningIssues(assets: [first, last], fileExists: { _ in true })
    XCTAssertTrue(pending.contains { $0.contains("Choose a Draw Things video model") })
    XCTAssertFalse(pending.contains { $0.contains("Remove") })
    clip.drawThings = DrawThingsSelection(profileID: "local", modelID: "h3", modelFamily: "minimaxH3")
    XCTAssertTrue(clip.drawThingsConditioningIssues(assets: [first, last], fileExists: { _ in true }).isEmpty)
    clip.drawThings = DrawThingsSelection(profileID: "local", modelID: "ltx", modelFamily: "ltx2.3")
    XCTAssertFalse(clip.drawThingsConditioningIssues(assets: [first, last], fileExists: { _ in true }).isEmpty)
    XCTAssertEqual(clip.attachments, authored)
    XCTAssertFalse(clip.supportsEndpoint(.last))
  }

  func testEndpointReplacementKeepsOtherAttachmentsAndSamplingOverrides() throws {
    var clip = Clip(engine: .ltx25)
    clip.generationSelection = GenerationSelection()
    clip.generationSelection?.steps = 12
    let first = MediaAsset(name: "First", kind: .image)
    let last = MediaAsset(name: "Last", kind: .image)
    let replacement = MediaAsset(name: "Replacement", kind: .image)
    try clip.assignEndpoint(first, role: .first, fps: 24)
    XCTAssertEqual(clip.inferredTask, "i2v")
    try clip.assignEndpoint(last, role: .last, fps: 24)
    try clip.assignEndpoint(replacement, role: .first, fps: 24)
    XCTAssertEqual(clip.inferredTask, "fflf")
    XCTAssertEqual(clip.generationSelection?.steps, 12)
    XCTAssertEqual(clip.attachments.count, 2)
    XCTAssertEqual(clip.attachments.first { $0.role == .first }?.assetID, replacement.id)
    XCTAssertEqual(clip.attachments.first { $0.role == .last }?.time ?? 0, 5 - 1.0 / 24, accuracy: 0.0001)
    clip.removeEndpoint(.last)
    XCTAssertEqual(clip.inferredTask, "i2v")
    clip.removeEndpoint(.first)
    XCTAssertEqual(clip.inferredTask, "t2v")
  }

  func testInvalidDropDoesNotMutateClip() {
    var clip = Clip(engine: .movie)
    let original = clip
    XCTAssertThrowsError(try clip.assignEndpoint(MediaAsset(name: "Image", kind: .image), role: .first, fps: 24))
    XCTAssertEqual(clip, original)
    clip.engine = .ltx25
    XCTAssertThrowsError(try clip.assignEndpoint(MediaAsset(name: "Movie", kind: .video), role: .last, fps: 24))
    XCTAssertTrue(clip.attachments.isEmpty)
  }
}
