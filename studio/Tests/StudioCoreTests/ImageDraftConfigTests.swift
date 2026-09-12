import XCTest
@testable import StudioCore

final class ImageDraftConfigTests: XCTestCase {
  func testDraftRecoveryKeepsReferencesSettingsAndDestinationSeparate() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: root) }
    var draft = DrawThingsImageDraft(destination: ImageAssetDestination(scope: .clip, projectID: UUID(), owner: UUID()))
    draft.prompt = "unfinished prompt"
    draft.canvas = ImageWorkspaceInput(path: "/missing-linked-image.png")
    draft.moodboard = [ImageWorkspaceInput(path: "/reference.png")]
    draft.moodboard[0].strength = 0.4
    draft.loras = [DrawThingsLoRA(modelID: "style.ckpt", weight: 0.6)]
    var library = ImageWorkspaceLibrary()
    library.record(draft, preview: "/result.png")
    try library.write(to: root.appendingPathComponent("drafts.json"))
    let restored = try ImageWorkspaceLibrary.read(from: root.appendingPathComponent("drafts.json"))
    XCTAssertEqual(restored.sessions[draft.destination.storageKey]?.draft, draft)
    XCTAssertEqual(restored.sessions[draft.destination.storageKey]?.previewPath, "/result.png")
    XCTAssertNotEqual(draft.destination.storageKey, ImageAssetDestination(scope: .project, projectID: draft.destination.projectID).storageKey)
  }

  func testDrawThingsConfigAliasesAndUnsupportedSettingsAreVisible() throws {
    let json = #"{"name":"My preset","configuration":{"model":"h3.ckpt","width":768,"height":448,"steps":4,"seed":42,"sampler":10,"shiftForAudio":3,"fpsId":24,"numFrames":124,"loras":[{"file":"turbo.ckpt","weight":0.6}],"hiresFix":true},"prompt":"saved prompt"}"#
    let result = try DrawThingsConfigImport.parse(Data(json.utf8), operation: "video")[0]
    XCTAssertEqual(result.modelID, "h3.ckpt")
    XCTAssertEqual(result.configuration["audioShift"], .number(3))
    XCTAssertEqual(result.configuration["fps"], .integer(24))
    XCTAssertEqual(result.loras?.first?.weight, 0.6)
    XCTAssertEqual(result.prompt, "saved prompt")
    XCTAssertTrue(result.warnings.contains { $0.contains("hiresFix") })
  }

  func testConfigImportRejectsMalformedValuesAndPreservesOmittedFields() throws {
    for json in [#"{"steps":true}"#, #"{"strength":70}"#, #"{"width":513}"#, #"{"sampler":127}"#,
      #"{"model":"one","modelID":"two"}"#] {
      XCTAssertThrowsError(try DrawThingsConfigImport.parse(Data(json.utf8), operation: "image"))
    }
    let result = try DrawThingsConfigImport.parse(Data(#"{"steps":8,"loras":[]}"#.utf8), operation: "image")[0]
    var draft = DrawThingsImageDraft(destination: ImageAssetDestination(scope: .global, projectID: UUID()))
    draft.prompt = "keep"; draft.width = 1024; draft.loras = [DrawThingsLoRA(modelID: "old")]
    result.apply(to: &draft, includePrompt: true)
    XCTAssertEqual(draft.steps, 8); XCTAssertEqual(draft.width, 1024)
    XCTAssertEqual(draft.prompt, "keep"); XCTAssertEqual(draft.loras, [])
  }

  func testNegativePromptOnlyConfigIsUseful() throws {
    let result = try DrawThingsConfigImport.parse(Data(#"{"negativePrompt":"blur"}"#.utf8), operation: "image")[0]
    XCTAssertEqual(result.negativePrompt, "blur")
  }
}
