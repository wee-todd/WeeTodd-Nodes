import Foundation
import XCTest

@testable import StudioCore

final class DrawThingsTests: XCTestCase {
  func testDrawThingsStatusAcceptsH3EndpointsButChecksBothAssets() {
    var clip = Clip(engine: .drawThings)
    clip.drawThings = DrawThingsSelection(profileID: "local", modelID: "h3", modelFamily: "minimaxH3")
    let first = MediaAsset(name: "First", kind: .image, path: "/first.png")
    let last = MediaAsset(name: "Last", kind: .image, path: "/last.png")
    clip.attachments = [Attachment(assetID: first.id, role: .first),
                        Attachment(assetID: last.id, role: .last)]
    XCTAssertTrue(clip.drawThingsConditioningIssues(assets: [first, last], fileExists: { _ in true }).isEmpty)
    XCTAssertFalse(clip.drawThingsConditioningIssues(assets: [first], fileExists: { _ in true }).isEmpty)
    XCTAssertFalse(clip.drawThingsConditioningIssues(assets: [first, last], fileExists: { $0 != last.path }).isEmpty)
    clip.drawThings?.modelFamily = "ltx2_3"
    XCTAssertFalse(clip.drawThingsConditioningIssues(assets: [first, last], fileExists: { _ in true }).isEmpty)
    clip.drawThings?.modelFamily = "minimaxH3"
    clip.attachments.removeFirst()
    XCTAssertFalse(clip.drawThingsConditioningIssues(assets: [last], fileExists: { _ in true }).isEmpty)
  }
  func testCompletedEndpointClipKeepsFinalFrameOnTimeline() {
    var clip = Clip(engine: .drawThings)
    clip.duration = 5
    clip.attachments = [Attachment(assetID: UUID(), role: .first),
                        Attachment(assetID: UUID(), role: .last)]
    clip.applyDrawThingsEndpointDuration(124.0 / 24)
    XCTAssertEqual(clip.duration, 124.0 / 24)
    clip.applyDrawThingsEndpointDuration(.nan)
    XCTAssertEqual(clip.duration, 124.0 / 24)
    clip.attachments.removeLast()
    clip.applyDrawThingsEndpointDuration(9)
    XCTAssertEqual(clip.duration, 124.0 / 24, "First-only clips retain their chosen trim")
  }
  func testDrawThingsInfersImageAndEndpointTasksFromAttachments() {
    var clip = Clip(engine: .drawThings)
    clip.attachments = [Attachment(assetID: UUID(), role: .first)]
    XCTAssertEqual(clip.inferredTask, "i2v")
    clip.attachments.append(Attachment(assetID: UUID(), role: .last))
    XCTAssertEqual(clip.inferredTask, "fflf")
    clip.generationSelection = GenerationSelection(task: "t2v", preset: .custom)
    XCTAssertEqual(clip.inferredTask, "t2v", "Explicit conflicting choices remain visible to validation")
  }
  func testLegacyProjectDecodesAndKeepsNativeFingerprint() throws {
    let data = Data(Self.legacyProject.utf8)
    let project = try JSONDecoder().decode(StudioProject.self, from: data)
    XCTAssertNil(project.clips[0].drawThings)
    XCTAssertNil(project.assets[0].generation)
    let fingerprint = project.clips[0].generationFingerprint
    XCTAssertEqual(
      fingerprint, "d4c9b4f412b05a407824e587a3af0a4fc4505fdd10db164cab892e895e541c7b")
    let encodedLegacyClip = String(
      decoding: try JSONEncoder().encode(project.clips[0]), as: UTF8.self)
    XCTAssertFalse(encodedLegacyClip.contains("drawThings"))
    let encodedLegacyAsset = String(
      decoding: try JSONEncoder().encode(project.assets[0]), as: UTF8.self)
    XCTAssertFalse(encodedLegacyAsset.contains("generation"))
    let roundTrip = try JSONDecoder().decode(
      StudioProject.self, from: JSONEncoder().encode(project))
    XCTAssertEqual(roundTrip.clips[0].generationFingerprint, fingerprint)
    XCTAssertNil(roundTrip.clips[0].drawThings)
  }

  func testDrawThingsSelectionChangesRemoteFingerprint() {
    var clip = Clip(name: "Remote", engine: .drawThings)
    clip.drawThings = DrawThingsSelection(
      profileID: "drawthings-local", modelID: "model-a", modelFamily: "flux",
      configuration: ["width": .integer(512), "guidance": .number(3.5)])
    let fingerprint = clip.generationFingerprint
    clip.drawThings?.modelID = "model-b"
    XCTAssertNotEqual(clip.generationFingerprint, fingerprint)
  }

  func testDrawThingsLoRAsRoundTripAndChangeFingerprint() throws {
    var clip = Clip(name: "Remote", engine: .drawThings)
    clip.drawThings = DrawThingsSelection(
      profileID: "local", modelID: "video-a", modelFamily: "ltx2.3",
      loras: [DrawThingsLoRA(modelID: "style-a", weight: 0.75)])
    let fingerprint = clip.generationFingerprint
    clip.drawThings?.loras[0].weight = 1.25
    XCTAssertNotEqual(clip.generationFingerprint, fingerprint)
    let restored = try JSONDecoder().decode(Clip.self, from: JSONEncoder().encode(clip))
    XCTAssertEqual(restored.drawThings?.loras, clip.drawThings?.loras)
  }

  func testLegacyDrawThingsSelectionDefaultsToNoLoRAs() throws {
    let data = #"{"profileID":"local","modelID":"video-a","modelFamily":"ltx2.3","configuration":{}}"#.data(using: .utf8)!
    XCTAssertEqual(try JSONDecoder().decode(DrawThingsSelection.self, from: data).loras, [])
  }

  func testUnavailableLoRAsRemainVisibleAfterModelOrProfileCatalogChanges() {
    let selection = DrawThingsSelection(
      profileID: "server-b", modelID: "video-b", modelFamily: "ltx2.3",
      loras: [
        DrawThingsLoRA(modelID: "still-compatible", weight: 0.8),
        DrawThingsLoRA(modelID: "old-server-style", weight: 1.25),
      ])
    XCTAssertEqual(
      selection.unavailableLoRAs(availableIDs: Set(["still-compatible"])),
      [DrawThingsLoRA(modelID: "old-server-style", weight: 1.25)])
    XCTAssertEqual(selection.unavailableLoRAs(availableIDs: []), selection.loras)
    XCTAssertEqual(selection.unavailableLoRAs(availableIDs: nil), [])
  }

  func testDrawThingsLoRAGroupValidatesIdentityAndCopiesValues() throws {
    let group = DrawThingsLoRAGroup(name: "Look", profileID: "local", family: "ltx2.3",
      compatibleModelIDs: ["video-a"], members: [DrawThingsLoRA(modelID: "style-a", weight: 0.8)])
    try group.validate(profileID: "local", family: "ltx2.3", modelID: "video-a")
    var selection = DrawThingsSelection(profileID: "local", modelID: "video-a", modelFamily: "ltx2.3")
    selection.apply(group)
    XCTAssertEqual(selection.loras, [DrawThingsLoRA(modelID: "style-a", weight: 0.8)])
    var edited = group
    edited.members[0].weight = 1.5
    XCTAssertEqual(selection.loras[0].weight, 0.8)
    XCTAssertThrowsError(try group.validate(profileID: "other", family: "ltx2.3", modelID: "video-a"))
  }

  func testImageGenerationProvenanceRoundTrips() throws {
    var asset = MediaAsset(name: "Generated", kind: .image)
    asset.generation = ImageGeneration(
      provider: "drawThings", requestFingerprint: "abc123", modelID: "model-a",
      prompt: "A cube")
    let restored = try JSONDecoder().decode(MediaAsset.self, from: JSONEncoder().encode(asset))
    XCTAssertEqual(restored, asset)
  }

  private static let legacyProject = #"""
    {"version":1,"id":"3BB99B76-1581-4EB3-BC9B-C49271467B27","name":"Legacy","settings":{"fps":24,"interpolatedFPS":48,"width":1920,"height":1080,"upscaleWidth":3840,"upscaleHeight":2160,"interpolation":"off","upscaling":"off","format":"mp4","rifeScale":1,"fit":"fit","quality":18},"clips":[{"id":"04CB1835-372A-4E04-BBD9-0500523028DC","name":"Native","engine":"ltx25","profileID":"auto","prompt":"Legacy prompt","soundscape":"Natural location sound. No dialogue.","music":"N/A","negativePrompt":"","duration":5,"sourceIn":0,"sourcePath":"","seed":42,"generationWidth":768,"generationHeight":448,"attachments":[],"versions":[],"transition":"cut","transitionDuration":0.5,"volume":1,"extensionDirection":"","extensionSource":"","depthDirectory":"","motionDirectory":"","renderedSignature":"","validatedSignature":""}],"assets":[{"id":"5740C19A-B338-4B7C-AB8C-B30FB00249B5","name":"Still","kind":"image","path":"still.png","scope":"project","duration":0,"width":0,"height":0,"fps":0,"thumbnail":"","text":""}],"titles":[],"audio":[],"audioTracks":[]}
    """#
}
