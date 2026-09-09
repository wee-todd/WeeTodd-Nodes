import Foundation
import XCTest

@testable import StudioCore

final class DrawThingsTests: XCTestCase {
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
