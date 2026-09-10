import XCTest
@testable import StudioCore

final class GenerationSelectionTests: XCTestCase {
  func testLegacyClipPreservesCustomRecipe() throws {
    var clip = Clip(engine: .h3)
    clip.profileID = "/recipes/exact.json"
    var object = try JSONSerialization.jsonObject(with: JSONEncoder().encode(clip)) as! [String: Any]
    object.removeValue(forKey: "generationSelection")
    let decoded = try JSONDecoder().decode(Clip.self, from: JSONSerialization.data(withJSONObject: object))
    XCTAssertNil(decoded.generationSelection)
    XCTAssertEqual(decoded.profileID, clip.profileID)
  }
  func testExplicitTaskPreservesAttachmentsAndReset() throws {
    var clip = Clip(engine: .h3)
    clip.attachments = [Attachment(assetID: UUID(), role: .reference)]
    clip.generationSelection = GenerationSelection(task: "t2v", preset: .balanced)
    XCTAssertEqual(clip.inferredTask, "t2v")
    XCTAssertEqual(clip.attachments.count, 1)
    let fingerprint = clip.generationFingerprint
    clip.generationSelection?.steps = 27
    XCTAssertTrue(clip.generationSelection!.isModified)
    XCTAssertNotEqual(fingerprint, clip.generationFingerprint)
    clip.generationSelection?.resetOverrides()
    XCTAssertEqual(fingerprint, clip.generationFingerprint)
    let decoded = try JSONDecoder().decode(Clip.self, from: JSONEncoder().encode(clip))
    XCTAssertEqual(decoded.generationSelection, clip.generationSelection)
  }
}

extension GenerationSelectionTests {
  func testAccelerationDefaultsAndOverrideRoundTrip() throws {
    var defaults = AccelerationSettings()
    XCTAssertEqual(defaults.h3MemoryPolicy, "automatic")
    XCTAssertEqual(defaults.h3ProjectionBackend, "auto")
    defaults.h3MemoryPolicy = "paged"
    defaults.h3ProjectionBackend = "mlx"
    XCTAssertEqual(try JSONDecoder().decode(AccelerationSettings.self,
      from: JSONEncoder().encode(defaults)), defaults)
  }
}

extension GenerationSelectionTests {
  func testReferencedAssetRelinkAndLoRAMetadataInvalidateResolution() {
    var asset = MediaAsset(name: "Reference", kind: .image, path: "/first.png")
    var clip = Clip(engine: .h3)
    clip.attachments = [Attachment(assetID: asset.id, role: .first)]
    let original = GenerationSelection.assetFingerprint(for: clip, assets: [asset])
    asset.path = "/relinked.png"
    XCTAssertNotEqual(original, GenerationSelection.assetFingerprint(for: clip, assets: [asset]))
    let relinked = GenerationSelection.assetFingerprint(for: clip, assets: [asset])
    asset.loraModel = .ltx23
    XCTAssertNotEqual(relinked, GenerationSelection.assetFingerprint(for: clip, assets: [asset]))
    XCTAssertNotEqual(relinked, GenerationSelection.assetFingerprint(for: clip, assets: []))
  }
}

extension GenerationSelectionTests {
  func testLargerWorkspacePagingSelectionRoundTripAndReset() throws {
    var clip = Clip(engine: .h3)
    clip.generationSelection = GenerationSelection()
    let original = clip.generationFingerprint
    clip.generationSelection?.memoryPolicy = "pagedNormal"
    let restored = try JSONDecoder().decode(Clip.self, from: JSONEncoder().encode(clip))
    XCTAssertEqual(restored.generationSelection?.memoryPolicy, "pagedNormal")
    XCTAssertNotEqual(restored.generationFingerprint, original)
    XCTAssertEqual(AccelerationSettings.memoryPolicyLabel("pagedNormal"), "Paged · larger workspace")
    clip.generationSelection?.resetOverrides()
    XCTAssertEqual(clip.generationFingerprint, original)
  }
}
