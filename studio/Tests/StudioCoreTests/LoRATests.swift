import XCTest

@testable import StudioCore

final class LoRATests: XCTestCase {
  func asset(_ model: LoRAModel, path: String = "/tmp/style.safetensors") -> MediaAsset {
    var a = MediaAsset(name: "Style", kind: .lora, path: path)
    a.loraModel = model
    return a
  }
  func testCompatibilityMatrix() {
    for engine in Engine.allCases {
      for model in LoRAModel.allCases {
        XCTAssertEqual(
          model.supports(engine),
          model.rawValue == engine.rawValue || (engine == .ltx25 && model == .ltx23))
      }
    }
  }
  func testMixedGroupRequiresLTX25() throws {
    let members = [
      LoRAMember(asset: asset(.ltx23)),
      LoRAMember(asset: asset(.ltx25, path: "/tmp/other.safetensors"), strength: 0.6),
    ]
    XCTAssertNoThrow(try LoRAGroup(name: "Mixed", engine: .ltx25, members: members).validate())
    XCTAssertThrowsError(try LoRAGroup(name: "Wrong", engine: .ltx23, members: members).validate())
    XCTAssertThrowsError(try LoRAGroup(name: "Empty", engine: .h3, members: []).validate())
  }
  func testApplicationSnapshotsAndRejectsDuplicatesAtomically() throws {
    var p = StudioProject()
    let c = Clip(engine: .ltx25)
    p.clips = [c]
    var g = LoRAGroup(
      name: "Style group", engine: .ltx25,
      members: [LoRAMember(asset: asset(.ltx23), strength: 0.7)])
    try p.applyLoRAs(g.members, to: c.id, groupName: g.name)
    let applied = p
    g.members[0].strength = 1.8
    g.members[0].asset.path = "/tmp/replacement.safetensors"
    XCTAssertEqual(p.clips[0].attachments[0].strength, 0.7)
    XCTAssertEqual(p.clips[0].attachments[0].loraGroupName, "Style group")
    XCTAssertEqual(p.assets[0].scope, .clip)
    XCTAssertEqual(p.assets[0].owner, c.id)
    XCTAssertEqual(p.assets[0].path, "/tmp/style.safetensors")
    XCTAssertThrowsError(try p.applyLoRAs([LoRAMember(asset: asset(.ltx23))], to: c.id))
    XCTAssertEqual(p, applied)
  }
  func testSplitRetainsIndependentClipAssetLinks() throws {
    var p = StudioProject()
    var c = Clip(engine: .ltx25)
    c.sourcePath = "/tmp/generated.mp4"
    p.clips = [c]
    try p.applyLoRAs([LoRAMember(asset: asset(.ltx23))], to: c.id, groupName: "Style")
    let secondID = try p.split(c.id, at: 2)
    let attachment = p.clips[1].attachments[0]
    XCTAssertNotEqual(attachment.assetID, p.clips[0].attachments[0].assetID)
    XCTAssertEqual(p.assets.first { $0.id == attachment.assetID }?.owner, secondID)
    p.assets.removeAll { $0.owner == c.id }
    XCTAssertNotNil(p.assets.first { $0.id == attachment.assetID })
  }
  func testInvalidMembersAndStrengths() {
    for strength in [Double.nan, .infinity, -0.1, 2.1] {
      XCTAssertThrowsError(try LoRAMember(asset: asset(.h3), strength: strength).validate(for: .h3))
    }
    XCTAssertThrowsError(
      try LoRAMember(asset: MediaAsset(name: "Unknown", kind: .lora)).validate(for: .ltx25))
    XCTAssertThrowsError(
      try LoRAGroup(
        name: "Duplicate", engine: .h3,
        members: [
          LoRAMember(asset: asset(.h3)),
          LoRAMember(asset: asset(.h3, path: "/tmp/./style.safetensors")),
        ]
      ).validate())
  }
  func testLegacyAssetsAndAttachmentsDecodeWithoutNewFields() throws {
    let a = MediaAsset(name: "Old", kind: .lora)
    let data = try JSONEncoder().encode(a)
    XCTAssertNil(try JSONDecoder().decode(MediaAsset.self, from: data).loraModel)
    let attachment = Attachment(assetID: a.id, role: .lora)
    XCTAssertNil(
      try JSONDecoder().decode(Attachment.self, from: JSONEncoder().encode(attachment))
        .loraGroupName)
  }
  func testStrengthChangesGenerationFingerprintAndProjectRoundTrips() throws {
    var p = StudioProject()
    let c = Clip(engine: .ltx25)
    p.clips = [c]
    try p.applyLoRAs([LoRAMember(asset: asset(.ltx25))], to: c.id, groupName: "Test")
    let signature = p.clips[0].generationFingerprint
    p.clips[0].attachments[0].strength = 0.3
    XCTAssertNotEqual(signature, p.clips[0].generationFingerprint)
    XCTAssertEqual(p, try JSONDecoder().decode(StudioProject.self, from: JSONEncoder().encode(p)))
  }
}
