import XCTest

@testable import StudioCore

final class ModelSetupTests: XCTestCase {
  let presetJSON =
    #"{"id":"ltx25-t2v","name":"LTX 2.5 Text to Video","engine":"ltx25","task":"t2v","description":"Create a shot","components":[{"key":"checkpoint","label":"Transformer","kind":"file"},{"key":"text_encoder","label":"Text encoder","kind":"directory"}]}"#

  func testCatalogContractDecodesAndRequiresEveryComponent() throws {
    let preset = try JSONDecoder().decode(ModelSetupPreset.self, from: Data(presetJSON.utf8))
    var selection = ModelSetupSelection()
    XCTAssertEqual(
      selection.missingComponents(for: preset).map(\.key), ["checkpoint", "text_encoder"])
    selection.components = ["checkpoint": "/models/checkpoint", "text_encoder": "  "]
    XCTAssertEqual(selection.missingComponents(for: preset).map(\.key), ["text_encoder"])
    selection.components["text_encoder"] = "/models/encoder"
    XCTAssertTrue(selection.missingComponents(for: preset).isEmpty)
  }

  func testScanAutoSelectsOnlyUniqueCandidatesAndKeepsExplicitChoice() {
    var selection = ModelSetupSelection()
    selection.applyScan(["checkpoint": ["/a", "/b"], "encoder": ["/encoder", "/encoder"]])
    XCTAssertNil(selection.components["checkpoint"])
    XCTAssertEqual(selection.components["encoder"], "/encoder")
    selection.components["checkpoint"] = "/manual"
    selection.applyScan(["checkpoint": ["/other"], "encoder": []])
    XCTAssertEqual(selection.components["checkpoint"], "/manual")
    XCTAssertEqual(selection.components["encoder"], "/encoder")
  }

  func testDownloadsOnlyAppearForCompatibleEngineAndLegacyCatalogStillDecodes() throws {
    let json =
      #"{"id":"encoder","name":"Encoder","description":"Prepare encoder","downloadBytes":10,"requiredDiskBytes":20,"sourceURL":"https://example.org/model","licenseURL":"https://example.org/license","outputKind":"directory","engines":["h3"],"licenseNotice":"Review source terms"}"#
    let download = try JSONDecoder().decode(ModelSetupDownload.self, from: Data(json.utf8))
    XCTAssertTrue(download.supports(engine: "h3"))
    XCTAssertFalse(download.supports(engine: "ltx25"))
    XCTAssertEqual(download.licenseNotice, "Review source terms")
    let legacy = json.replacingOccurrences(
      of: #","engines":["h3"],"licenseNotice":"Review source terms""#, with: "")
    let legacyDownload = try JSONDecoder().decode(ModelSetupDownload.self, from: Data(legacy.utf8))
    XCTAssertTrue(legacyDownload.supports(engine: "ltx25"))
  }

  func testRecipeSelectionChecksEngineAndCurrentMediaRoles() throws {
    var preset = try JSONDecoder().decode(ModelSetupPreset.self, from: Data(presetJSON.utf8))
    var clip = Clip(engine: .ltx25)
    XCTAssertTrue(preset.supports(clip))
    clip.engine = .h3
    XCTAssertFalse(preset.supports(clip))
    preset.engine = "h3"
    XCTAssertTrue(preset.supports(clip))
    clip.attachments = [Attachment(assetID: UUID(), role: .first)]
    XCTAssertFalse(preset.supports(clip))
    preset.task = "fflf"
    XCTAssertTrue(preset.supports(clip))
    clip.attachments = [Attachment(assetID: UUID(), role: .reference)]
    XCTAssertFalse(preset.supports(clip))
  }

  func testH3DownloadsFilterByTaskAndSharedComponentsRemainVisible() throws {
    let json =
      #"{"id":"ref","name":"Reference transformer","description":"Prepared pages","downloadBytes":10,"requiredDiskBytes":20,"sourceURL":"https://example.org/model","licenseURL":"https://example.org/license","outputKind":"directory","engines":["h3"],"tasks":["ref2va"]}"#
    let download = try JSONDecoder().decode(ModelSetupDownload.self, from: Data(json.utf8))
    XCTAssertTrue(download.supports(engine: "h3", task: "ref2va"))
    XCTAssertFalse(download.supports(engine: "h3", task: "t2v"))
    XCTAssertFalse(download.supports(engine: "h3", task: "fflf"))
    XCTAssertFalse(download.supports(engine: "ltx25", task: "ref2va"))
    let sharedJSON = json.replacingOccurrences(of: #","tasks":["ref2va"]"#, with: "")
    let shared = try JSONDecoder().decode(ModelSetupDownload.self, from: Data(sharedJSON.utf8))
    XCTAssertTrue(shared.supports(engine: "h3", task: "fflf"))
  }

  func testComponentDownloadButtonsOnlySelectPackagesProvidingThatComponent() throws {
    let json =
      #"{"id":"support","name":"Support files","description":"Audio and task files","downloadBytes":10,"requiredDiskBytes":20,"sourceURL":"https://example.org/model","licenseURL":"https://example.org/license","outputKind":"directory","engines":["h3"],"tasks":["t2v","fflf"],"components":["checkpoint","audio_vae","tokenizer","processor"]}"#
    let download = try JSONDecoder().decode(ModelSetupDownload.self, from: Data(json.utf8))
    XCTAssertTrue(download.supports(engine: "h3", task: "fflf", component: "audio_vae"))
    XCTAssertFalse(download.supports(engine: "h3", task: "ref2va", component: "audio_vae"))
    XCTAssertFalse(download.supports(engine: "h3", task: "t2v", component: "transformer"))
    let legacyJSON = json.replacingOccurrences(
      of: #","components":["checkpoint","audio_vae","tokenizer","processor"]"#, with: "")
    let legacy = try JSONDecoder().decode(ModelSetupDownload.self, from: Data(legacyJSON.utf8))
    XCTAssertTrue(legacy.supports(engine: "h3", task: "t2v"))
    XCTAssertFalse(legacy.supports(engine: "h3", task: "t2v", component: "audio_vae"))
  }

  func testLTX25BaseRecipeSupportsExistingImageAndAudioTaskFallbacks() throws {
    let preset = try JSONDecoder().decode(ModelSetupPreset.self, from: Data(presetJSON.utf8))
    for role in [MediaRole.first, .audioDriver] {
      var clip = Clip(engine: .ltx25)
      clip.attachments = [Attachment(assetID: UUID(), role: role)]
      XCTAssertTrue(preset.supports(clip))
    }
    var referenceClip = Clip(engine: .ltx25)
    referenceClip.attachments = [Attachment(assetID: UUID(), role: .reference)]
    XCTAssertFalse(preset.supports(referenceClip))
  }

  func testMemoryModesUseBackendIdentifiers() {
    XCTAssertEqual(
      ModelSetupMemoryMode.allCases.map(\.rawValue), ["automatic", "lower_memory", "custom"])
  }
}
