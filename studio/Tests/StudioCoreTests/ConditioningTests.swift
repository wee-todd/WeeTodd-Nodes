import XCTest
@testable import StudioCore

final class ConditioningTests: XCTestCase {
  func testLegacyAttachmentDefaultsAndMSRControlsRoundTrip() throws {
    let original = Attachment(assetID: UUID(), role: .reference)
    let encoder = JSONEncoder()
    let legacy = try encoder.encode(original)
    var decoded = try JSONDecoder().decode(Attachment.self, from: legacy)
    XCTAssertNil(decoded.referenceRole)
    XCTAssertNil(decoded.referencePriority)
    XCTAssertNil(decoded.referenceFrames)
    XCTAssertNil(decoded.referenceSizePolicy)
    XCTAssertNil(decoded.attentionStrength)
    decoded.referenceRole = "clothing"
    decoded.referencePriority = "supporting"
    decoded.referenceFrames = "33"
    decoded.referenceSizePolicy = "quality"
    decoded.attentionStrength = 0.4
    XCTAssertEqual(try JSONDecoder().decode(Attachment.self, from: encoder.encode(decoded)), decoded)
  }

  func testH3CacheBudgetIsOptionalAndPersistsPerClip() throws {
    let encoder = JSONEncoder()
    var clip = Clip(engine: .h3)
    XCTAssertNil(try JSONDecoder().decode(Clip.self, from: encoder.encode(clip)).h3PagingCacheGB)
    clip.h3PagingCacheGB = 8
    XCTAssertEqual(try JSONDecoder().decode(Clip.self, from: encoder.encode(clip)).h3PagingCacheGB, 8)
  }
}
