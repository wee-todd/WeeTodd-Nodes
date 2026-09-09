import XCTest

@testable import StudioCore

final class ModelDownloadTokenTests: XCTestCase {
  func testSavedTokenOverridesOnlyHFTokenInChildEnvironment() throws {
    let environment = ["HF_TOKEN": "existing-login", "OTHER_SETTING": "preserved"]
    let result = try ModelDownloadToken.environment(environment, savedToken: "  hf_test_token\n")
    XCTAssertEqual(result["HF_TOKEN"], "hf_test_token")
    XCTAssertEqual(result["OTHER_SETTING"], "preserved")
    XCTAssertEqual(environment["HF_TOKEN"], "existing-login")
  }

  func testMissingSavedTokenPreservesExistingLoginEnvironment() throws {
    let environment = ["HF_TOKEN": "existing-login"]
    XCTAssertEqual(try ModelDownloadToken.environment(environment, savedToken: nil), environment)
    XCTAssertNil(try ModelDownloadToken.environment([:], savedToken: nil)["HF_TOKEN"])
  }

  func testEmptyOrMultilineTokenIsRejectedWithoutEchoingInput() {
    for token in ["  \n", "hf_private\nsecond-line", "hf_private\u{0}value"] {
      XCTAssertThrowsError(try ModelDownloadToken.normalized(token)) { error in
        XCTAssertFalse(error.localizedDescription.contains("hf_private"))
      }
    }
  }
}
