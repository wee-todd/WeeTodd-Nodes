import XCTest
@testable import StudioCore

final class BackgroundCredentialTests: XCTestCase {
  @MainActor func testPermissionWaitDoesNotRunOnMainThread() async throws {
    let value = try await BackgroundCredential.read {
      XCTAssertFalse(Thread.isMainThread)
      return "test-only"
    }
    XCTAssertEqual(value, "test-only")
    let missing = try await BackgroundCredential.read { nil }
    XCTAssertNil(missing)
  }

  func testKeychainErrorIsReturned() async {
    do {
      _ = try await BackgroundCredential.read { throw StudioError.invalid("denied") }
      XCTFail("Expected the credential error")
    } catch { XCTAssertTrue(error.localizedDescription.contains("denied")) }
  }
}
