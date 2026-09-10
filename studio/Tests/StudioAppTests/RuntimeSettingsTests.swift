import Foundation
import XCTest
@testable import WeeToddStudio

final class RuntimeSettingsTests: XCTestCase {
  private let defaults = RuntimeSettings(
    root: "/new/source", pythonPath: "/new/python", profilesDirectory: "/new/profiles",
    drawThingsHelperPath: "/Applications/Studio.app/Contents/MacOS/WeeToddDrawThings")

  func testLegacySettingsAdoptBundledHelperWithoutReplacingNativeRuntime() throws {
    let legacy = Data("""
      {"root":"/existing/source","pythonPath":"/existing/python",
       "profilesDirectory":"/existing/profiles","ffmpegPath":"/existing/ffmpeg",
       "ffprobePath":"","rifePath":"","rifeWeights":"","metalPath":""}
      """.utf8)
    let result = RuntimeSettings.restoring(legacy, defaults: defaults)
    XCTAssertEqual(result.drawThingsHelperPath, defaults.drawThingsHelperPath)
    XCTAssertEqual(result.root, "/existing/source")
    XCTAssertEqual(result.pythonPath, "/existing/python")
    XCTAssertEqual(result.ffmpegPath, "/existing/ffmpeg")
  }

  func testExplicitImportedHelperIsPreserved() throws {
    var saved = defaults
    saved.drawThingsHelperPath = "/custom/WeeToddDrawThings"
    let result = RuntimeSettings.restoring(try JSONEncoder().encode(saved), defaults: defaults)
    XCTAssertEqual(result.drawThingsHelperPath, saved.drawThingsHelperPath)
  }

  func testEmptyHelperUsesBundleAndMissingBundleRemainsOptional() throws {
    var saved = defaults
    saved.drawThingsHelperPath = ""
    let data = try JSONEncoder().encode(saved)
    XCTAssertEqual(RuntimeSettings.restoring(data, defaults: defaults).drawThingsHelperPath,
                   defaults.drawThingsHelperPath)
    var withoutHelper = defaults
    withoutHelper.drawThingsHelperPath = nil
    XCTAssertNil(RuntimeSettings.restoring(data, defaults: withoutHelper).drawThingsHelperPath)
  }
}
