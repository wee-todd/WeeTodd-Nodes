import Foundation
import XCTest
@testable import DrawThingsTransport

final class ConfigurationTests: XCTestCase {
  var request: [String: Any] {
    ["modelID": "flux_2_klein_4b_q8p.ckpt", "operation": "image",
     "configuration": ["width": 512, "height": 512, "steps": 4, "seed": 42],
     "inputs": [], "loras": []]
  }

  func testOfflineEstimateUsesCanonicalPixelDimensions() throws {
    let result = try ComputeEstimate.evaluate(request)
    XCTAssertEqual(result["cu"] as? Int, 208)
    let effective = try XCTUnwrap(result["configuration"] as? [String: Any])
    XCTAssertEqual(effective["width"] as? Int, 512)
    XCTAssertEqual(effective["steps"] as? Int, 4)
    XCTAssertEqual(effective["seed"] as? Int, 42)
  }

  func testRejectsUnrepresentableDimensionsAndBooleans() {
    for width: Any in [513, 0, -64, true, 512.5, 4194304] {
      var value = request
      value["configuration"] = ["width": width, "height": 512, "steps": 4, "seed": 42]
      XCTAssertThrowsError(try ComputeEstimate.evaluate(value))
    }
  }

  func testUnknownModelDoesNotBecomeDefaultModel() {
    var value = request
    value["modelID"] = "unrecognized-model"
    XCTAssertThrowsError(try ComputeEstimate.evaluate(value))
  }

  func testUnknownConfigurationCannotBeSilentlyIgnored() {
    var value = request
    value["configuration"] = ["width": 512, "height": 512, "mysteryOption": 1]
    XCTAssertThrowsError(try ComputeEstimate.evaluate(value))
  }

  func testConditioningNeedsAnImplementedMapping() {
    var value = request
    value["inputs"] = [["role": "reference", "path": "image.png"]]
    XCTAssertThrowsError(try ComputeEstimate.evaluate(value))
    value = request
    value["loras"] = [["modelID": "style", "strength": 0.5]]
    XCTAssertThrowsError(try ComputeEstimate.evaluate(value))
  }

  func testVideoRequiresExplicitTiming() {
    var value = request
    value["operation"] = "video"
    XCTAssertThrowsError(try ComputeEstimate.evaluate(value))
  }
}
