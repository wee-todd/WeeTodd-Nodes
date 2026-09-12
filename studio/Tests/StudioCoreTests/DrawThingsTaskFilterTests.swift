import XCTest
@testable import StudioCore

final class DrawThingsTaskFilterTests: XCTestCase {
  func testTaskFiltersUseExactAdvertisedImageCombinations() {
    let rules: [String: Any] = [
      "h3": ["operations": ["video": ["inputRoleCombinations": [[], ["first"], ["first", "last"]]]]],
      "ltx": ["operations": ["video": ["inputRoleCombinations": [[], ["first"]]]]],
      "textOnly": ["operations": ["video": ["inputRoleCombinations": [[]]]]],
      "unknown": ["operations": ["video": [:]]],
      "imageOnly": ["operations": ["image": ["inputRoleCombinations": [[]]]]],
    ]
    XCTAssertEqual(DrawThingsTaskFilter.modelIDs(in: rules, task: "t2v"), Set(["h3", "ltx", "textOnly"]))
    XCTAssertEqual(DrawThingsTaskFilter.modelIDs(in: rules, task: "i2v"), Set(["h3", "ltx"]))
    XCTAssertEqual(DrawThingsTaskFilter.modelIDs(in: rules, task: "fflf"), Set(["h3"]))
    XCTAssertTrue(DrawThingsTaskFilter.modelIDs(in: rules, task: "unsupported").isEmpty)
    XCTAssertTrue(DrawThingsTaskFilter.modelIDs(in: [:], task: "fflf").isEmpty)
  }
}
