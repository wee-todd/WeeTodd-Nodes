import XCTest
@testable import StudioCore

final class RenderReportingTests: XCTestCase {
  func testSplitJSONAndUTF8LinesAreReassembled() throws {
    var stream = BridgeProgressStream()
    let data = Data("log output\n{\"event\":\"progress\",\"message\":\"Encoding café\",\"fraction\":0.25}\n".utf8)
    var events: [BridgeProgressEvent] = []
    for byte in data { events += stream.append(Data([byte])) }
    XCTAssertEqual(events.count, 1)
    XCTAssertEqual(events.first?.message, "Encoding café")
    XCTAssertEqual(events.first?.fraction, 0.25)
  }
  func testOversizedLogAndMalformedProgressCannotBreakNextEvent() {
    var stream = BridgeProgressStream()
    XCTAssertTrue(stream.append(Data(repeating: 65, count: 100000)).isEmpty)
    let events = stream.append(Data("\n{\"event\":\"progress\",\"message\":\"Sampling 2/16\",\"fraction\":0.125}\n".utf8))
    XCTAssertEqual(events.first?.message, "Sampling 2/16")
    XCTAssertTrue(stream.append(Data("{\"event\":\"progress\",\"message\":\"invalid\",\"fraction\":8}\n".utf8)).isEmpty)
  }
  func testStatsDistinguishWallTimeAndMemoryScopes() throws {
    let stats = RenderStats(result: ["seconds": 1750.0, "process_peak_rss_bytes": 8_000_000_000,
      "sampling_seconds": 1628.0, "metadata": ["phase_memory": ["run_peak_bytes": 6_240_000_000],
      "mlx_process_peak_bytes": 99]])
    XCTAssertEqual(stats?.elapsedSeconds, 1750)
    XCTAssertEqual(stats?.samplingSeconds, 1628)
    XCTAssertEqual(stats?.processPeakBytes, 8_000_000_000)
    XCTAssertEqual(stats?.mlxPeakBytes, 6_240_000_000)
    XCTAssertEqual(stats?.mlxPeakScope, "Instrumented stages")
    XCTAssertEqual(try JSONDecoder().decode(RenderStats.self, from: JSONEncoder().encode(stats!)), stats)
  }
  func testLTXStatsAndMissingOrInvalidMetrics() {
    let stats = RenderStats(result: ["metadata": ["total_seconds": 164.8,
      "mlx_peak_bytes": 9_200_000_000, "stage_timings": ["sampling_total_seconds": 148.9]]])
    XCTAssertEqual(stats?.elapsedSeconds, 164.8)
    XCTAssertEqual(stats?.mlxPeakScope, "MLX generation")
    XCTAssertEqual(stats?.samplingSeconds, 148.9)
    XCTAssertNil(stats?.processPeakBytes)
    XCTAssertNil(RenderStats(result: [:]))
    XCTAssertNil(RenderStats(result: ["seconds": Double.nan, "process_peak_rss_bytes": -1]))
  }
  func testOldRenderVersionsDecodeWithoutStats() throws {
    let version = RenderVersion(path: "/tmp/render.mp4", seed: 1, prompt: "test", recipePath: "recipe.json")
    var object = try JSONSerialization.jsonObject(with: JSONEncoder().encode(version)) as! [String: Any]
    object.removeValue(forKey: "stats")
    let restored = try JSONDecoder().decode(RenderVersion.self, from: JSONSerialization.data(withJSONObject: object))
    XCTAssertNil(restored.stats)
  }
}
