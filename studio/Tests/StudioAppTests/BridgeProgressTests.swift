import XCTest
import Combine
import StudioCore
@testable import WeeToddStudio

final class BridgeProgressTests: XCTestCase {
  @MainActor func testLiveChildProgressAndCompletedResponseAreBothDelivered() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: root) }
    try FileManager.default.createDirectory(at: root.appendingPathComponent("scripts"), withIntermediateDirectories: true)
    let script = """
    import sys, time
    from pathlib import Path
    sys.stdout.write('{"event":"progress","message":"Sampling ')
    sys.stdout.flush()
    sys.stdout.write('3/16","fraction":0.1875}\\n')
    sys.stdout.flush()
    deadline = time.monotonic() + 10
    while not (Path(__file__).parents[1] / 'release').exists():
        if time.monotonic() > deadline: raise RuntimeError('test handshake timed out')
        time.sleep(0.01)
    print('{"status":"success","result":{"fixture":true}}', flush=True)
    """
    try script.write(to: root.appendingPathComponent("scripts/studio_bridge.py"), atomically: true, encoding: .utf8)
    let bridge = Bridge()
    let settings = RuntimeSettings(root: root.path, pythonPath: "/usr/bin/python3", profilesDirectory: root.path)
    let delivered = expectation(description: "Progress arrives before the child completes")
    let subscription = bridge.$message.sink { if $0 == "Sampling 3/16" { delivered.fulfill() } }
    defer { subscription.cancel() }
    let invocation = Task { try await bridge.invoke("render", runtime: settings, payload: [:]) }
    await fulfillment(of: [delivered], timeout: 5)
    XCTAssertTrue(bridge.busy)
    XCTAssertEqual(bridge.message, "Sampling 3/16")
    XCTAssertEqual(bridge.fraction, 0.1875)
    try Data().write(to: root.appendingPathComponent("release"))
    let result = try await invocation.value
    XCTAssertEqual(result["fixture"] as? Bool, true)
    XCTAssertNotNil(bridge.startedAt)
    XCTAssertNotNil(bridge.lastOutputAt)
    XCTAssertFalse(bridge.busy)
  }
}
