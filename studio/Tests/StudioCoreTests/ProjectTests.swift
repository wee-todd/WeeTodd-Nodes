import XCTest

@testable import StudioCore

final class ProjectTests: XCTestCase {
  func testSplitPreservesSourceTimingAndIdentity() throws {
    var p = StudioProject()
    var c = Clip(name: "Source", engine: .movie)
    c.sourcePath = "/tmp/source.mov"
    c.sourceIn = 2
    c.duration = 8
    p.clips = [c]
    let id = try p.split(c.id, at: 3)
    XCTAssertEqual(p.clips.count, 2)
    XCTAssertEqual(p.clips[0].duration, 3)
    XCTAssertEqual(p.clips[1].sourceIn, 5)
    XCTAssertEqual(p.clips[1].duration, 5)
    XCTAssertEqual(p.clips[1].id, id)
    XCTAssertEqual(p.duration, 8)
  }
  func testSplitRejectsDraftAndEndpoint() {
    var p = StudioProject()
    let c = Clip()
    p.clips = [c]
    XCTAssertThrowsError(try p.split(c.id, at: 2))
    p.clips[0].sourcePath = "/tmp/example.mov"
    XCTAssertThrowsError(try p.split(c.id, at: 0))
    XCTAssertThrowsError(try p.split(c.id, at: 5))
  }
  func testTransitionTimelineAccounting() {
    var p = StudioProject()
    var a = Clip()
    a.duration = 4
    var b = Clip()
    b.duration = 3
    b.transition = "dissolve"
    b.transitionDuration = 0.5
    p.clips = [a, b]
    XCTAssertEqual(p.start(of: 1), 3.5)
    XCTAssertEqual(p.duration, 6.5)
  }
  func testMovieSettingsInheritanceAndOverride() {
    var p = StudioProject()
    var c = Clip()
    p.settings.width = 1280
    XCTAssertEqual(c.settings(in: p).width, 1280)
    var own = MovieSettings()
    own.width = 640
    c.settingsOverride = own
    p.settings.width = 1920
    XCTAssertEqual(c.settings(in: p).width, 640)
    c.settingsOverride = nil
    XCTAssertEqual(c.settings(in: p).width, 1920)
  }
  func testInterpolationRejectsUnrepresentableRate() {
    var s = MovieSettings()
    s.interpolation = .rife
    s.interpolatedFPS = 60
    XCTAssertThrowsError(try s.validate())
    s.interpolatedFPS = 48
    XCTAssertNoThrow(try s.validate())
  }
  func testKeyframesAndAudioInferTasks() {
    var c = Clip()
    XCTAssertEqual(c.inferredTask, "t2v")
    c.attachments = [Attachment(assetID: UUID(), role: .first)]
    XCTAssertEqual(c.inferredTask, "fflf")
    c.attachments.append(Attachment(assetID: UUID(), role: .audioDriver))
    XCTAssertEqual(c.inferredTask, "a2v")
  }
  func testFingerprintOnlyTracksGenerationChanges() {
    var c = Clip()
    let original = c.generationFingerprint
    c.name = "Renamed"
    c.sourcePath = "/tmp/render.mp4"
    c.transition = "dissolve"
    c.volume = 0
    XCTAssertEqual(original, c.generationFingerprint)
    c.prompt = "A new shot"
    XCTAssertNotEqual(original, c.generationFingerprint)
  }
  func testAssetScopesAndAudioTracksRoundTrip() throws {
    var p = StudioProject()
    let c = Clip()
    p.clips = [c]
    p.assets = [
      MediaAsset(name: "Reference", kind: .image, path: "/tmp/ref.png", scope: .clip, owner: c.id)
    ]
    p.audioTracks.append(AudioTrack(name: "Effects"))
    p.audioTracks[1].replacesSource = true
    let folder = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: folder) }
    let file = folder.appendingPathComponent("test.weetodd")
    try ProjectStorage.write(p, to: file)
    let restored = try ProjectStorage.read(file)
    XCTAssertEqual(p, restored)
  }
  func testReorderPreservesClips() {
    var p = StudioProject()
    let a = Clip(name: "A")
    let b = Clip(name: "B")
    let c = Clip(name: "C")
    p.clips = [a, b, c]
    p.move(c.id, before: a.id)
    XCTAssertEqual(p.clips.map(\.name), ["C", "A", "B"])
    p.move(c.id, before: nil)
    XCTAssertEqual(p.clips.map(\.name), ["A", "B", "C"])
  }
  func testCollectedProjectResolvesMediaAfterMovingFolder() throws {
    var p = StudioProject()
    var c = Clip(name: "Portable", engine: .movie)
    c.sourcePath = "Media/movie.mov"
    p.clips = [c]
    p.assets = [
      MediaAsset(name: "Movie", kind: .video, path: c.sourcePath, scope: .clip, owner: c.id)
    ]
    let folder = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    let moved = folder.appendingPathComponent("Moved")
    try FileManager.default.createDirectory(at: moved, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: folder) }
    let original = folder.appendingPathComponent("Movie.weetodd")
    try ProjectStorage.write(p, to: original)
    let target = moved.appendingPathComponent("Movie.weetodd")
    try FileManager.default.moveItem(at: original, to: target)
    let loaded = try ProjectStorage.read(target)
    XCTAssertEqual(loaded.clips[0].sourcePath, moved.appendingPathComponent("Media/movie.mov").path)
    XCTAssertEqual(loaded.assets[0].path, loaded.clips[0].sourcePath)
  }

}
