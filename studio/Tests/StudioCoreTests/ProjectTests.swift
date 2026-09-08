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

extension ProjectTests {
  func testMotionSettingsDoNotInvalidateBaseGenerationAndOldClipsDecode() throws {
    var clip = Clip(name: "H3", engine: .h3)
    let oldData = try JSONEncoder().encode(clip)
    let fingerprint = clip.generationFingerprint
    clip.motionFidelity = MotionFidelitySettings()
    clip.motionFidelity!.enabled = true
    clip.motionFidelity!.strength = 0.7
    clip.motionRecipeID = "repair.json"
    XCTAssertEqual(clip.generationFingerprint, fingerprint)
    XCTAssertFalse(clip.motionIsCurrent)
    let restored = try JSONDecoder().decode(Clip.self, from: oldData)
    XCTAssertNil(restored.motionFidelity)
    XCTAssertNil(restored.motionResult)
    XCTAssertEqual(restored.playbackPath, restored.sourcePath)
    XCTAssertEqual(restored.playbackIn, restored.sourceIn)
    let roundTrip = try JSONDecoder().decode(Clip.self, from: JSONEncoder().encode(clip))
    XCTAssertEqual(roundTrip.motionFidelity, clip.motionFidelity)
  }
}

extension ProjectTests {
  func testEnhancedPlaybackAndPortablePathsRejectStaleInputs() throws {
    let folder = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: folder) }
    let source = folder.appendingPathComponent("source.mp4")
    let output = folder.appendingPathComponent("enhanced.mp4")
    let recipe = folder.appendingPathComponent("repair.json")
    try Data("source".utf8).write(to: source)
    try Data("enhanced".utf8).write(to: output)
    try Data("{}".utf8).write(to: recipe)
    var clip = Clip(name: "Motion", engine: .h3)
    clip.sourcePath = source.path
    clip.sourceIn = 2
    clip.duration = 3
    clip.motionFidelity = MotionFidelitySettings()
    clip.motionFidelity!.enabled = true
    func modified(_ url: URL) throws -> Double {
      let a = try FileManager.default.attributesOfItem(atPath: url.path)
      return (a[.modificationDate] as! Date).timeIntervalSince1970
    }
    let values: [String: Any] = [
      "path": output.path, "sourcePath": source.path, "sourceIn": 2, "duration": 3,
      "outputIn": 0, "recipeID": "", "sourceSHA256": "source-hash", "sha256": "output-hash",
      "settings": try JSONSerialization.jsonObject(with: JSONEncoder().encode(clip.motionFidelity)),
      "sourceSize": 6, "outputSize": 8, "sourceModified": try modified(source),
      "recipePath": recipe.path, "recipeSize": 2, "recipeModified": try modified(recipe),
      "outputModified": try modified(output), "report": folder.appendingPathComponent("plan.json").path
    ]
    clip.motionResult = try JSONDecoder().decode(MotionFidelityResult.self,
      from: JSONSerialization.data(withJSONObject: values))
    XCTAssertTrue(clip.motionIsCurrent)
    XCTAssertEqual(clip.playbackPath, output.path)
    XCTAssertEqual(clip.playbackIn, 0)
    clip.motionFidelity!.enabled = false
    XCTAssertEqual(clip.playbackPath, source.path)
    XCTAssertEqual(clip.playbackIn, 2)
    clip.motionFidelity!.enabled = true
    var project = StudioProject()
    project.clips = [clip]
    ProjectStorage.mapPaths(&project) { "Media/" + URL(fileURLWithPath: $0).lastPathComponent }
    XCTAssertEqual(project.clips[0].motionResult?.path, "Media/enhanced.mp4")
    XCTAssertEqual(project.clips[0].motionResult?.sourcePath, "Media/source.mp4")
    XCTAssertEqual(project.clips[0].motionResult?.recipePath, "Media/repair.json")
    let copiedRecipe = folder.appendingPathComponent("collected-repair.json")
    try FileManager.default.copyItem(at: recipe, to: copiedRecipe)
    clip.motionResult!.recipePath = copiedRecipe.path
    try FileManager.default.removeItem(at: recipe)
    XCTAssertTrue(clip.motionIsCurrent)
    clip.duration = 2.5
    XCTAssertFalse(clip.motionIsCurrent)
    clip.duration = 3
    try Data("changed source".utf8).write(to: source)
    XCTAssertFalse(clip.motionIsCurrent)
  }
}
