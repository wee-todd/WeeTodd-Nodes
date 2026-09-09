import CryptoKit
import Foundation

public enum Engine: String, Codable, CaseIterable, Identifiable {
  case h3, ltx23, ltx25, movie
  public var id: String { rawValue }
  public var label: String {
    switch self {
    case .h3: return "MiniMax H3"
    case .ltx23: return "LTX 2.3"
    case .ltx25: return "LTX 2.5"
    case .movie: return "Movie / Still"
    }
  }
}
public enum AssetScope: String, Codable, CaseIterable { case global, project, clip }
public enum AssetKind: String, Codable, CaseIterable {
  case video, image, audio, sequence, text, lora
}
public enum MediaRole: String, Codable, CaseIterable, Identifiable {
  case reference, first, last, keyframe, audioDriver, control, lora
  public var id: String { rawValue }
  public var label: String {
    switch self {
    case .reference: return "Reference"
    case .first: return "First frame"
    case .last: return "Last frame"
    case .keyframe: return "Keyframe"
    case .audioDriver: return "Audio driver"
    case .control: return "Control guide"
    case .lora: return "LoRA"
    }
  }
}
public struct MediaAsset: Codable, Identifiable, Equatable {
  public var id = UUID()
  public var name: String
  public var kind: AssetKind
  public var path: String
  public var scope: AssetScope
  public var owner: UUID?
  public var duration: Double = 0
  public var width: Int = 0
  public var height: Int = 0
  public var fps: Double = 0
  public var thumbnail: String = ""
  public var text: String = ""
  public var loraModel: LoRAModel?
  public init(
    name: String, kind: AssetKind, path: String = "", scope: AssetScope = .project,
    owner: UUID? = nil
  ) {
    self.name = name
    self.kind = kind
    self.path = path
    self.scope = scope
    self.owner = owner
  }
}
public struct Attachment: Codable, Identifiable, Equatable {
  public var id = UUID()
  public var assetID: UUID
  public var role: MediaRole
  public var time: Double = 0
  public var strength: Double = 1
  public var loraGroupID: UUID?
  public var loraGroupName: String?
  public var controlType = "canny_edges"
  public var description = ""
  public init(assetID: UUID, role: MediaRole, time: Double = 0) {
    self.assetID = assetID
    self.role = role
    self.time = time
  }
}
public enum Interpolation: String, Codable, CaseIterable { case off, rife, metalFX }
public enum Upscaling: String, Codable, CaseIterable { case off, lanczos, metalFX }
public enum MovieFormat: String, Codable, CaseIterable { case mp4, mov, proRes, pngSequence }
public struct MovieSettings: Codable, Equatable {
  public var fps: Double = 24
  public var interpolatedFPS: Double = 48
  public var width = 1920
  public var height = 1080
  public var upscaleWidth = 3840
  public var upscaleHeight = 2160
  public var interpolation: Interpolation = .off
  public var upscaling: Upscaling = .off
  public var format: MovieFormat = .mp4
  public var rifeScale: Double = 1
  public var fit = "fit"
  public var quality = 18
  public init() {}
  public var outputFPS: Double { interpolation == .off ? fps : interpolatedFPS }
  public var outputWidth: Int { upscaling == .off ? width : upscaleWidth }
  public var outputHeight: Int { upscaling == .off ? height : upscaleHeight }
  public func validate() throws {
    guard fps.isFinite, fps >= 1, fps <= 120, interpolatedFPS.isFinite,
      interpolatedFPS >= 1, interpolatedFPS <= 240,
      [width, height, upscaleWidth, upscaleHeight].allSatisfy({
        $0 >= 64 && $0 <= 8192 && $0 % 2 == 0
      })
    else {
      throw StudioError.invalid(
        "Choose even movie dimensions from 64 to 8192 and valid frame rates.")
    }
    if interpolation != .off {
      let ratio = interpolatedFPS / fps
      guard ratio >= 2, ratio <= 4, abs(ratio.rounded() - ratio) < 0.0001 else {
        throw StudioError.invalid(
          "Interpolation requires an integer 2×, 3×, or 4× frame-rate multiplier.")
      }
      if interpolation == .metalFX && abs(ratio - 2) > 0.0001 {
        throw StudioError.invalid("MetalFX frame interpolation currently supports 2× only.")
      }
    }
  }
}
public struct RenderVersion: Codable, Identifiable, Equatable {
  public var id = UUID()
  public var path: String
  public var created = Date()
  public var seed: Int
  public var prompt: String
  public var recipePath: String
  public init(path: String, seed: Int, prompt: String, recipePath: String) {
    self.path = path
    self.seed = seed
    self.prompt = prompt
    self.recipePath = recipePath
  }
}
public struct Clip: Codable, Identifiable, Equatable {
  public var id = UUID()
  public var name = "Untitled clip"
  public var engine: Engine = .ltx25
  public var profileID = "auto"
  public var prompt = ""
  public var soundscape = "Natural location sound. No dialogue."
  public var music = "N/A"
  public var negativePrompt = ""
  public var duration: Double = 5
  public var sourceIn: Double = 0
  public var sourcePath = ""
  public var seed = 42
  public var generationWidth = 768
  public var generationHeight = 448
  public var attachments: [Attachment] = []
  public var versions: [RenderVersion] = []
  public var settingsOverride: MovieSettings?
  public var transition = "cut"
  public var transitionDuration: Double = 0.5
  public var volume: Double = 1
  public var extensionDirection = ""
  public var extensionSource = ""
  public var depthDirectory = ""
  public var motionDirectory = ""
  public var motionFidelity: MotionFidelitySettings?
  public var motionResult: MotionFidelityResult?
  public var motionRecipeID: String?
  public var motionPrompt: String?
  public var renderedSignature = ""
  public var validatedSignature = ""
  public var extensionClipID: UUID?
  public init(name: String = "Untitled clip", engine: Engine = .ltx25) {
    self.name = name
    self.engine = engine
  }
  public func settings(in project: StudioProject) -> MovieSettings {
    settingsOverride ?? project.settings
  }
  public var inferredTask: String {
    if !extensionDirection.isEmpty { return "extension" }
    if attachments.contains(where: { $0.role == .control }) { return "control" }
    if attachments.contains(where: { $0.role == .audioDriver }) { return "a2v" }
    if attachments.contains(where: { $0.role == .reference }) { return "ref2va" }
    if attachments.contains(where: { [.first, .last, .keyframe].contains($0.role) }) {
      return "fflf"
    }
    return "t2v"
  }
  public var displayTask: String {
    if engine == .movie { return "Imported media" }
    switch inferredTask {
    case "fflf": return "Image / keyframe video"
    case "ref2va": return "Reference video"
    case "a2v": return "Audio-driven video"
    case "control": return "Controlled video"
    case "extension": return "Video extension"
    default: return "Text to video"
    }
  }
}
public struct TitleOverlay: Codable, Identifiable, Equatable {
  public var id = UUID()
  public var text = "Your title"
  public var start: Double = 0
  public var duration: Double = 3
  public var position = "lower"
  public var fontSize = 56
  public var color = "white"
  public init() {}
}
public struct AudioTrack: Codable, Identifiable, Equatable {
  public var id = UUID()
  public var name = "Music"
  public var muted = false
  public var solo = false
  public var replacesSource = false
  public init(name: String = "Music") { self.name = name }
}
public struct AudioRegion: Codable, Identifiable, Equatable {
  public var id = UUID()
  public var assetID: UUID
  public var trackID: UUID?
  public var path: String
  public var start: Double = 0
  public var sourceIn: Double = 0
  public var duration: Double = 5
  public var volume: Double = 0.8
  public var fade: Double = 0.2
  public init(assetID: UUID, path: String) {
    self.assetID = assetID
    self.path = path
  }
}
public struct StudioProject: Codable, Equatable {
  public var version = 1
  public var id = UUID()
  public var name = "Untitled movie"
  public var settings = MovieSettings()
  public var clips: [Clip] = []
  public var assets: [MediaAsset] = []
  public var titles: [TitleOverlay] = []
  public var audio: [AudioRegion] = []
  public var audioTracks: [AudioTrack] = [AudioTrack()]
  public init() {}
  public var duration: Double {
    clips.enumerated().reduce(0) { value, pair in
      value + pair.element.duration - overlap(before: pair.offset)
    }
  }
  public func overlap(before index: Int) -> Double {
    guard index > 0, index < clips.count, clips[index].transition != "cut" else { return 0 }
    return max(
      0,
      min(clips[index].transitionDuration, clips[index - 1].duration / 2, clips[index].duration / 2)
    )
  }
  public func start(of index: Int) -> Double {
    guard index > 0 else { return 0 }
    return (0..<min(index, clips.count)).reduce(0) { $0 + clips[$1].duration - overlap(before: $1) }
      - overlap(before: index)
  }
  public mutating func split(_ id: UUID, at seconds: Double) throws -> UUID {
    guard let i = clips.firstIndex(where: { $0.id == id }) else {
      throw StudioError.invalid("Select a clip to split.")
    }
    let original = clips[i]
    guard !original.sourcePath.isEmpty, seconds > 0.05, seconds < original.duration - 0.05 else {
      throw StudioError.invalid("Split a rendered or imported clip inside its duration.")
    }
    var second = original
    second.id = UUID()
    second.name += " · B"
    second.sourceIn += seconds
    second.duration -= seconds
    second.transition = "cut"
    // Split edits the existing movie; rerendering remains an explicit version operation.
    second.attachments = original.attachments.filter {
      $0.time >= seconds || $0.role == .reference || $0.role == .lora
    }
    for j in second.attachments.indices where second.attachments[j].role == .keyframe {
      second.attachments[j].time -= seconds
    }
    // Clip-owned links must survive deletion or relinking of either split half.
    var cloned: [UUID: UUID] = [:]
    for j in second.attachments.indices {
      let sourceID = second.attachments[j].assetID
      if let newID = cloned[sourceID] {
        second.attachments[j].assetID = newID
      } else if var linked = assets.first(where: { $0.id == sourceID && $0.scope == .clip }) {
        linked.id = UUID()
        linked.owner = second.id
        assets.append(linked)
        cloned[sourceID] = linked.id
        second.attachments[j].assetID = linked.id
      }
    }
    clips[i].duration = seconds
    clips[i].name += " · A"
    clips[i].attachments.removeAll { $0.role == .keyframe && $0.time >= seconds }
    clips.insert(second, at: i + 1)
    return second.id
  }
  public mutating func move(_ id: UUID, before target: UUID?) {
    guard id != target, let from = clips.firstIndex(where: { $0.id == id }) else { return }
    let clip = clips.remove(at: from)
    let to = target.flatMap { id in clips.firstIndex(where: { $0.id == id }) } ?? clips.count
    clips.insert(clip, at: to)
  }
  public func validate() throws {
    guard version == 1 else {
      throw StudioError.invalid("This project uses an unsupported document version.")
    }
    try settings.validate()
    guard Set(clips.map(\.id)).count == clips.count, Set(assets.map(\.id)).count == assets.count
    else {
      throw StudioError.invalid("Project contains duplicate clip or asset IDs.")
    }
    for c in clips {
      guard c.duration.isFinite, c.duration > 0, c.sourceIn.isFinite, c.sourceIn >= 0 else {
        throw StudioError.invalid(
          "Clip duration must be positive and its in point cannot be negative.")
      }
      try c.settings(in: self).validate()
    }
  }
}
public enum StudioError: LocalizedError {
  case invalid(String)
  public var errorDescription: String? {
    if case .invalid(let message) = self { return message }
    return nil
  }
}
public enum ProjectStorage {
  public static func read(_ url: URL) throws -> StudioProject {
    var p = try JSONDecoder().decode(StudioProject.self, from: Data(contentsOf: url))
    mapPaths(&p) { value in
      guard !value.isEmpty, !value.hasPrefix("/") else { return value }
      return url.deletingLastPathComponent().appendingPathComponent(value).standardizedFileURL.path
    }
    try p.validate()
    return p
  }
  public static func mapPaths(_ project: inout StudioProject, transform: (String) -> String) {
    for i in project.assets.indices {
      project.assets[i].path = transform(project.assets[i].path)
      if project.assets[i].kind == .sequence {
        project.assets[i].text = transform(project.assets[i].text)
      }
    }
    for i in project.clips.indices {
      project.clips[i].sourcePath = transform(project.clips[i].sourcePath)
      if project.clips[i].motionResult != nil {
        project.clips[i].motionResult!.path = transform(project.clips[i].motionResult!.path)
        project.clips[i].motionResult!.sourcePath = transform(
          project.clips[i].motionResult!.sourcePath)
        project.clips[i].motionResult!.report = transform(project.clips[i].motionResult!.report)
        if let recipe = project.clips[i].motionResult!.recipePath {
          project.clips[i].motionResult!.recipePath = transform(recipe)
        }
      }
      project.clips[i].extensionSource = transform(project.clips[i].extensionSource)
      project.clips[i].depthDirectory = transform(project.clips[i].depthDirectory)
      project.clips[i].motionDirectory = transform(project.clips[i].motionDirectory)
      for j in project.clips[i].versions.indices {
        project.clips[i].versions[j].path = transform(project.clips[i].versions[j].path)
        project.clips[i].versions[j].recipePath = transform(project.clips[i].versions[j].recipePath)
      }
    }
    for i in project.audio.indices { project.audio[i].path = transform(project.audio[i].path) }
  }
  public static func write(_ project: StudioProject, to url: URL) throws {
    try project.validate()
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    try encoder.encode(project).write(to: url, options: .atomic)
  }
}

extension Clip {
  public var generationFingerprint: String {
    var c = self
    c.motionFidelity = nil
    c.motionResult = nil
    c.motionRecipeID = nil
    c.motionPrompt = nil
    c.sourcePath = ""
    c.sourceIn = 0
    c.versions = []
    c.renderedSignature = ""
    c.validatedSignature = ""
    c.name = ""
    c.transition = "cut"
    c.transitionDuration = 0
    c.volume = 1
    c.settingsOverride = nil
    c.depthDirectory = ""
    c.motionDirectory = ""
    let encoder = JSONEncoder()
    encoder.outputFormatting = .sortedKeys
    return SHA256.hash(data: (try? encoder.encode(c)) ?? Data()).map { String(format: "%02x", $0) }
      .joined()
  }
}
