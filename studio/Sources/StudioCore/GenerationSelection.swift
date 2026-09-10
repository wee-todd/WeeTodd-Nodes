import Foundation

public enum GenerationPreset: String, Codable, CaseIterable, Identifiable {
  case balanced, speed, lowMemory, custom
  public var id: String { rawValue }
  public var label: String {
    switch self {
    case .balanced: return "Balanced"
    case .speed: return "Speed"
    case .lowMemory: return "Low memory"
    case .custom: return "Custom"
    }
  }
}

/// Explicit user intent. Nil on older clips preserves their exact recipe behavior.
public struct GenerationSelection: Codable, Equatable {
  public var task: String
  public var preset: GenerationPreset
  public var steps: Int?
  public var refinementSteps: Int?
  public var cfg: Double?
  public var shift: Double?
  public var memoryPolicy: String?
  public var projectionBackend: String?
  public init(task: String = "t2v", preset: GenerationPreset = .balanced) {
    self.task = task
    self.preset = preset
  }
  public var isModified: Bool {
    steps != nil || refinementSteps != nil || cfg != nil || shift != nil
      || memoryPolicy != nil || projectionBackend != nil
  }
  public mutating func resetOverrides() {
    steps = nil; refinementSteps = nil; cfg = nil; shift = nil
    memoryPolicy = nil; projectionBackend = nil
  }
  public static func taskLabel(_ task: String) -> String {
    switch task {
    case "t2v": return "Text to video"
    case "i2v": return "Image to video"
    case "fflf": return "First and last frames"
    case "ref2va": return "Reference video"
    case "a2v": return "Audio-driven video"
    case "control": return "Controlled video"
    case "extension": return "Video extension"
    default: return task
    }
  }
}

public struct GenerationControls: Codable, Equatable {
  public var evaluations: Int?
  public var refinementSteps: Int?
  public var cfg: Double?
  public var shift: Double?
  public var stepsEditable: Bool
  public var refinementStepsEditable: Bool
  public var cfgEditable: Bool
  public var shiftEditable: Bool
  public var stepsExplanation: String
  public var cfgExplanation: String
  public var shiftExplanation: String
}
public struct GenerationDescriptor: Codable, Equatable {
  public struct Preset: Codable, Equatable, Identifiable {
    public var id: String
    public var name: String
    public var description: String
  }
  public var supportedTasks: [String]
  public var controls: GenerationControls
  public var presets: [Preset]
}

public struct AccelerationSettings: Codable, Equatable {
  public var h3MemoryPolicy: String = "automatic"
  public var h3ProjectionBackend: String = "auto"
  public init() {}
}

extension GenerationSelection {
  /// Asset records can be relinked or reclassified without changing the clip's attachments.
  public static func assetFingerprint(for clip: Clip, assets: [MediaAsset]) -> String {
    let referenced = Set(clip.attachments.map(\.assetID))
    let records = assets.filter { referenced.contains($0.id) }
      .sorted { $0.id.uuidString < $1.id.uuidString }
    let encoder = JSONEncoder()
    encoder.outputFormatting = .sortedKeys
    return ((try? encoder.encode(records)) ?? Data()).base64EncodedString()
  }
}

extension AccelerationSettings {
  public static func memoryPolicyLabel(_ policy: String) -> String {
    switch policy {
    case "automatic": return "Automatic"
    case "recipe": return "Recipe default"
    case "paged": return "Paged · lower memory"
    case "pagedNormal": return "Paged · larger workspace"
    case "resident": return "Resident · experimental high RAM"
    default: return policy
    }
  }
}
