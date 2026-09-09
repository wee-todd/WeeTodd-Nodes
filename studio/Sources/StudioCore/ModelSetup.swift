import Foundation

public struct ModelSetupComponent: Codable, Identifiable, Equatable {
  public var key: String
  public var label: String
  public var kind: String
  public var accepts: [String]?
  public var id: String { key }
}

public struct ModelSetupPreset: Codable, Identifiable, Equatable {
  public var id: String
  public var name: String
  public var engine: String
  public var task: String
  public var description: String
  public var components: [ModelSetupComponent]

  public func supports(_ clip: Clip) -> Bool {
    guard engine == clip.engine.rawValue else { return false }
    return task == clip.inferredTask
      || (engine == "ltx25" && task == "t2v" && ["fflf", "a2v"].contains(clip.inferredTask))
  }
}

public enum ModelSetupMemoryMode: String, CaseIterable, Identifiable {
  case automatic
  case lowerMemory = "lower_memory"
  case custom
  public var id: String { rawValue }
  public var label: String {
    switch self {
    case .automatic: return "Automatic"
    case .lowerMemory: return "Lower Memory"
    case .custom: return "Custom"
    }
  }
  public var detail: String {
    switch self {
    case .automatic:
      return
        "Use supported memory-saving options on Macs with 64 GB or less; otherwise keep preset settings. Physical memory guides this policy, not a guaranteed peak-memory estimate."
    case .lowerMemory:
      return "Prefer staged component unloading and the preset’s supported memory-saving options."
    case .custom: return "Keep preset settings; adjust advanced clip settings after setup."
    }
  }
}

public struct ModelSetupSelection {
  public var components: [String: String] = [:]
  public var candidates: [String: [String]] = [:]
  public init() {}

  public mutating func applyScan(_ found: [String: [String]]) {
    candidates = found.mapValues { Array(Set($0)).sorted() }
    for (key, paths) in candidates where paths.count == 1 {
      if components[key, default: ""].trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
        components[key] = paths[0]
      }
    }
  }

  public func missingComponents(for preset: ModelSetupPreset) -> [ModelSetupComponent] {
    preset.components.filter {
      components[$0.key, default: ""].trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }
  }
}

public struct ModelSetupDownload: Codable, Identifiable {
  public var id: String
  public var name: String
  public var description: String
  public var downloadBytes: Int64
  public var requiredDiskBytes: Int64
  public var sourceURL: String
  public var licenseURL: String
  public var outputKind: String
  public var engines: [String]?
  public var licenseNotice: String?

  public func supports(engine: String) -> Bool {
    engines?.contains(engine) ?? true
  }
}
