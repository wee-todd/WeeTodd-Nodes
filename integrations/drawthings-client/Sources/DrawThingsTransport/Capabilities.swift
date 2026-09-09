import Foundation
import ModelZoo

public enum Capabilities {
  static func rules(for model: String) -> [String: Any]? {
    guard ModelZoo.specificationForModel(model) != nil else { return nil }
    let operation: String
    switch ModelZoo.versionForModel(model) {
    case .ltx2, .ltx2_3: operation = "video"
    case .flux1, .flux2, .flux2_4b, .flux2_9b, .qwenImage, .zImage: operation = "image"
    default: return nil
    }
    let dimensions = ["min": 64, "max": 4096, "multipleOf": 64]
    var rule: [String: Any] = [
      "width": dimensions, "height": dimensions, "inputRoleCombinations": [[]] as [[String]],
      "maxLoRAs": 16, "automaticSettings": [:] as [String: Any],
      "requiresAudio": operation == "video"
    ]
    if operation == "video" {
      rule["inputRoleCombinations"] = [[], ["first"]]
      rule["numFrames"] = ["min": 1, "max": 100000, "multipleOf": 8, "offset": 1]
      rule["fps"] = ["min": 1, "max": 240, "multipleOf": 1]
    }
    return ["operations": [operation: rule], "confidence": "verified",
            "source": "adapter-rules-intersect-endpoint-files", "revision": ComputeEstimate.revision]
  }

  static func catalog(files: [String]) -> [String: Any] {
    var models: [String: Any] = [:]
    for model in files { if let rules = rules(for: model) { models[model] = rules } }
    return models
  }
}
