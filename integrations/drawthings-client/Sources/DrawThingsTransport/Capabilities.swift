import Foundation
import ModelZoo

public enum Capabilities {
  static func rules(for model: String) -> [String: Any]? {
    guard ModelZoo.specificationForModel(model) != nil else { return nil }
    let operation: String
    switch ModelZoo.versionForModel(model) {
    case .minimaxH3:
      guard ModelZoo.modifierForModel(model) == .fl2va else { return nil }
      operation = "video"
    case .ltx2, .ltx2_3: operation = "video"
    case .flux1, .flux2, .flux2_4b, .flux2_9b, .qwenImage, .zImage, .krea2: operation = "image"
    default: return nil
    }
    let dimensions = ["min": 64, "max": 4096, "multipleOf": 64]
    var rule: [String: Any] = [
      "width": dimensions, "height": dimensions, "inputRoleCombinations": [[]] as [[String]],
      "maxLoRAs": 16, "automaticSettings": [:] as [String: Any],
      "requiresAudio": operation == "video"
    ]
    if operation == "image" {
      var combinations: [[String]] = [[], ["canvas"]]
      if [.flux2, .flux2_4b, .flux2_9b].contains(ModelZoo.versionForModel(model)) {
        for count in 1...8 {
          let references = Array(repeating: "moodboard", count: count)
          combinations.append(references)
          combinations.append(["canvas"] + references)
        }
      }
      rule["inputRoleCombinations"] = combinations
    }
    if operation == "video" {
      rule["inputRoleCombinations"] = [[], ["first"]]
      rule["numFrames"] = ["min": 1, "max": 100000, "multipleOf": 8, "offset": 1]
      rule["fps"] = ["min": 1, "max": 240, "multipleOf": 1]
      if ModelZoo.versionForModel(model) == .minimaxH3 {
        rule["inputRoleCombinations"] = [[], ["first"], ["first", "last"]]
        rule["numFrames"] = ["min": 5, "max": 100000, "multipleOf": 17, "offset": 5]
        rule["fps"] = ["min": 24, "max": 24, "multipleOf": 1]
      }
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
