import CoreFoundation
import Foundation

public struct DrawThingsConfigImport: Identifiable {
  public let id = UUID()
  public var name: String
  public var modelID: String?
  public var prompt: String?
  public var negativePrompt: String?
  public var configuration: [String: JSONValue] = [:]
  public var loras: [DrawThingsLoRA]?
  public var warnings: [String] = []
  public static let presetsURL = URL(string: "https://github.com/drawthingsai/community-models/tree/main/configs")!

  public static func parse(_ data: Data, operation: String) throws -> [Self] {
    guard data.count <= 2 * 1024 * 1024 else { throw StudioError.invalid("Choose a config smaller than 2 MiB.") }
    let json = try JSONSerialization.jsonObject(with: data)
    let entries: [[String: Any]]
    if let list = json as? [[String: Any]] { entries = list }
    else if let object = json as? [String: Any] { entries = [object] }
    else { throw StudioError.invalid("Use a Draw Things JSON object or array of named configurations.") }
    guard !entries.isEmpty, entries.count <= 200 else { throw StudioError.invalid("Choose 1–200 configurations.") }
    return try entries.map { entry in
      let values: [String: Any]
      if let nested = entry["configuration"] {
        guard let object = nested as? [String: Any] else { throw StudioError.invalid("configuration must be a JSON object.") }
        values = object
      } else { values = entry }
      var result = Self(name: entry["name"] as? String ?? "Imported configuration")
      func string(_ key: String, in object: [String: Any]) throws -> String? {
        guard let value = object[key], !(value is NSNull) else { return nil }
        guard let text = value as? String else { throw StudioError.invalid("\(key) must be text.") }
        return text
      }
      let model = try string("model", in: values)
      let modelID = try string("modelID", in: values)
      guard model == nil || modelID == nil || model == modelID else {
        throw StudioError.invalid("Conflicting model and modelID values.")
      }
      result.modelID = model ?? modelID
      result.prompt = try string("prompt", in: entry) ?? string("prompt", in: values)
      result.negativePrompt = try string("negativePrompt", in: entry) ?? string("negativePrompt", in: values)
      let aliases = ["fpsId": "fps", "shiftForAudio": "audioShift"]
      let integerBounds: [String: ClosedRange<Double>] = ["width": 64...4096, "height": 64...4096,
        "steps": 1...1000, "seed": 0...Double(UInt32.max), "sampler": 0...19, "fps": 1...240, "numFrames": 1...100000]
      let realBounds: [String: ClosedRange<Double>] = ["guidanceScale": 0...100, "strength": 0...1,
        "shift": 0...100, "audioShift": 0.1...100]
      for (source, value) in values.sorted(by: { $0.key < $1.key }) {
        let key = aliases[source] ?? source
        if let bounds = integerBounds[key] ?? realBounds[key] {
          if operation == "image" && ["fps", "numFrames", "audioShift"].contains(key) {
            result.warnings.append("\(source): video setting is not applied to an image."); continue
          }
          guard let number = value as? NSNumber, CFGetTypeID(number) != CFBooleanGetTypeID(),
            number.doubleValue.isFinite, bounds.contains(number.doubleValue) else {
            throw StudioError.invalid("\(source) is outside its supported numeric range.")
          }
          let n = number.doubleValue
          if integerBounds[key] != nil {
            guard n.rounded() == n, !["width", "height"].contains(key) || Int(n) % 64 == 0 else {
              throw StudioError.invalid("\(source) must be an integer; dimensions use multiples of 64.")
            }
          }
          let mapped: JSONValue = integerBounds[key] == nil ? .number(n) : .integer(Int(n))
          if let prior = result.configuration[key], prior != mapped {
            throw StudioError.invalid("Conflicting values for \(key) and its alias.")
          }
          result.configuration[key] = mapped
        } else if !["model", "modelID", "prompt", "negativePrompt", "name", "version", "loras"].contains(source) {
          result.warnings.append("\(source): not imported by this version of Studio.")
        }
      }
      if let value = values["loras"] {
        guard let list = value as? [[String: Any]], list.count <= 16 else { throw StudioError.invalid("Use up to 16 LoRAs.") }
        result.loras = try list.compactMap { value in
          if let mode = value["mode"] as? String, mode != "all" {
            result.warnings.append("LoRA mode \(mode): this LoRA was not imported."); return nil
          }
          guard let id = (value["file"] ?? value["modelID"]) as? String,
            let weight = value["weight"] as? NSNumber, CFGetTypeID(weight) != CFBooleanGetTypeID() else {
            throw StudioError.invalid("Each LoRA needs its server filename and numeric weight.")
          }
          let lora = DrawThingsLoRA(modelID: id, weight: weight.doubleValue)
          try lora.validate(); return lora
        }
        guard Set(result.loras!.map(\.modelID)).count == result.loras!.count else { throw StudioError.invalid("The config has duplicate LoRAs.") }
      }
      guard result.modelID != nil || !result.configuration.isEmpty || result.prompt != nil || result.negativePrompt != nil || result.loras != nil else {
        throw StudioError.invalid("No supported Draw Things settings were found.")
      }
      return result
    }
  }
  public func apply(to draft: inout DrawThingsImageDraft, includePrompt: Bool) {
    if let modelID { draft.modelID = modelID }
    if let loras { draft.loras = loras }
    if includePrompt {
      if let prompt { draft.prompt = prompt }
      if let negativePrompt { draft.negativePrompt = negativePrompt }
    }
    for (key, value) in configuration {
      switch (key, value) {
      case ("width", .integer(let n)): draft.width = n
      case ("height", .integer(let n)): draft.height = n
      case ("steps", .integer(let n)): draft.steps = n
      case ("seed", .integer(let n)): draft.seed = n
      case ("sampler", .integer(let n)): draft.sampler = n
      case ("guidanceScale", .number(let n)): draft.guidance = n
      case ("strength", .number(let n)): draft.strength = n
      case ("shift", .number(let n)): draft.shift = n
      default: break
      }
    }
  }
}
