import CoreFoundation
import DataModels
import Foundation
import ModelZoo
import ScriptDataModels

public enum TransportError: String, Error {
  case invalidRequest = "invalid_request"
  case invalidMedia = "invalid_media"
  case submissionUncertain = "submission_uncertain"
  case billingUnverified = "billing_unverified"
  case unsupportedModel = "unsupported_model"
  case unsupportedConditioning = "unsupported_conditioning"
  case unsupportedOperation = "unsupported_operation"
  case connectionFailed = "connection_failed"
  case authenticationRequired = "authentication_required"
}

public enum Configuration {
  static let allowed: Set<String> = [
    "width", "height", "steps", "seed", "guidanceScale", "strength", "numFrames", "fps",
    "shift", "audioShift", "sampler"
  ]

  static func number(_ value: Any?, min: Double, max: Double, integer: Bool = false) throws -> Double {
    guard let number = value as? NSNumber, CFGetTypeID(number) != CFBooleanGetTypeID() else {
      throw TransportError.invalidRequest
    }
    let result = number.doubleValue
    guard result.isFinite, result >= min, result <= max,
      !integer || result.rounded() == result else { throw TransportError.invalidRequest }
    return result
  }

  public static func resolve(_ request: [String: Any]) throws -> GenerationConfiguration {
    guard let model = request["modelID"] as? String,
      ModelZoo.specificationForModel(model) != nil else { throw TransportError.unsupportedModel }
    guard let operation = request["operation"] as? String, ["image", "video"].contains(operation),
      let values = request["configuration"] as? [String: Any],
      Set(values.keys).isSubset(of: allowed) else { throw TransportError.invalidRequest }
    let version = ModelZoo.versionForModel(model)
    switch version {
    case .minimaxH3:
      guard operation == "video", ModelZoo.modifierForModel(model) == .fl2va else {
        throw TransportError.unsupportedOperation
      }
    case .ltx2, .ltx2_3:
      guard operation == "video" else { throw TransportError.unsupportedOperation }
    case .flux1, .flux2, .flux2_4b, .flux2_9b, .qwenImage, .zImage, .krea2:
      guard operation == "image" else { throw TransportError.unsupportedOperation }
    default: throw TransportError.unsupportedModel
    }
    let inputs = try Conditioning.inputs(request)
    guard operation == "image" || inputs.count < 2 || version == .minimaxH3 else {
      throw TransportError.unsupportedConditioning
    }
    if operation == "image", inputs.contains(where: { $0["role"] as? String == "moodboard" }),
      ![.flux2, .flux2_4b, .flux2_9b].contains(version) {
      throw TransportError.unsupportedConditioning
    }
    let loras = try Conditioning.loras(request)
    let width = try number(values["width"], min: 64, max: 4096, integer: true)
    let height = try number(values["height"], min: 64, max: 4096, integer: true)
    guard Int(width) % 64 == 0, Int(height) % 64 == 0 else { throw TransportError.invalidRequest }
    let steps = try number(values["steps"], min: 1, max: 1000, integer: true)
    let seed = try number(values["seed"], min: 0, max: Double(UInt32.max), integer: true)
    let config = JSGenerationConfiguration(configuration: GenerationConfiguration.default)
    config.model = model
    config.width = UInt32(width)
    config.height = UInt32(height)
    config.steps = UInt32(steps)
    config.seed = Int64(seed)
    config.loras = loras.map { JSLoRA(lora: $0) }
    if version == .minimaxH3 {
      config.sampler = SamplerType.dDIMTrailing.rawValue
      config.guidanceScale = 1
      config.shift = 12
      config.shiftForAudio = 3
    }
    if let value = values["guidanceScale"] {
      config.guidanceScale = Float(try number(value, min: 0, max: 100))
    }
    if let value = values["strength"] {
      config.strength = Float(try number(value, min: 0, max: 1))
    }
    if let value = values["shift"] {
      config.shift = Float(try number(value, min: 0, max: 100))
    }
    if let value = values["audioShift"] {
      guard version == .minimaxH3 else { throw TransportError.invalidRequest }
      config.shiftForAudio = Float(try number(value, min: 0.1, max: 100))
    }
    if let value = values["sampler"] {
      let raw = Int8(try number(value, min: 0, max: 127, integer: true))
      guard SamplerType(rawValue: raw) != nil else { throw TransportError.invalidRequest }
      config.sampler = raw
    }
    if operation == "video" {
      config.numFrames = UInt32(try number(values["numFrames"], min: 1, max: 100000, integer: true))
      config.fps = UInt32(try number(values["fps"], min: 1, max: 240, integer: true))
      if version == .minimaxH3 {
        guard config.fps == 24, config.numFrames >= 5, (config.numFrames - 5) % 17 == 0 else {
          throw TransportError.invalidRequest
        }
      } else {
        guard (config.numFrames - 1) % 8 == 0 else { throw TransportError.invalidRequest }
      }
    } else if values["numFrames"] != nil || values["fps"] != nil {
      throw TransportError.invalidRequest
    }
    return config.createGenerationConfiguration()
  }
  static func requiresAudio(_ config: GenerationConfiguration) -> Bool {
    guard let model = config.model else { return false }
    switch ModelZoo.versionForModel(model) {
    case .ltx2, .ltx2_3, .minimaxH3: return true
    default: return false
    }
  }
}

public enum ComputeEstimate {
  public static let revision = "08e798b5ad59c3db78b2be53f0ed60b071653302"

  public static func evaluate(_ request: [String: Any]) throws -> [String: Any] {
    if let model = request["modelID"] as? String,
      ModelZoo.specificationForModel(model) == nil, request["profile"] != nil {
      _ = try Discovery.fetch(request, inspectAccount: false)
    }
    let config = try Configuration.resolve(request)
    let inputs = try Conditioning.inputs(request)
    let hasImage = inputs.contains { ["canvas", "first"].contains($0["role"] as? String ?? "") }
    let shuffleCount = inputs.filter { ["moodboard", "last"].contains($0["role"] as? String ?? "")
      && (($0["strength"] as? NSNumber)?.doubleValue ?? 0) > 0 }.count
    guard let cu = ComputeUnits.from(config, hasImage: hasImage, shuffleCount: shuffleCount) else {
      throw TransportError.unsupportedModel
    }
    let encoded = try JSONEncoder().encode(JSGenerationConfiguration(configuration: config))
    guard let full = try JSONSerialization.jsonObject(with: encoded) as? [String: Any] else {
      throw TransportError.invalidRequest
    }
    var supported = Configuration.allowed
    if request["operation"] as? String == "image" { supported.subtract(["numFrames", "fps"]) }
    var normalized = full.filter { supported.contains($0.key) }
    if ModelZoo.versionForModel(config.model ?? "") == .minimaxH3 {
      normalized["audioShift"] = full["shiftForAudio"]
    }
    return ["cu": cu, "estimatorRevision": revision, "configuration": normalized]
  }
}
