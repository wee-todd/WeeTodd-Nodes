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
    "shift", "sampler"
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
    case .ltx2, .ltx2_3:
      guard operation == "video" else { throw TransportError.unsupportedOperation }
    case .flux1, .flux2, .flux2_4b, .flux2_9b, .qwenImage, .zImage:
      guard operation == "image" else { throw TransportError.unsupportedOperation }
    default: throw TransportError.unsupportedModel
    }
    _ = try Conditioning.inputs(request)
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
    if let value = values["guidanceScale"] {
      config.guidanceScale = Float(try number(value, min: 0, max: 100))
    }
    if let value = values["strength"] {
      config.strength = Float(try number(value, min: 0, max: 1))
    }
    if let value = values["shift"] {
      config.shift = Float(try number(value, min: 0, max: 100))
    }
    if let value = values["sampler"] {
      let raw = Int8(try number(value, min: 0, max: 127, integer: true))
      guard SamplerType(rawValue: raw) != nil else { throw TransportError.invalidRequest }
      config.sampler = raw
    }
    if operation == "video" {
      config.numFrames = UInt32(try number(values["numFrames"], min: 1, max: 100000, integer: true))
      config.fps = UInt32(try number(values["fps"], min: 1, max: 240, integer: true))
      guard (config.numFrames - 1) % 8 == 0 else { throw TransportError.invalidRequest }
    } else if values["numFrames"] != nil || values["fps"] != nil {
      throw TransportError.invalidRequest
    }
    return config.createGenerationConfiguration()
  }
  static func requiresAudio(_ config: GenerationConfiguration) -> Bool {
    guard let model = config.model else { return false }
    switch ModelZoo.versionForModel(model) {
    case .ltx2, .ltx2_3: return true
    default: return false
    }
  }
}

public enum ComputeEstimate {
  public static let revision = "d473a2f148b3e7dc9b90d0b7cfccc5cda999eb66"

  public static func evaluate(_ request: [String: Any]) throws -> [String: Any] {
    let config = try Configuration.resolve(request)
    guard let cu = ComputeUnits.from(config, hasImage: !(try Conditioning.inputs(request)).isEmpty, shuffleCount: 0) else {
      throw TransportError.unsupportedModel
    }
    let encoded = try JSONEncoder().encode(JSGenerationConfiguration(configuration: config))
    guard let full = try JSONSerialization.jsonObject(with: encoded) as? [String: Any] else {
      throw TransportError.invalidRequest
    }
    var supported = Configuration.allowed
    if request["operation"] as? String == "image" { supported.subtract(["numFrames", "fps"]) }
    let normalized = full.filter { supported.contains($0.key) }
    return ["cu": cu, "estimatorRevision": revision, "configuration": normalized]
  }
}
