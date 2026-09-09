import Foundation

public enum JSONValue: Codable, Equatable {
  case string(String)
  case integer(Int)
  case number(Double)
  case boolean(Bool)
  case array([JSONValue])
  case object([String: JSONValue])
  case null

  public init(from decoder: Decoder) throws {
    let container = try decoder.singleValueContainer()
    if container.decodeNil() { self = .null }
    else if let value = try? container.decode(Bool.self) { self = .boolean(value) }
    else if let value = try? container.decode(Int.self) { self = .integer(value) }
    else if let value = try? container.decode(Double.self) { self = .number(value) }
    else if let value = try? container.decode(String.self) { self = .string(value) }
    else if let value = try? container.decode([JSONValue].self) { self = .array(value) }
    else { self = .object(try container.decode([String: JSONValue].self)) }
  }

  public func encode(to encoder: Encoder) throws {
    var container = encoder.singleValueContainer()
    switch self {
    case .string(let value): try container.encode(value)
    case .integer(let value): try container.encode(value)
    case .number(let value): try container.encode(value)
    case .boolean(let value): try container.encode(value)
    case .array(let value): try container.encode(value)
    case .object(let value): try container.encode(value)
    case .null: try container.encodeNil()
    }
  }
}

public struct DrawThingsLoRA: Codable, Equatable, Identifiable {
  public var modelID: String
  public var weight: Double
  public var id: String { modelID }
  public init(modelID: String, weight: Double = 1) {
    self.modelID = modelID; self.weight = weight
  }
  public func validate() throws {
    guard !modelID.isEmpty, weight.isFinite, (0...2).contains(weight) else {
      throw StudioError.invalid("Choose a server LoRA and a finite strength from 0 to 2.")
    }
  }
}

public struct DrawThingsLoRAGroup: Codable, Equatable, Identifiable {
  public var id = UUID()
  public var name: String
  public var profileID: String
  public var family: String
  public var compatibleModelIDs: [String]
  public var members: [DrawThingsLoRA]
  public init(name: String, profileID: String, family: String, compatibleModelIDs: [String], members: [DrawThingsLoRA]) {
    self.name = name; self.profileID = profileID; self.family = family
    self.compatibleModelIDs = compatibleModelIDs; self.members = members
  }
  public func validate(profileID targetProfile: String, family targetFamily: String, modelID: String) throws {
    guard !name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty, !members.isEmpty else {
      throw StudioError.invalid("Name the Draw Things LoRA group and add at least one server LoRA.")
    }
    guard profileID == targetProfile, family == targetFamily, compatibleModelIDs.contains(modelID) else {
      throw StudioError.invalid("This Draw Things LoRA group belongs to a different server profile, model, or family.")
    }
    guard members.count <= 16, Set(members.map(\.modelID)).count == members.count else {
      throw StudioError.invalid("A Draw Things group can contain up to 16 unique server LoRAs.")
    }
    for member in members { try member.validate() }
  }
}

public struct DrawThingsSelection: Codable, Equatable {
  public var profileID: String
  public var modelID: String
  public var modelFamily: String
  public var configuration: [String: JSONValue]
  public var loras: [DrawThingsLoRA]

  public init(
    profileID: String, modelID: String, modelFamily: String,
    configuration: [String: JSONValue] = [:], loras: [DrawThingsLoRA] = []
  ) {
    self.profileID = profileID
    self.modelID = modelID
    self.modelFamily = modelFamily
    self.configuration = configuration
    self.loras = loras
  }
  private enum CodingKeys: String, CodingKey { case profileID, modelID, modelFamily, configuration, loras }
  public init(from decoder: Decoder) throws {
    let c = try decoder.container(keyedBy: CodingKeys.self)
    profileID = try c.decode(String.self, forKey: .profileID)
    modelID = try c.decode(String.self, forKey: .modelID)
    modelFamily = try c.decode(String.self, forKey: .modelFamily)
    configuration = try c.decodeIfPresent([String: JSONValue].self, forKey: .configuration) ?? [:]
    loras = try c.decodeIfPresent([DrawThingsLoRA].self, forKey: .loras) ?? []
  }
  public mutating func apply(_ group: DrawThingsLoRAGroup) { loras = group.members }
  public func unavailableLoRAs(availableIDs: Set<String>) -> [DrawThingsLoRA] {
    loras.filter { !availableIDs.contains($0.modelID) }
  }
}

public struct DrawThingsConnection: Codable, Equatable, Identifiable {
  public var id: String
  public var name: String
  public var route: String
  public var host: String
  public var port: Int
  public var useTLS: Bool
  public var credentialRef: String?
  public var selfHostedConfirmed: Bool?
  public init(id: String = UUID().uuidString, name: String = "Draw Things", route: String = "grpc",
              host: String = "127.0.0.1", port: Int = 7859, useTLS: Bool = false,
              credentialRef: String? = nil, selfHostedConfirmed: Bool? = nil) {
    self.id = id; self.name = name; self.route = route; self.host = host; self.port = port
    self.useTLS = useTLS; self.credentialRef = credentialRef; self.selfHostedConfirmed = selfHostedConfirmed
  }
}
