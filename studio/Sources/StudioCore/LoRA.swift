import Foundation

/// Training provenance filters candidates; the renderer validates tensor targets and shapes.
public enum LoRAModel: String, Codable, CaseIterable, Identifiable {
  case h3, ltx23, ltx25
  public var id: String { rawValue }
  public var label: String { Engine(rawValue: rawValue)!.label }
  public func supports(_ engine: Engine) -> Bool {
    rawValue == engine.rawValue || (self == .ltx23 && engine == .ltx25)
  }
}

public struct LoRAMember: Codable, Identifiable, Equatable {
  public var id = UUID()
  public var asset: MediaAsset
  public var strength: Double
  public init(asset: MediaAsset, strength: Double = 1) {
    self.asset = asset
    self.strength = strength
  }
  public var fileKey: String {
    URL(fileURLWithPath: asset.path).standardizedFileURL.resolvingSymlinksInPath().path
  }
  public func validate(for engine: Engine) throws {
    guard asset.kind == .lora, asset.loraModel?.supports(engine) == true else {
      throw StudioError.invalid(
        "Choose a LoRA trained for \(engine.label). LTX 2.5 also accepts compatible LTX 2.3 adapters."
      )
    }
    guard !asset.path.isEmpty,
      URL(fileURLWithPath: asset.path).pathExtension.lowercased() == "safetensors"
    else {
      throw StudioError.invalid("Link a SafeTensors LoRA file for \(asset.name).")
    }
    guard strength.isFinite, (0...2).contains(strength) else {
      throw StudioError.invalid("LoRA strength must be a finite number from 0 to 2.")
    }
  }
}

public struct LoRAGroup: Codable, Identifiable, Equatable {
  public var id = UUID()
  public var name: String
  public var engine: Engine
  public var members: [LoRAMember]
  public init(name: String, engine: Engine, members: [LoRAMember] = []) {
    self.name = name
    self.engine = engine
    self.members = members
  }
  public func validate() throws {
    guard !name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty, !members.isEmpty else {
      throw StudioError.invalid("Name the group and add at least one LoRA.")
    }
    for member in members { try member.validate(for: engine) }
    guard Set(members.map(\.fileKey)).count == members.count else {
      throw StudioError.invalid("A group can contain each LoRA file only once.")
    }
  }
  public func supports(_ target: Engine) -> Bool {
    engine == target && (try? validate()) != nil
  }
}

extension StudioProject {
  /// Applying a library item creates clip-owned descriptors, independent of future library edits.
  public mutating func applyLoRAs(
    _ members: [LoRAMember], to clipID: UUID, groupName: String? = nil
  ) throws {
    guard let index = clips.firstIndex(where: { $0.id == clipID }) else {
      throw StudioError.invalid("Select a clip first.")
    }
    let clip = clips[index]
    for member in members { try member.validate(for: clip.engine) }
    let existing = Set(
      clip.attachments.filter { $0.role == .lora }.compactMap { attachment in
        assets.first { $0.id == attachment.assetID }.map { LoRAMember(asset: $0).fileKey }
      })
    let incoming = members.map(\.fileKey)
    guard Set(incoming).count == incoming.count, existing.isDisjoint(with: incoming) else {
      throw StudioError.invalid(
        "This clip already uses a LoRA in this selection. Remove its existing entry before applying it again."
      )
    }
    let applicationID = groupName == nil ? nil : UUID()
    for member in members {
      var asset = member.asset
      asset.id = UUID()
      asset.scope = .clip
      asset.owner = clipID
      assets.append(asset)
      var attachment = Attachment(assetID: asset.id, role: .lora)
      attachment.strength = member.strength
      attachment.loraGroupName = groupName
      attachment.loraGroupID = applicationID
      clips[index].attachments.append(attachment)
    }
  }
}
