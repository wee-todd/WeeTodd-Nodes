import Foundation

public struct ImageGeneration: Codable, Equatable {
  public var provider: String
  public var requestFingerprint: String
  public var modelID: String
  public var prompt: String
  public var profileID: String?
  public var negativePrompt: String?
  public var configuration: [String: JSONValue]?
  public var inputIDs: [String]?
  public var generatedAt: Date?

  public init(provider: String, requestFingerprint: String, modelID: String, prompt: String) {
    self.provider = provider
    self.requestFingerprint = requestFingerprint
    self.modelID = modelID
    self.prompt = prompt
  }
}

public struct ImageAssetDestination: Codable, Equatable {
  public let scope: AssetScope
  public let projectID: UUID
  public let owner: UUID?
  public init(scope: AssetScope, projectID: UUID, owner: UUID? = nil) {
    self.scope = scope; self.projectID = projectID; self.owner = owner
  }
  public func asset(name: String, path: String, in project: StudioProject) throws -> MediaAsset {
    guard scope == .global || project.id == projectID else {
      throw StudioError.invalid("The destination project changed. The generated image is saved at \(path).")
    }
    if scope == .clip {
      guard let owner, project.clips.contains(where: { $0.id == owner }) else {
        throw StudioError.invalid("The destination clip was removed. The generated image is saved at \(path).")
      }
    }
    return MediaAsset(name: name, kind: .image, path: path, scope: scope, owner: scope == .clip ? owner : nil)
  }
}
