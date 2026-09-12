import Foundation

extension ImageAssetDestination {
  public var storageKey: String {
    scope == .global ? "global" : projectID.uuidString + ":" + scope.rawValue + ":" + (owner?.uuidString ?? "")
  }
}

public struct ImageWorkspaceSession: Codable, Equatable {
  public var draft: DrawThingsImageDraft
  public var previewPath: String?
}

public struct ImageWorkspaceLibrary: Codable {
  public var version = 1
  public var sessions: [String: ImageWorkspaceSession] = [:]
  public var activeKey: String?
  public init() {}
  public mutating func record(_ draft: DrawThingsImageDraft, preview: String?) {
    let key = draft.destination.storageKey
    sessions[key] = ImageWorkspaceSession(draft: draft, previewPath: preview)
    activeKey = key
  }
  public func write(to url: URL) throws {
    let data = try JSONEncoder().encode(self)
    guard data.count <= 8 * 1024 * 1024 else { throw StudioError.invalid("Image draft library is too large to save.") }
    try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
    try data.write(to: url, options: .atomic)
  }
  public static func read(from url: URL) throws -> Self {
    let attributes = try FileManager.default.attributesOfItem(atPath: url.path)
    guard (attributes[.size] as? NSNumber)?.intValue ?? 0 <= 8 * 1024 * 1024 else {
      throw StudioError.invalid("Image draft library is too large to open.")
    }
    let library = try JSONDecoder().decode(Self.self, from: Data(contentsOf: url))
    guard library.version == 1 else { throw StudioError.invalid("This image draft format requires a newer Studio.") }
    return library
  }
}
