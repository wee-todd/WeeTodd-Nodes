import Foundation

extension Clip {
  public func supportsEndpoint(_ role: MediaRole, supportedTasks: [String]? = nil) -> Bool {
    guard role == .first || role == .last, extensionDirection.isEmpty else { return false }
    switch engine {
    case .movie: return false
    case .drawThings:
      // Draft clips may collect endpoint images before choosing their renderer model.
      if drawThings?.modelID.isEmpty != false { return true }
      switch drawThings?.modelFamily.lowercased() {
      case "minimaxh3": return true
      case "ltx2", "ltx2.3", "ltx23", "ltx2_3": return role == .first
      default: return false
      }
    case .h3, .ltx23, .ltx25:
      guard let supportedTasks else { return true }
      return supportedTasks.contains("fflf") || (role == .first && supportedTasks.contains("i2v"))
    }
  }

  public mutating func assignEndpoint(_ asset: MediaAsset, role: MediaRole, fps: Double,
                                     supportedTasks: [String]? = nil) throws {
    guard asset.kind == .image, supportsEndpoint(role, supportedTasks: supportedTasks) else {
      throw NSError(domain: "FrameEndpoint", code: 1, userInfo: [NSLocalizedDescriptionKey:
        "Choose an image and a clip model/recipe that supports \(role.label.lowercased())."])
    }
    attachments.removeAll { $0.role == role }
    let frameRate = fps.isFinite && fps > 0 ? fps : 24
    attachments.append(Attachment(assetID: asset.id, role: role,
      time: role == .first ? 0 : max(0, duration - 1 / frameRate)))
    if !(engine == .drawThings && generationSelection?.task == "fflf" && role == .first) {
      synchronizeEndpointTask()
    }
  }

  public mutating func removeEndpoint(_ role: MediaRole) {
    guard role == .first || role == .last else { return }
    attachments.removeAll { $0.role == role }
    synchronizeEndpointTask()
  }

  private mutating func synchronizeEndpointTask() {
    guard generationSelection == nil || ["t2v", "i2v", "fflf"].contains(generationSelection!.task) else { return }
    let task = attachments.contains { $0.role == .last } ? "fflf"
      : attachments.contains { $0.role == .keyframe } ? "fflf"
      : attachments.contains { $0.role == .first } ? "i2v" : "t2v"
    var selection = generationSelection ?? GenerationSelection(task: task, preset: .custom)
    selection.task = task
    generationSelection = selection
  }
}
