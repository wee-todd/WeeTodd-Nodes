import AppKit
import ImageIO
import StudioCore
import UniformTypeIdentifiers

extension StudioStore {
  func endpointTasks(for clip: Clip) -> [String]? {
    guard clip.profileID != "auto" else { return nil }
    return profiles.first { $0.id == clip.profileID && $0.engine == clip.engine.rawValue }?.generation?.supportedTasks
  }

  func supportsEndpoint(_ role: MediaRole, for clip: Clip) -> Bool {
    clip.supportsEndpoint(role, supportedTasks: endpointTasks(for: clip))
  }

  @discardableResult func assignEndpoint(_ asset: MediaAsset, to clipID: UUID, role: MediaRole) -> Bool {
    guard let index = project.clips.firstIndex(where: { $0.id == clipID }) else { return false }
    var clip = project.clips[index]
    var linked = project.assets.first { $0.scope == .clip && $0.owner == clipID && $0.path == asset.path && $0.kind == .image } ?? asset
    let existing = project.assets.contains { $0.id == linked.id && $0.scope == .clip && $0.owner == clipID }
    if !existing { linked.id = UUID(); linked.scope = .clip; linked.owner = clipID }
    do {
      try clip.assignEndpoint(linked, role: role, fps: clip.settings(in: project).fps,
                              supportedTasks: endpointTasks(for: clip))
      change { project in
        if !existing { project.assets.append(linked) }
        project.clips[index] = clip
      }
      select(clipID)
      notice = "\(role.label): \(asset.name)"
      return true
    } catch { self.error = error.localizedDescription; return false }
  }

  @discardableResult func importEndpoint(_ url: URL, to clipID: UUID, role: MediaRole) -> Bool {
    guard url.isFileURL, let source = CGImageSourceCreateWithURL(url as CFURL, nil),
      CGImageSourceGetCount(source) > 0 else { error = "Drop an image into the frame slot."; return false }
    var asset = MediaAsset(name: url.deletingPathExtension().lastPathComponent, kind: .image, path: url.path, scope: .clip, owner: clipID)
    let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any]
    asset.width = properties?[kCGImagePropertyPixelWidth] as? Int ?? 0
    asset.height = properties?[kCGImagePropertyPixelHeight] as? Int ?? 0
    return assignEndpoint(asset, to: clipID, role: role)
  }

  func chooseEndpoint(for clipID: UUID, role: MediaRole) {
    let panel = NSOpenPanel()
    panel.allowedContentTypes = [.image]
    panel.allowsMultipleSelection = false
    guard panel.runModal() == .OK, let url = panel.url else { return }
    importEndpoint(url, to: clipID, role: role)
  }

  func removeEndpoint(from clipID: UUID, role: MediaRole) {
    guard let index = project.clips.firstIndex(where: { $0.id == clipID }) else { return }
    change { $0.clips[index].removeEndpoint(role) }
  }
}
