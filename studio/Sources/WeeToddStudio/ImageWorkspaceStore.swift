import AppKit
import ImageIO
import StudioCore
import UniformTypeIdentifiers

extension StudioStore {
  func persistImageWorkspace() {
    guard !restoringImageWorkspace else { return }
    if let draft = imageDraft { imageWorkspaceLibrary.record(draft, preview: imagePreviewPath) }
    else { imageWorkspaceLibrary.activeKey = nil }
    do { try imageWorkspaceLibrary.write(to: Self.supportDirectory.appendingPathComponent("image-workspaces.json")) }
    catch { notice = "Image draft could not be saved: \(error.localizedDescription)" }
  }
  func restoreImageWorkspaces() {
    restoringImageWorkspace = true
    defer { restoringImageWorkspace = false }
    let url = Self.supportDirectory.appendingPathComponent("image-workspaces.json")
    guard FileManager.default.fileExists(atPath: url.path) else { return }
    do {
      imageWorkspaceLibrary = try ImageWorkspaceLibrary.read(from: url)
      if let key = imageWorkspaceLibrary.activeKey, let session = imageWorkspaceLibrary.sessions[key],
        (try? session.draft.destination.asset(name: "draft", path: "", in: project)) != nil {
        imageDraft = session.draft; imagePreviewPath = session.previewPath
        notice = "Recovered your image workspace. Check Settings & CU before generating."
      }
    } catch {
      let backup = url.deletingLastPathComponent().appendingPathComponent("image-workspaces-unreadable-\(UUID().uuidString).json")
      do {
        try FileManager.default.moveItem(at: url, to: backup)
        notice = "The image draft file could not be read. It was preserved as \(backup.lastPathComponent)."
      } catch { notice = "Image draft recovery needs attention: \(error.localizedDescription)" }
    }
  }
  func replaceImageReference(_ id: UUID) {
    let panel = NSOpenPanel(); panel.allowedContentTypes = [.image]
    guard panel.runModal() == .OK, let url = panel.url,
      let source = CGImageSourceCreateWithURL(url as CFURL, nil), CGImageSourceGetCount(source) > 0,
      let index = imageDraft?.moodboard.firstIndex(where: { $0.id == id }) else { return }
    imageDraft?.moodboard[index].path = url.path; imageEstimate = nil
  }
  func chooseImageInputs(canvas: Bool) {
    let panel = NSOpenPanel()
    panel.allowedContentTypes = [.image]
    panel.allowsMultipleSelection = !canvas
    guard panel.runModal() == .OK else { return }
    loadImageInputs(panel.urls, canvas: canvas)
  }
  func loadImageInputs(_ urls: [URL], canvas: Bool) {
    guard imageDraft != nil else { return }
    guard !canvas || urls.count == 1 else { error = "Drop one image onto the canvas."; return }
    guard canvas || (imageDraft?.moodboard.count ?? 0) + urls.count <= 8 else {
      error = "Use up to eight mood-board references in this initial version."; return
    }
    for url in urls {
      guard url.isFileURL, let source = CGImageSourceCreateWithURL(url as CFURL, nil), CGImageSourceGetCount(source) > 0 else {
        error = "Choose a readable image file."; return
      }
    }
    let inputs = urls.map { ImageWorkspaceInput(path: $0.path) }
    if canvas { imageDraft?.canvas = inputs.first; imagePreviewPath = nil }
    else { imageDraft?.moodboard.append(contentsOf: inputs) }
    imageEstimate = nil
  }
  func dropImageInputs(_ providers: [NSItemProvider], canvas: Bool) -> Bool {
    guard !providers.isEmpty, !canvas || providers.count == 1 else { return false }
    // Process a multi-file drop in provider order, not asynchronous completion order.
    let destination = imageDraft?.destination
    Task { @MainActor in
      var urls: [URL] = []
      for provider in providers {
        let url: URL? = await withCheckedContinuation { continuation in
          if provider.hasItemConformingToTypeIdentifier(UTType.fileURL.identifier) {
            provider.loadItem(forTypeIdentifier: UTType.fileURL.identifier, options: nil) { value, _ in
              continuation.resume(returning: (value as? Data).flatMap { URL(dataRepresentation: $0, relativeTo: nil) } ?? value as? URL)
            }
          } else {
            provider.loadObject(ofClass: NSString.self) { value, _ in
              let text = value as? String ?? ""
              Task { @MainActor in
                let id = text.hasPrefix("asset:") ? UUID(uuidString: String(text.dropFirst(6))) : nil
                let asset = self.allAssets.first { $0.id == id && $0.kind == .image }
                continuation.resume(returning: asset.map { URL(fileURLWithPath: $0.path) })
              }
            }
          }
        }
        guard let url else { error = "Drop image files or image assets here."; return }
        urls.append(url)
      }
      guard imageDraft?.destination == destination else { return }
      loadImageInputs(urls, canvas: canvas)
    }
    return true
  }
  func imageModelsForInputs() -> [(id: String, name: String)] {
    guard let draft = imageDraft else { return [] }
    var roles = draft.canvas?.enabled == true ? ["canvas"] : []
    roles += draft.moodboard.filter { $0.enabled && $0.strength > 0 }.map { _ in "moodboard" }
    let rules = drawThingsCatalogs[draft.profileID]?["capabilities"] as? [String: Any] ?? [:]
    return drawThingsModels(draft.profileID, operation: "image").filter { model in
      let operations = (rules[model.id] as? [String: Any])?["operations"] as? [String: Any]
      let spec = operations?["image"] as? [String: Any]
      return (spec?["inputRoleCombinations"] as? [[String]] ?? []).contains { $0.sorted() == roles.sorted() }
    }
  }
}
