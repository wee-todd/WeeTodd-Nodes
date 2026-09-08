import AppKit
import CryptoKit
import StudioCore
import SwiftUI

struct ActionItem: Identifiable {
  var id: String
  var priority: Int
  var title: String
  var detail: String
  var clipID: UUID?
  var destination: String
}
enum ClipState: String {
  case generated, updated, ready, attention, movie
  var color: Color {
    switch self {
    case .generated: return .green
    case .updated: return .yellow
    case .ready: return .orange
    case .attention: return .red
    case .movie: return .blue
    }
  }
  var label: String {
    switch self {
    case .generated: return "Generated"
    case .updated: return "Updated · processing needed"
    case .ready: return "Ready to generate"
    case .attention: return "Needs attention"
    case .movie: return "Movie asset"
    }
  }
}
@MainActor extension StudioStore {
  func signature(for clip: Clip) -> String {
    var parts = [clip.generationFingerprint]
    var paths = clip.attachments.compactMap { attachment in
      allAssets.first { $0.id == attachment.assetID }?.path
    }
    if clip.profileID != "auto" {
      paths.append(clip.profileID)
    } else if let profile = profiles.first(where: {
      $0.engine == clip.engine.rawValue && $0.task == clip.inferredTask
    }) {
      paths.append(profile.id)
    }
    for path in paths.sorted() {
      let attributes = try? FileManager.default.attributesOfItem(atPath: path)
      parts.append(
        path + "|" + String(describing: attributes?[.modificationDate]) + "|"
          + String(describing: attributes?[.size]))
    }
    return SHA256.hash(data: Data(parts.joined(separator: "\n").utf8)).map {
      String(format: "%02x", $0)
    }.joined()
  }
  func issues(for clip: Clip) -> [String] {
    if clip.engine == .movie {
      return clip.sourcePath.isEmpty || !FileManager.default.fileExists(atPath: clip.sourcePath)
        ? ["Relink the source movie"] : []
    }
    var result: [String] = []
    if let message = validationErrors[clip.id] { result.append(message) }
    if clip.prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
      result.append("Add a prompt")
    }
    if clip.generationWidth % 32 != 0 || clip.generationHeight % 32 != 0
      || clip.generationWidth < 64 || clip.generationHeight < 64
    {
      result.append("Choose generation dimensions on the 32-pixel grid")
    }
    if clip.duration <= 0 { result.append("Set a positive duration") }
    if clip.profileID == "auto" {
      if !profiles.contains(where: {
        $0.engine == clip.engine.rawValue
          && ($0.task == clip.inferredTask
            || (clip.engine == .ltx25 && ["a2v", "fflf"].contains(clip.inferredTask)
              && $0.task == "t2v"))
      }) {
        result.append("Add a compatible \(clip.engine.label) \(clip.inferredTask) recipe")
      }
    } else if !profiles.contains(where: {
      $0.id == clip.profileID && $0.engine == clip.engine.rawValue
    }) {
      result.append("Select a matching model recipe")
    }
    for a in clip.attachments {
      guard let asset = allAssets.first(where: { $0.id == a.assetID }) else {
        result.append("Relink a missing attachment")
        continue
      }
      if asset.path.isEmpty || !FileManager.default.fileExists(atPath: asset.path) {
        result.append("Relink \(asset.name)")
      }
      if [.first, .last, .keyframe].contains(a.role) && asset.kind != .image {
        result.append("Use an image for \(a.role.label)")
      }
      if a.role == .audioDriver && asset.kind != .audio {
        result.append("Use an audio file for the audio driver")
      }
    }
    if clip.attachments.filter({ $0.role == .audioDriver }).count > 1 {
      result.append("Use only one audio driver")
    }
    return Array(Set(result)).sorted()
  }
  func clipState(_ clip: Clip) -> ClipState {
    if clip.engine == .movie { return .movie }
    if !issues(for: clip).isEmpty { return .attention }
    if !clip.versions.isEmpty {
      if clip.renderedSignature == signature(for: clip)
        && FileManager.default.fileExists(atPath: clip.sourcePath)
      {
        return clip.motionFidelity?.enabled == true && !clip.motionIsCurrent ? .updated : .generated
      }
      return .updated
    }
    return .ready
  }
  var actionItems: [ActionItem] {
    var items: [ActionItem] = []
    do { try project.settings.validate() } catch {
      items.append(
        ActionItem(
          id: "movie-settings", priority: 0, title: "Correct movie settings",
          detail: error.localizedDescription, destination: "project"))
    }
    for region in project.audio {
      if !FileManager.default.fileExists(atPath: region.path) {
        items.append(
          ActionItem(
            id: region.id.uuidString + "-audio", priority: 0, title: "Relink audio",
            detail: URL(fileURLWithPath: region.path).lastPathComponent,
            destination: "audio:" + region.id.uuidString))
      }
    }
    if !FileManager.default.isExecutableFile(atPath: runtime.pythonPath) {
      items.append(
        ActionItem(
          id: "runtime", priority: 0, title: "Connect the MLX renderer",
          detail: "Choose the repository and Python environment.", destination: "runtime"))
    }
    if profiles.isEmpty {
      items.append(
        ActionItem(
          id: "profiles", priority: 0, title: "Import model recipes",
          detail: "Generation needs at least one compatible component recipe.",
          destination: "runtime"))
    }
    for c in project.clips {
      let blockers = issues(for: c)
      for (i, text) in blockers.enumerated() {
        items.append(
          ActionItem(
            id: c.id.uuidString + "-\(i)", priority: 0, title: c.name + " · " + text,
            detail: "Resolve before generation or export.", clipID: c.id,
            destination: text.contains("recipe")
              ? "runtime" : text.contains("prompt") ? "prompt" : "clip"))
      }
      if c.motionFidelity?.enabled == true && !c.motionIsCurrent {
        items.append(ActionItem(id: c.id.uuidString + "-motion", priority: 1,
          title: c.name + " · Enhance motion", detail: "Analyze or refine Motion Fidelity in the clip inspector.",
          clipID: c.id, destination: "clip"))
      }
      if blockers.isEmpty {
        let state = clipState(c)
        if (state == .updated && c.renderedSignature != signature(for: c)) || state == .ready {
          items.append(
            ActionItem(
              id: c.id.uuidString + "-render", priority: 1,
              title: (state == .updated ? "Regenerate " : "Generate ") + c.name,
              detail: state == .updated
                ? "Generation settings changed since the current version."
                : "Prepare, review the exact prompt, and render.", clipID: c.id,
              destination: "prompt"))
        }
      }
      let s = c.settings(in: project)
      if s.interpolation == .metalFX && (c.depthDirectory.isEmpty || c.motionDirectory.isEmpty) {
        items.append(
          ActionItem(
            id: c.id.uuidString + "-guides", priority: 0, title: c.name + " · Add MetalFX guides",
            detail: "Provide matching depth and motion, or select RIFE.", clipID: c.id,
            destination: "clip"))
      }
      if s.interpolation == .rife && (runtime.rifePath.isEmpty || runtime.rifeWeights.isEmpty) {
        items.append(
          ActionItem(
            id: c.id.uuidString + "-rife", priority: 0, title: "Connect RIFE for " + c.name,
            detail: "Choose the executable and weights folder.", clipID: c.id,
            destination: "runtime"))
      }
    }
    if dirty {
      items.append(
        ActionItem(
          id: "save", priority: 2, title: "Save project file",
          detail: "Autosave recovery is current; update your named project too.",
          destination: "save"))
    }
    if !project.clips.isEmpty {
      items.append(
        ActionItem(
          id: "headless", priority: 3, title: "Export a headless movie job",
          detail:
            "Close the editor and run generation, finishing and assembly sequentially from the CLI.",
          destination: "job"))
    }
    return items.sorted { $0.priority == $1.priority ? $0.id < $1.id : $0.priority < $1.priority }
  }
  func act(_ item: ActionItem) {
    showActions = false
    if let id = item.clipID { select(id) }
    if item.destination.hasPrefix("audio:"),
      let id = UUID(uuidString: String(item.destination.dropFirst(6))),
      let region = project.audio.first(where: { $0.id == id })
    {
      selectedAudioID = id
      selectedTitleID = nil
      if let asset = allAssets.first(where: { $0.id == region.assetID }) { relink(asset) }
      return
    }
    switch item.destination {
    case "project": showProjectSettings = true
    case "runtime": showRuntime = true
    case "prompt": showPrompt = true
    case "save": save()
    case "job": exportJob(clipOnly: false)
    default: break
    }
  }
  func addAudioTrack() {
    let t = AudioTrack(
      name: project.audioTracks.isEmpty ? "Music" : "Audio \(project.audioTracks.count+1)")
    change { $0.audioTracks.append(t) }
    selectedTrackID = t.id
  }
  func exportJob(clipOnly: Bool) {
    guard !clipOnly || selectedClip != nil else {
      error = "Select a clip first."
      return
    }
    let panel = NSSavePanel()
    panel.title = clipOnly ? "Export Clip Headless Job" : "Export Movie Headless Job"
    panel.nameFieldStringValue =
      (clipOnly ? selectedClip!.name : project.name) + ".weetodd-job.json"
    guard panel.runModal() == .OK, let url = panel.url else { return }
    Task {
      do {
        var body = try payload()
        body["clipOnly"] = clipOnly
        body["generateIDs"] = project.clips.filter {
          $0.engine != .movie && ($0.renderedSignature != signature(for: $0) || !FileManager.default.fileExists(atPath: $0.sourcePath))
        }.map { $0.id.uuidString }
        _ = try await bridge.invoke("export-job", runtime: runtime, payload: body, output: url)
        notice = "Exported resumable headless job. Run it with render_headless.py --job."
        NSWorkspace.shared.activateFileViewerSelecting([url])
      } catch { self.error = error.localizedDescription }
    }
  }
}
struct ActionList: View {
  @EnvironmentObject var store: StudioStore
  var body: some View {
    VStack(alignment: .leading, spacing: 0) {
      HStack {
        Text("Actions & suggestions").font(.headline)
        Spacer()
        Text("\(store.actionItems.count)").foregroundStyle(.secondary)
      }.padding(16)
      Divider()
      ScrollView {
        if store.actionItems.isEmpty {
          Label("Everything is up to date", systemImage: "checkmark.circle").padding(24)
            .foregroundStyle(.secondary)
        }
        ForEach(store.actionItems) { item in
          Button {
            store.act(item)
          } label: {
            HStack(alignment: .top, spacing: 10) {
              Image(
                systemName: item.priority == 0
                  ? "exclamationmark.circle.fill"
                  : item.priority == 1 ? "play.circle.fill" : "lightbulb"
              ).foregroundStyle(
                item.priority == 0 ? Color.red : item.priority == 1 ? .orange : .secondary)
              VStack(alignment: .leading, spacing: 4) {
                Text(item.title).font(.system(size: 12, weight: .medium))
                Text(item.detail).font(.system(size: 11)).foregroundStyle(.secondary)
              }
              Spacer()
              Image(systemName: "chevron.right").font(.caption).foregroundStyle(.tertiary)
            }.padding(13).frame(maxWidth: .infinity, alignment: .leading)
          }.buttonStyle(.plain)
          Divider().padding(.leading, 38)
        }
      }.frame(maxHeight: 450)
    }.frame(width: 410)
  }
}
