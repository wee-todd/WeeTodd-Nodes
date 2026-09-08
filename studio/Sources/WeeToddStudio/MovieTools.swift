import AVFoundation
import AppKit
import Foundation
import StudioCore

@MainActor extension StudioStore {
  func previewMovie() async {
    do {
      let body = try payload()
      let snapshot = try JSONEncoder().encode(project)
      let target = Self.supportDirectory.appendingPathComponent("Previews/\(UUID().uuidString).mp4")
      _ = try await bridge.invoke("preview", runtime: runtime, payload: body, output: target)
      guard try JSONEncoder().encode(project) == snapshot else {
        throw StudioError.invalid(
          "The edit changed while preparing its preview. Build the preview again.")
      }
      player.pause()
      isPlaying = false
      previewMode = "Movie"
      playhead = 0
      player.replaceCurrentItem(with: AVPlayerItem(url: target))
      notice = "Movie preview includes titles, transitions and all active audio tracks."
    } catch { self.error = error.localizedDescription }
  }
  func importSequence() {
    let panel = NSOpenPanel()
    panel.title = "Import Image Sequence at Movie Frame Rate"
    panel.canChooseDirectories = true
    panel.canChooseFiles = false
    guard panel.runModal() == .OK, let folder = panel.url else { return }
    Task {
      do {
        let target = Self.supportDirectory.appendingPathComponent("Sequences/\(UUID().uuidString)")
        let result = try await bridge.invoke(
          "sequence", runtime: runtime,
          payload: ["path": folder.path, "fps": project.settings.fps], output: target)
        guard let video = result["video"] as? String else {
          throw StudioError.invalid("No sequence proxy was produced.")
        }
        let info = try await bridge.invoke("inspect", runtime: runtime, payload: ["path": video])
        var asset = MediaAsset(
          name: folder.lastPathComponent, kind: .sequence, path: video, scope: .project)
        asset.duration = result["duration"] as? Double ?? 0
        asset.width = info["width"] as? Int ?? 0
        asset.height = info["height"] as? Int ?? 0
        asset.fps = project.settings.fps
        asset.text = folder.path
        change { $0.assets.append(asset) }
        selectedAssetID = asset.id
        notice = "Linked sequence frames and built a ProRes editing proxy at \(asset.fps) fps."
      } catch { self.error = error.localizedDescription }
    }
  }
  func insertBridge() async {
    guard let c = selectedClip, let index = project.clips.firstIndex(where: { $0.id == c.id }),
      index + 1 < project.clips.count
    else {
      error = "Select a clip with a following neighbor to bridge."
      return
    }
    let nextID = project.clips[index + 1].id
    do {
      let destination = Self.supportDirectory.appendingPathComponent("Anchors/\(UUID().uuidString)")
      let result = try await bridge.invoke(
        "bridge-frames", runtime: runtime, payload: try payload(), output: destination)
      guard let current = project.clips.firstIndex(where: { $0.id == c.id }),
        current + 1 < project.clips.count,
        project.clips[current + 1].id == nextID
      else { throw StudioError.invalid("The neighboring clips changed. Insert the bridge again.") }
      var clip = Clip(name: "Bridge · " + c.name, engine: .ltx25)
      clip.duration = 2
      var assets: [MediaAsset] = []
      for (key, role) in [("first", MediaRole.first), ("last", MediaRole.last)] {
        guard let value = result[key] as? String else {
          throw StudioError.invalid("A bridge anchor was missing.")
        }
        let asset = MediaAsset(
          name: "Bridge " + key + " frame", kind: .image, path: value, scope: .clip, owner: clip.id)
        assets.append(asset)
        clip.attachments.append(Attachment(assetID: asset.id, role: role))
      }
      change {
        $0.clips.insert(clip, at: current + 1)
        $0.assets.append(contentsOf: assets)
      }
      select(clip.id)
      showPrompt = true
      notice = "First and last frames are linked. Describe the movement between them."
    } catch { self.error = error.localizedDescription }
  }
}
