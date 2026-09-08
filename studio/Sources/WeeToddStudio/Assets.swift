import AVFoundation
import AppKit
import StudioCore
import SwiftUI
import UniformTypeIdentifiers

struct AssetBrowser: View {
  @EnvironmentObject var store: StudioStore
  @State private var globalOpen = false
  @State private var projectOpen = true
  @State private var clipOpen = true
  @State private var search = ""
  var body: some View {
    VStack(spacing: 0) {
      PanelHeading(
        title: "MEDIA & ASSETS", icon: "square.grid.2x2", trailing: "\(store.allAssets.count)")
      HStack {
        Image(systemName: "magnifyingglass").foregroundStyle(.secondary)
        TextField("Search assets", text: $search).textFieldStyle(.plain)
      }.font(.system(size: 11)).padding(10).background(
        Theme.raised, in: RoundedRectangle(cornerRadius: 6)
      ).padding(12)
      ScrollView {
        VStack(alignment: .leading, spacing: 18) {
          section("Global", scope: .global, expanded: $globalOpen, assets: store.globalAssets)
          section(
            "Project", scope: .project, expanded: $projectOpen,
            assets: store.project.assets.filter { $0.scope == .project })
          section(
            "Clip", scope: .clip, expanded: $clipOpen,
            assets: store.project.assets.filter {
              $0.scope == .clip && $0.owner == store.selectedClipID
            })
        }.padding(.horizontal, 12).padding(.bottom, 16)
      }
      if let asset = store.selectedAsset {
        Divider()
        VStack(alignment: .leading, spacing: 9) {
          Text(asset.name).font(.system(size: 12, weight: .semibold)).lineLimit(2)
          Text("\(asset.kind.rawValue.capitalized) · \(asset.scope.rawValue.capitalized) store")
            .font(.system(size: 10)).foregroundStyle(.secondary)
          if asset.width > 0 {
            Text("\(asset.width) × \(asset.height) · \(asset.duration,specifier:"%.2f") s").font(
              .system(size: 10, design: .monospaced)
            ).foregroundStyle(.secondary)
          }
          HStack {
            Menu("Use in clip") { roleButtons(asset) }.disabled(store.selectedClip == nil)
            Button {
              store.addAssetToTimeline(asset)
            } label: {
              Image(systemName: "rectangle.stack.badge.plus")
            }.help("Add to timeline").disabled(
              ![.video, .image, .audio, .sequence].contains(asset.kind))
            Button {
              NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: asset.path)])
            } label: {
              Image(systemName: "folder")
            }.help("Reveal original")
          }
          if !asset.path.isEmpty && !FileManager.default.fileExists(atPath: asset.path) {
            Button("Missing file · Relink…") { store.relink(asset) }.foregroundStyle(.orange)
          }
        }.padding(14).background(Theme.raised.opacity(0.5))
      }
      HStack {
        Image(systemName: "link")
        Text("Linked files · originals stay in place")
        Spacer()
      }.font(.system(size: 9)).foregroundStyle(.secondary).padding(12)
    }.background(Theme.panel)
  }
  func section(_ name: String, scope: AssetScope, expanded: Binding<Bool>, assets: [MediaAsset])
    -> some View
  {
    VStack(alignment: .leading, spacing: 10) {
      HStack {
        Button {
          expanded.wrappedValue.toggle()
        } label: {
          HStack(spacing: 7) {
            Image(systemName: expanded.wrappedValue ? "chevron.down" : "chevron.right").font(
              .system(size: 8, weight: .bold))
            Text(name).font(.system(size: 12, weight: .semibold))
            Text("\(assets.count)").font(.system(size: 10, design: .monospaced)).foregroundStyle(
              .secondary)
          }
        }.buttonStyle(.plain)
        Spacer()
        Button {
          store.chooseImports(scope: scope)
        } label: {
          Image(systemName: "plus").font(.system(size: 11))
        }.buttonStyle(.borderless).help("Import into \(name) store").disabled(
          scope == .clip && store.selectedClip == nil)
      }
      if expanded.wrappedValue {
        if assets.isEmpty {
          VStack(spacing: 7) {
            Image(systemName: scope == .global ? "globe" : scope == .project ? "folder" : "film")
              .font(.system(size: 18, weight: .light))
            Text(
              scope == .clip && store.selectedClip == nil ? "Select a clip" : "Drop or import media"
            ).font(.system(size: 10))
          }.foregroundStyle(.secondary).frame(maxWidth: .infinity).frame(height: 72).overlay(
            RoundedRectangle(cornerRadius: 6).strokeBorder(
              Theme.line, style: StrokeStyle(lineWidth: 1, dash: [4, 4])))
        }
        LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: 10) {
          ForEach(
            assets.filter { search.isEmpty || $0.name.localizedCaseInsensitiveContains(search) }
          ) { asset in
            AssetCard(asset: asset).onTapGesture { store.selectedAssetID = asset.id }.draggable(
              "asset:" + asset.id.uuidString
            )
            .contextMenu {
              roleButtons(asset)
              Divider()
              Button("Add to timeline") { store.addAssetToTimeline(asset) }
              Button("Relink…") { store.relink(asset) }
              Button("Remove link", role: .destructive) { remove(asset) }
            }
          }
        }
      }
    }.onDrop(of: [UTType.fileURL.identifier], isTargeted: nil) { providers in
      for p in providers {
        p.loadItem(forTypeIdentifier: UTType.fileURL.identifier, options: nil) { value, _ in
          let url =
            (value as? Data).flatMap { URL(dataRepresentation: $0, relativeTo: nil) } ?? value
            as? URL
          if let url { Task { @MainActor in await store.importURLs([url], scope: scope) } }
        }
      }
      return true
    }
  }
  @ViewBuilder func roleButtons(_ asset: MediaAsset) -> some View {
    if asset.kind == .image {
      Button("First frame") { store.useAsset(asset, role: .first) }
      Button("Last frame") { store.useAsset(asset, role: .last) }
      Button("Keyframe at playhead") {
        store.useAsset(asset, role: .keyframe, time: store.playhead)
      }
      Button("Reference") { store.useAsset(asset, role: .reference) }
    }
    if [.video, .sequence].contains(asset.kind) {
      Button("Video reference") { store.useAsset(asset, role: .reference) }
      Button("Preprocessed control guide") { store.useAsset(asset, role: .control) }
    }
    if asset.kind == .audio {
      Button("Audio driver") { store.useAsset(asset, role: .audioDriver) }
      Button("Audio reference · H3") { store.useAsset(asset, role: .reference) }
    }
    if asset.kind == .lora { Button("Apply LoRA") { store.useAsset(asset, role: .lora) } }
    if asset.kind == .text { Button("Use prompt text") { store.useAsset(asset, role: .reference) } }
  }
  func remove(_ asset: MediaAsset) {
    if store.project.clips.contains(where: { $0.attachments.contains { $0.assetID == asset.id } })
      || store.project.audio.contains(where: { $0.assetID == asset.id })
    {
      store.error = "This asset is in use. Remove its clip attachment or audio region first."
      return
    }
    if asset.scope == .global {
      store.globalAssets.removeAll { $0.id == asset.id }
      store.saveGlobals()
    } else {
      store.change { $0.assets.removeAll { $0.id == asset.id } }
    }
    store.selectedAssetID = nil
  }
}
struct AssetCard: View {
  @EnvironmentObject var store: StudioStore
  var asset: MediaAsset
  @State private var image: NSImage?
  var body: some View {
    VStack(alignment: .leading, spacing: 6) {
      ZStack {
        Theme.raised
        if let image {
          Image(nsImage: image).resizable().scaledToFill()
        } else {
          Image(systemName: icon).font(.system(size: 22, weight: .light)).foregroundStyle(
            Theme.mint.opacity(0.7))
        }
      }
      .frame(height: 69).clipped().overlay(alignment: .bottomTrailing) {
        Text(asset.kind.rawValue.uppercased()).foregroundStyle(.white).font(
          .system(size: 7, weight: .semibold)
        ).padding(4)
          .background(.black.opacity(0.6), in: RoundedRectangle(cornerRadius: 3)).padding(4)
      }
      Text(asset.name).font(.system(size: 10, weight: .medium)).lineLimit(1).padding(.horizontal, 7)
        .padding(.bottom, 7)
    }.background(Theme.raised.opacity(0.5), in: RoundedRectangle(cornerRadius: 6)).clipShape(
      RoundedRectangle(cornerRadius: 6)
    )
    .overlay(
      RoundedRectangle(cornerRadius: 6).strokeBorder(
        store.selectedAssetID == asset.id ? Theme.mint : Theme.line)
    )
    .task(id: asset.path) { await thumbnail() }
  }
  var icon: String {
    switch asset.kind {
    case .video: return "film"
    case .image: return "photo"
    case .audio: return "waveform"
    case .sequence: return "square.stack.3d.up"
    case .text: return "text.alignleft"
    case .lora: return "slider.horizontal.3"
    }
  }
  func thumbnail() async {
    if asset.kind == .image {
      image = NSImage(contentsOfFile: asset.path)
    } else if [.video, .sequence].contains(asset.kind) {
      let path = asset.path
      let cg = await Task.detached {
        let gen = AVAssetImageGenerator(asset: AVURLAsset(url: URL(fileURLWithPath: path)))
        gen.appliesPreferredTrackTransform = true
        gen.maximumSize = CGSize(width: 260, height: 160)
        return try? gen.copyCGImage(
          at: CMTime(seconds: 0, preferredTimescale: 600), actualTime: nil)
      }.value
      if let cg { image = NSImage(cgImage: cg, size: .zero) }
    }
  }
}
