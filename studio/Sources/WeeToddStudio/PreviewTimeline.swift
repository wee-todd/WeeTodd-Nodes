import AVKit
import AppKit
import StudioCore
import SwiftUI
import UniformTypeIdentifiers

struct PreviewPane: View {
  @EnvironmentObject var store: StudioStore
  var body: some View {
    VStack(spacing: 0) {
      HStack {
        SmallLabel(text: "Viewport")
        Spacer()
        Text(store.selectedClip?.name ?? "No clip selected").font(.system(size: 10))
          .foregroundStyle(.secondary)
        Spacer()
        Button(store.previewMode == "Movie" ? "Clip preview" : "Preview movie") {
          if store.previewMode == "Movie" {
            store.refreshPreview()
          } else {
            Task { await store.previewMovie() }
          }
        }.disabled(store.bridge.busy || store.project.clips.isEmpty)
      }.padding(.horizontal, 20).frame(height: 40)
      GeometryReader { geo in
        ZStack {
          Color.black
          if store.previewMode == "Movie" {
            NativePlayer(player: store.player)
          } else if let clip = store.selectedClip, !clip.sourcePath.isEmpty {
            if ["png", "jpg", "jpeg", "webp", "tif", "tiff", "heic"].contains(
              URL(fileURLWithPath: clip.sourcePath).pathExtension.lowercased()),
              let image = NSImage(contentsOfFile: clip.sourcePath)
            {
              Image(nsImage: image).resizable().scaledToFit()
            } else {
              NativePlayer(player: store.player)
            }
            titlePreview(clip)
          } else {
            VStack(spacing: 18) {
              Image(systemName: "viewfinder").font(.system(size: 43, weight: .ultraLight))
                .foregroundStyle(Theme.mint.opacity(0.7))
              Text(
                store.selectedClip == nil
                  ? "Every movie begins with a shot." : "Your next shot starts here."
              ).font(.system(size: 23, weight: .light))
              Text(
                store.selectedClip == nil
                  ? "Drop a movie below, or create a clip with H3 or LTX."
                  : "Write a prompt. Add references. Make it move."
              ).font(.system(size: 12)).foregroundStyle(.secondary)
              HStack(spacing: 10) {
                Button {
                  if store.selectedClip == nil { store.addClip() }
                  store.showPrompt = true
                } label: {
                  Label("Create a shot", systemImage: "plus")
                }.buttonStyle(.borderedProminent).foregroundStyle(.white)
                Button("Import movie") { store.chooseImports(addToTimeline: true) }.buttonStyle(
                  .bordered)
              }.padding(.top, 4)
            }.padding(28).foregroundStyle(.white)
          }
        }.clipShape(RoundedRectangle(cornerRadius: 5))
          .overlay(RoundedRectangle(cornerRadius: 5).strokeBorder(Theme.line))
          .frame(width: geo.size.width, height: geo.size.height)
      }.padding(.horizontal, 20).padding(.bottom, 12)
      HStack(spacing: 15) {
        Text(timecode(store.playhead)).font(.system(size: 11, design: .monospaced)).foregroundStyle(
          Theme.mint
        ).frame(width: 92, alignment: .leading)
        Spacer()
        Button {
          store.seek(0)
        } label: {
          Image(systemName: "backward.end.fill")
        }
        Button {
          store.seek(store.playhead - 1 / store.project.settings.fps)
        } label: {
          Image(systemName: "backward.frame.fill")
        }
        Button {
          store.togglePlayback()
        } label: {
          Image(systemName: store.isPlaying ? "pause.fill" : "play.fill").font(.system(size: 16))
        }
        Button {
          store.seek(store.playhead + 1 / store.project.settings.fps)
        } label: {
          Image(systemName: "forward.frame.fill")
        }
        Button {
          store.seek(store.selectedClip?.duration ?? 0)
        } label: {
          Image(systemName: "forward.end.fill")
        }
        Spacer()
        Text("\(store.selectedClip?.duration ?? 0,specifier:"%.2f") s").font(
          .system(size: 10, design: .monospaced)
        ).foregroundStyle(.secondary).frame(width: 75, alignment: .trailing)
      }.buttonStyle(.borderless).padding(.horizontal, 22).frame(height: 40)
    }.background(Theme.background)
  }
  func timecode(_ seconds: Double) -> String {
    let frames = Int(max(0, seconds) * store.project.settings.fps)
    let fps = Int(store.project.settings.fps)
    return String(
      format: "%02d:%02d:%02d:%02d", frames / fps / 3600, frames / fps / 60 % 60, frames / fps % 60,
      frames % fps)
  }
  @ViewBuilder func titlePreview(_ clip: Clip) -> some View {
    let index = store.project.clips.firstIndex { $0.id == clip.id } ?? 0
    let time = store.project.start(of: index) + store.playhead
    ForEach(store.project.titles.filter { time >= $0.start && time < $0.start + $0.duration }) {
      title in
      VStack {
        if title.position != "center" { Spacer() }
        Text(title.text).font(.system(size: CGFloat(title.fontSize) * 0.45, weight: .semibold))
          .multilineTextAlignment(.center).foregroundStyle(.white).padding(10).background(
            .black.opacity(0.35), in: RoundedRectangle(cornerRadius: 4))
        if title.position == "center" { EmptyView() } else { Color.clear.frame(height: 25) }
      }.frame(
        maxWidth: .infinity, maxHeight: .infinity,
        alignment: title.position == "center" ? .center : .bottom
      ).allowsHitTesting(false)
    }
  }
}
struct NativePlayer: NSViewRepresentable {
  var player: AVPlayer
  func makeNSView(context: Context) -> AVPlayerView {
    let v = AVPlayerView()
    v.controlsStyle = .none
    v.videoGravity = .resizeAspect
    v.player = player
    return v
  }
  func updateNSView(_ view: AVPlayerView, context: Context) { view.player = player }
}
struct TimelineView: View {
  @EnvironmentObject var store: StudioStore
  var body: some View {
    VStack(spacing: 0) {
      HStack(spacing: 14) {
        SmallLabel(text: "Timeline")
        Text("\(store.project.clips.count) clips · \(store.project.duration,specifier:"%.2f") s")
          .font(.system(size: 10, design: .monospaced)).foregroundStyle(.secondary)
        Spacer()
        Button {
          store.split()
        } label: {
          Image(systemName: "scissors")
        }.help("Split at playhead · ⌘B")
        Button {
          store.addAudioTrack()
        } label: {
          Image(systemName: "waveform.badge.plus")
        }.help("Add audio track")
        Button {
          store.addTitle()
        } label: {
          Image(systemName: "textformat")
        }.help("Add title")
        Button {
          store.chooseImports(addToTimeline: true)
        } label: {
          Image(systemName: "square.and.arrow.down")
        }.help("Import movie")
        Menu {
          ForEach(Engine.allCases.filter { $0 != .movie }) { e in
            Button(e.label) { store.addClip(e) }
          }
        } label: {
          Image(systemName: "plus")
        }.help("Add generated clip")
        Divider().frame(height: 14)
        Image(systemName: "minus.magnifyingglass").foregroundStyle(.secondary)
        Slider(value: $store.zoom, in: 12...100).frame(width: 80)
      }.buttonStyle(.borderless).padding(.horizontal, 16).frame(height: 38)
      Divider().overlay(Theme.line)
      ScrollView(.vertical) {
        HStack(spacing: 0) {
          VStack(alignment: .leading, spacing: 0) {
            Text("TIME").frame(height: 23)
            Label("VIDEO", systemImage: "film").frame(height: 104)
            Label("TITLES", systemImage: "textformat").frame(height: 35)
            ForEach(store.project.audioTracks) { t in
              Button {
                store.selectedTrackID = t.id
                store.selectedTitleID = nil
                store.selectedAudioID = nil
              } label: {
                Text(t.name.uppercased()).lineLimit(1)
              }.buttonStyle(.plain).frame(height: 38)
            }
          }.font(.system(size: 8, weight: .medium)).foregroundStyle(.secondary).padding(
            .horizontal, 12
          ).frame(width: 77, alignment: .leading)
          GeometryReader { geo in
            ScrollView(.horizontal) {
              let width = max(geo.size.width - 10, store.project.duration * store.zoom + 130)
              ZStack(alignment: .topLeading) {
                VStack(spacing: 0) {
                  ruler(width: width)
                  ZStack(alignment: .leading) {
                    Theme.raised.opacity(0.18)
                    if store.project.clips.isEmpty {
                      Text("Drop a movie here, or + add a generated clip").font(.system(size: 11))
                        .foregroundStyle(.secondary).padding(.leading, 18)
                    }
                    ForEach(Array(store.project.clips.enumerated()), id: \.element.id) { i, c in
                      TimelineClip(clip: c, index: i).frame(
                        width: max(35, c.duration * store.zoom), height: 89
                      ).offset(x: store.project.start(of: i) * store.zoom)
                    }
                    Button {
                      store.addClip()
                    } label: {
                      Image(systemName: "plus").frame(width: 33, height: 48).background(
                        Theme.raised, in: RoundedRectangle(cornerRadius: 5))
                    }.buttonStyle(.borderless).offset(x: store.project.duration * store.zoom + 15)
                      .opacity(store.project.clips.isEmpty ? 0 : 1)
                  }.frame(height: 104)
                  ZStack(alignment: .leading) {
                    Theme.violet.opacity(0.04)
                    ForEach(store.project.titles) { t in
                      Button {
                        store.selectedTitleID = t.id
                        store.selectedAudioID = nil
                      } label: {
                        Text(t.text).font(.system(size: 10)).lineLimit(1).padding(.horizontal, 8)
                          .frame(
                            width: max(30, t.duration * store.zoom), height: 25, alignment: .leading
                          ).background(
                            Theme.violet.opacity(0.22), in: RoundedRectangle(cornerRadius: 4))
                      }.buttonStyle(.plain).offset(x: t.start * store.zoom)
                    }
                  }.frame(height: 35)
                  ForEach(store.project.audioTracks) { t in
                    ZStack(alignment: .leading) {
                      Theme.mint.opacity(0.035)
                      ForEach(
                        store.project.audio.filter {
                          $0.trackID == t.id
                            || ($0.trackID == nil && t.id == store.project.audioTracks.first?.id)
                        }
                      ) { a in
                        Button {
                          store.selectedAudioID = a.id
                          store.selectedTitleID = nil
                          store.selectedTrackID = t.id
                        } label: {
                          Label(
                            URL(fileURLWithPath: a.path).lastPathComponent, systemImage: "waveform"
                          ).font(.system(size: 10)).lineLimit(1).padding(.horizontal, 8).frame(
                            width: max(30, a.duration * store.zoom), height: 27, alignment: .leading
                          ).background(
                            Theme.mint.opacity(t.muted ? 0.05 : 0.2),
                            in: RoundedRectangle(cornerRadius: 4))
                        }.buttonStyle(.plain).offset(x: a.start * store.zoom)
                      }
                    }.frame(height: 38)
                  }
                }
                if let id = store.selectedClipID,
                  let i = store.project.clips.firstIndex(where: { $0.id == id })
                {
                  Rectangle().fill(Theme.mint).frame(width: 1, height: 199).overlay(alignment: .top)
                  {
                    Image(systemName: "arrowtriangle.down.fill").font(.system(size: 9))
                      .foregroundStyle(Theme.mint).offset(y: -1)
                  }.offset(
                    x: (store.previewMode == "Movie"
                      ? store.playhead : store.project.start(of: i) + store.playhead) * store.zoom
                  ).allowsHitTesting(false)
                }
              }.frame(width: width, height: CGFloat(162 + store.project.audioTracks.count * 38))
                .dropDestination(for: String.self) { items, _ in
                  handleDrop(items)
                  return true
                }
                .onDrop(of: [UTType.fileURL.identifier], isTargeted: nil) { providers in
                  receiveFiles(providers)
                  return true
                }
            }.scrollIndicators(.visible)
          }
        }.frame(height: CGFloat(162 + store.project.audioTracks.count * 38))
      }
    }.background(Theme.panel).overlay(alignment: .top) {
      Rectangle().fill(Theme.line).frame(height: 1)
    }
  }
  func ruler(width: Double) -> some View {
    ZStack(alignment: .leading) {
      Theme.background.opacity(0.4)
      ForEach(0...max(1, Int(width / store.zoom)), id: \.self) { i in
        VStack(spacing: 2) {
          Text(i % 5 == 0 || store.zoom > 35 ? String(format: "%02d:%02d", i / 60, i % 60) : "")
            .font(.system(size: 8, design: .monospaced)).foregroundStyle(.secondary)
          Rectangle().fill(Theme.line).frame(width: 1, height: 5)
        }.frame(width: 30).offset(x: Double(i) * store.zoom - 15)
      }
    }.frame(height: 23)
  }
  func handleDrop(_ items: [String]) {
    for item in items {
      if item.hasPrefix("asset:"), let id = UUID(uuidString: String(item.dropFirst(6))),
        let a = store.allAssets.first(where: { $0.id == id })
      {
        store.addAssetToTimeline(a)
      } else if item.hasPrefix("clip:"), let id = UUID(uuidString: String(item.dropFirst(5))) {
        store.change { $0.move(id, before: nil) }
      }
    }
  }
  func receiveFiles(_ providers: [NSItemProvider]) {
    for p in providers {
      p.loadItem(forTypeIdentifier: UTType.fileURL.identifier, options: nil) { item, _ in
        let url =
          (item as? Data).flatMap { URL(dataRepresentation: $0, relativeTo: nil) } ?? item as? URL
        if let url {
          Task { @MainActor in await store.importURLs([url], scope: .clip, addToTimeline: true) }
        }
      }
    }
  }
}
struct TimelineClip: View {
  @EnvironmentObject var store: StudioStore
  var clip: Clip
  var index: Int
  var body: some View {
    GeometryReader { geo in
      VStack(alignment: .leading, spacing: 7) {
        HStack(spacing: 5) {
          Image(systemName: clip.engine == .movie ? "film" : "sparkles")
          Text(clip.name).fontWeight(.semibold).lineLimit(1)
          Spacer(minLength: 0)
        }.font(.system(size: 10))
        HStack {
          Text(store.clipState(clip).label.uppercased()).font(.system(size: 8, weight: .medium))
            .lineLimit(1)
          Spacer(minLength: 0)
          Text("\(clip.duration,specifier:"%.1f")s").font(.system(size: 9, design: .monospaced))
        }.foregroundStyle(.secondary)
        HStack(spacing: 0) {
          let slotWidth = min(56, max(8, (geo.size.width - 18) / 2))
          if store.supportsEndpoint(.first, for: clip) || clip.attachments.contains(where: { $0.role == .first }) {
            TimelineFrameSlot(clip: clip, role: .first, width: slotWidth)
          }
          Spacer(minLength: 0)
          ForEach(clip.attachments.filter { $0.role == .keyframe }) { a in
            Image(systemName: "diamond.fill").font(.system(size: 6)).foregroundStyle(Theme.mint)
              .help(a.role.label)
          }
          Spacer(minLength: 0)
          if store.supportsEndpoint(.last, for: clip) || clip.attachments.contains(where: { $0.role == .last }) {
            TimelineFrameSlot(clip: clip, role: .last, width: slotWidth)
          }
        }.frame(height: 30)
      }.padding(9).frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        .background(
          store.clipState(clip).color.opacity(store.selectedClipID == clip.id ? 0.22 : 0.10),
          in: RoundedRectangle(cornerRadius: 6)
        )
        .overlay(
          RoundedRectangle(cornerRadius: 6).strokeBorder(
            store.selectedClipID == clip.id ? store.clipState(clip).color : Theme.line,
            lineWidth: store.selectedClipID == clip.id ? 1.5 : 1)
        )
        .overlay(alignment: .topLeading) {
          if index > 0 && clip.transition != "cut" {
            Image(systemName: "arrow.left.and.right.righttriangle.left.righttriangle.right").font(
              .system(size: 8)
            ).padding(3).background(Theme.raised, in: RoundedRectangle(cornerRadius: 3)).offset(
              x: -4, y: -3)
          }
        }
        .contentShape(Rectangle())
        .onTapGesture { location in
          store.select(clip.id)
          store.seek(location.x / max(1, geo.size.width) * clip.duration)
        }
        .draggable("clip:" + clip.id.uuidString)
        .dropDestination(for: String.self) { items, location in
          for value in items {
            if value.hasPrefix("clip:"), let id = UUID(uuidString: String(value.dropFirst(5))) {
              store.change { $0.move(id, before: clip.id) }
            }
            if value.hasPrefix("asset:"), let id = UUID(uuidString: String(value.dropFirst(6))),
              let asset = store.allAssets.first(where: { $0.id == id })
            {
              store.select(clip.id)
              if asset.kind == .image {
                let ratio = location.x / max(1, geo.size.width)
                if clip.engine == .drawThings || clip.engine == .movie {
                  store.notice = "Drop the image directly onto a supported FF or LF slot."
                } else {
                  store.useAsset(asset, role: .keyframe,
                    time: max(0, min(clip.duration - 1 / 24, ratio * clip.duration)))
                }
              } else {
                store.useAsset(
                  asset,
                  role: asset.kind == .audio
                    ? .audioDriver : asset.kind == .lora ? .lora : .reference)
              }
            }
          }
          return true
        }
        .contextMenu {
          Button("Edit prompt") {
            store.select(clip.id)
            store.showPrompt = true
          }.disabled(clip.engine == .movie)
          Button("Duplicate") {
            store.select(clip.id)
            store.duplicateClip()
          }
          Button("Insert bridge to next clip…") {
            store.select(clip.id)
            Task { await store.insertBridge() }
          }
          Button("Split at playhead") { store.split() }
          Divider()
          Button("Delete", role: .destructive) {
            store.select(clip.id)
            store.deleteClip()
          }
        }
    }.padding(.trailing, 2)
  }
}
