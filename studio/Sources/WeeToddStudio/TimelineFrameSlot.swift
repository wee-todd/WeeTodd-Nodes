import AppKit
import ImageIO
import StudioCore
import SwiftUI
import UniformTypeIdentifiers

struct TimelineFrameSlot: View {
  @EnvironmentObject var store: StudioStore
  var clip: Clip
  var role: MediaRole
  var width: CGFloat
  @State private var targeted = false
  @State private var thumbnail: NSImage?
  var attachment: Attachment? { clip.attachments.first { $0.role == role } }
  var asset: MediaAsset? { attachment.flatMap { a in store.allAssets.first { $0.id == a.assetID } } }
  var supported: Bool { store.supportsEndpoint(role, for: clip) }
  var label: String { role == .first ? "FF" : "LF" }

  var body: some View {
    Button {
      store.select(clip.id)
      store.seek(role == .first ? 0 : max(0, clip.duration - 1 / max(1, clip.settings(in: store.project).fps)))
      if supported && attachment == nil { store.chooseEndpoint(for: clip.id, role: role) }
    } label: {
      ZStack(alignment: .bottomLeading) {
        RoundedRectangle(cornerRadius: 4).fill(targeted ? Color.accentColor.opacity(0.25) : Color(nsColor: .controlBackgroundColor))
        if let thumbnail {
          Image(nsImage: thumbnail).resizable().scaledToFit().frame(width: width, height: 30)
        } else {
          Image(systemName: attachment == nil ? "plus" : "exclamationmark.triangle")
            .font(.system(size: 10)).frame(maxWidth: .infinity, maxHeight: .infinity)
        }
        Text(label).font(.system(size: 8, weight: .bold))
          .padding(.horizontal, 2).background(.regularMaterial, in: RoundedRectangle(cornerRadius: 2))
      }.frame(width: width, height: 30)
        .overlay(RoundedRectangle(cornerRadius: 4).strokeBorder(
          targeted ? Color.accentColor : attachment == nil ? Color.secondary.opacity(0.6) : Color.accentColor,
          style: StrokeStyle(lineWidth: 1, dash: attachment == nil ? [3, 2] : [])))
        .opacity(supported ? 1 : 0.6)
    }.buttonStyle(.plain)
      .accessibilityLabel("\(clip.name), \(role.label): \(asset?.name ?? (attachment == nil ? "Empty" : "Missing image"))")
      .help("\(role.label) · \(asset?.name ?? "Drop an image or click to import")\(supported ? "" : " · unsupported by the selected model/recipe")")
      .onDrop(of: [UTType.fileURL.identifier, UTType.text.identifier], isTargeted: $targeted) { providers in
        guard supported, providers.count == 1, let provider = providers.first else { return false }
        let targetID = clip.id
        if provider.hasItemConformingToTypeIdentifier(UTType.fileURL.identifier) {
          provider.loadItem(forTypeIdentifier: UTType.fileURL.identifier, options: nil) { item, _ in
            guard let url = (item as? Data).flatMap({ URL(dataRepresentation: $0, relativeTo: nil) }) ?? item as? URL else { return }
            Task { @MainActor in store.importEndpoint(url, to: targetID, role: role) }
          }
        } else {
          provider.loadObject(ofClass: NSString.self) { object, _ in
            guard let value = object as? String, value.hasPrefix("asset:"),
              let id = UUID(uuidString: String(value.dropFirst(6))) else { return }
            Task { @MainActor in
              if let asset = store.allAssets.first(where: { $0.id == id }) {
                store.assignEndpoint(asset, to: targetID, role: role)
              }
            }
          }
        }
        return true
      }
      .contextMenu {
        Button("Replace \(role.label)…") { store.chooseEndpoint(for: clip.id, role: role) }.disabled(!supported)
        if attachment != nil {
          Button("Remove \(role.label)") { store.removeEndpoint(from: clip.id, role: role) }
        }
      }
      .task(id: asset?.path) {
        thumbnail = nil
        guard let asset, let source = CGImageSourceCreateWithURL(URL(fileURLWithPath: asset.path) as CFURL, nil),
          let image = CGImageSourceCreateThumbnailAtIndex(source, 0, [
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceThumbnailMaxPixelSize: 112,
            kCGImageSourceCreateThumbnailWithTransform: true,
          ] as CFDictionary) else { return }
        thumbnail = NSImage(cgImage: image, size: .zero)
      }
  }
}
