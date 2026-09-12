import AppKit
import StudioCore
import SwiftUI

struct ImageGenerationEditor: View {
  @EnvironmentObject var store: StudioStore
  func binding<T>(_ key: WritableKeyPath<DrawThingsImageDraft, T>, fallback: T) -> Binding<T> {
    Binding(get: { store.imageDraft?[keyPath: key] ?? fallback }, set: { value in
      store.imageDraft?[keyPath: key] = value; store.imageEstimate = nil
    })
  }
  var body: some View {
    VStack(spacing: 0) {
      HStack {
        Button { store.imageDraft = nil } label: { Label("Back to movie", systemImage: "arrow.left") }
          .keyboardShortcut(.escape, modifiers: [])
        Divider().frame(height: 20)
        Text("Generate Image").font(.headline)
        Spacer()
        Text("\(store.imageDraft?.destination.scope.rawValue.capitalized ?? "") assets").foregroundStyle(.secondary)
        Button("Connections…") { store.showDrawThings = true }
      }.padding(24)
      Divider()
      HSplitView {
        VStack(alignment: .leading, spacing: 16) {
          TextField("Image name", text: binding(\.name, fallback: "Generated image")).font(.title2).textFieldStyle(.plain)
          Text("Describe the image you want to create.").foregroundStyle(.secondary)
          TextEditor(text: binding(\.prompt, fallback: "")).font(.system(size: 15))
            .scrollContentBackground(.hidden).padding(12).background(Theme.raised, in: RoundedRectangle(cornerRadius: 8))
            .frame(minHeight: 180, maxHeight: 300)
          if let path = store.imagePreviewPath, let preview = NSImage(contentsOfFile: path) {
            Image(nsImage: preview).resizable().scaledToFit().frame(maxWidth: .infinity, maxHeight: .infinity)
          } else {
            ContentUnavailableView("Image Preview", systemImage: "photo", description: Text("Generated images appear here and in the selected asset store."))
          }
        }.padding(28).frame(minWidth: 500, maxWidth: .infinity)
        Form {
          Picker("Connection", selection: binding(\.profileID, fallback: "")) {
            Text("Choose a connection").tag("")
            ForEach(store.drawThingsConnections) { Text($0.name).tag($0.id) }
          }.onChange(of: store.imageDraft?.profileID) { _, _ in store.imageDraft?.modelID = "" }
          Button("Refresh Models") {
            if let connection = store.drawThingsConnections.first(where: { $0.id == store.imageDraft?.profileID }) {
              Task { await store.testDrawThings(connection) }
            }
          }.disabled(store.bridge.busy || store.imageDraft?.profileID.isEmpty != false)
          Picker("Model", selection: binding(\.modelID, fallback: "")) {
            Text("Choose an image model").tag("")
            ForEach(store.drawThingsModels(store.imageDraft?.profileID ?? "", operation: "image"), id: \.id) {
              Text($0.name).tag($0.id)
            }
          }
          TextField("Width", value: binding(\.width, fallback: 512), format: .number.grouping(.never))
          TextField("Height", value: binding(\.height, fallback: 512), format: .number.grouping(.never))
          Text("Dimensions use multiples of 64.").font(.caption).foregroundStyle(.secondary)
          TextField("Steps", value: binding(\.steps, fallback: 4), format: .number.grouping(.never))
          DisclosureGroup("Advanced") {
            TextField("Seed", value: binding(\.seed, fallback: 42), format: .number.grouping(.never))
            TextField("Guidance", value: binding(\.guidance, fallback: 1), format: .number)
            TextField("Negative prompt", text: binding(\.negativePrompt, fallback: ""), axis: .vertical)
          }
          Divider()
          if let estimate = store.imageEstimate {
            Text("Estimated CU: \(number(estimate["estimateCU"]))")
            Text(estimate["limitMode"] as? String == "notApplicable" ? "Self-hosted · no cloud CU limit" :
              estimate["limitEnforcement"] as? String == "server" ? "CU limit checked by Draw Things on submission" :
              "Current per-job limit: \(number(estimate["limitCU"]))")
              .font(.caption).foregroundStyle(.secondary)
            if let message = estimate["accountMessage"] as? String { Text(message).font(.caption) }
            ForEach(Array((estimate["issues"] as? [[String: Any]] ?? []).enumerated()), id: \.offset) { _, issue in
              Text(issue["message"] as? String ?? "Connection needs attention").font(.caption).foregroundStyle(.secondary)
            }
          } else { Text("Check settings to calculate CU and generation eligibility.").font(.caption).foregroundStyle(.secondary) }
        }.formStyle(.grouped).padding(16).frame(minWidth: 330, idealWidth: 390, maxWidth: 460)
      }
      Divider()
      HStack {
        if store.bridge.busy {
          ProgressView().controlSize(.small)
          Text(store.bridge.message).font(.caption)
          Button("Cancel") { store.bridge.cancel() }
        } else { Text("Each generation is saved as a new image asset.").font(.caption).foregroundStyle(.secondary) }
        Spacer()
        Button("Export Headless Job…") { store.exportDrawThingsImageJob() }
          .disabled(store.bridge.busy || store.imageDraft?.modelID.isEmpty != false)
        Button("Check Settings & CU") { Task { await store.prepareImageGeneration() } }
          .disabled(store.bridge.busy || store.imageDraft?.modelID.isEmpty != false)
        Button("Generate Image") { Task { await store.generateImageAsset() } }
          .buttonStyle(.borderedProminent)
          .disabled(store.bridge.busy || store.imageEstimate?["eligibility"] as? String != "allowed")
      }.padding(24)
    }.background(Theme.background).frame(maxWidth: .infinity, maxHeight: .infinity)
  }
  func number(_ value: Any?) -> String { (value as? NSNumber).map { $0.stringValue } ?? "Unknown" }
}
