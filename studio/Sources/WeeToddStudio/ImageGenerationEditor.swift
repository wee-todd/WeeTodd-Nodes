import AppKit
import ImageIO
import StudioCore
import SwiftUI
import UniformTypeIdentifiers

struct ImageGenerationEditor: View {
  @EnvironmentObject var store: StudioStore
  @State private var showResult = true
  @State private var zoom = 1.0
  @State private var groupName = ""
  @State private var referencePreview: String?
  var draft: DrawThingsImageDraft? { store.imageDraft }
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
        Text("Image Workspace").font(.headline)
        Button("Import Config…") { store.configImportClipID = nil; store.showDrawThingsConfigImport = true }
        Link("Draw Things presets", destination: DrawThingsConfigImport.presetsURL).font(.caption)
        TextField("Image name", text: binding(\.name, fallback: "Generated image")).frame(maxWidth: 260)
        Spacer()
        Text("\(draft?.destination.scope.rawValue.capitalized ?? "") assets").foregroundStyle(.secondary)
      }.padding(16)
      Divider()
      HSplitView {
        settings.frame(minWidth: 260, idealWidth: 300, maxWidth: 370)
        VStack(spacing: 12) {
          HStack {
            Text(showResult && store.imagePreviewPath != nil ? "RESULT" : "CANVAS").font(.caption.bold())
            if store.imagePreviewPath != nil {
              Picker("Display", selection: $showResult) { Text("Input").tag(false); Text("Result").tag(true) }.pickerStyle(.segmented).frame(width: 170)
              Button("Use result as canvas") {
                if let path = store.imagePreviewPath { store.loadImageInputs([URL(fileURLWithPath: path)], canvas: true); showResult = false }
              }
            }
            Spacer()
            Button("Fit") { zoom = 1 }
            Slider(value: $zoom, in: 0.5...2).frame(width: 85).help("Canvas zoom")
          }.font(.caption)
          GeometryReader { geometry in
            let path = showResult ? store.imagePreviewPath ?? draft?.canvas?.path : draft?.canvas?.path
            ZStack {
              Color.black.opacity(0.9)
              if let path {
                WorkspaceImage(path: path, fill: !(showResult && store.imagePreviewPath != nil) && draft?.canvas?.fit == "fill")
                  .aspectRatio(CGFloat(draft?.width ?? 512) / CGFloat(draft?.height ?? 512), contentMode: .fit)
                  .scaleEffect(zoom)
              } else {
                VStack(spacing: 12) {
                  Image(systemName: "photo.badge.plus").font(.largeTitle)
                  Text("Drop a canvas image here").font(.headline)
                  Text("Or leave it empty for text-to-image / mood-board generation.").font(.caption)
                  Button("Load image…") { store.chooseImageInputs(canvas: true) }
                }.foregroundStyle(.white)
              }
            }.frame(width: geometry.size.width, height: geometry.size.height).clipped()
              .onDrop(of: [UTType.fileURL.identifier, UTType.text.identifier], isTargeted: nil) { store.dropImageInputs($0, canvas: true) }
          }.frame(minHeight: 220)
          HStack {
            Button("Load canvas…") { store.chooseImageInputs(canvas: true); showResult = false }
            assetsMenu(canvas: true)
            if draft?.canvas != nil {
              Toggle("Use canvas", isOn: Binding(get: { draft?.canvas?.enabled ?? false }, set: { store.imageDraft?.canvas?.enabled = $0; store.imageEstimate = nil }))
              Picker("Placement", selection: Binding(get: { draft?.canvas?.fit ?? "fit" }, set: { store.imageDraft?.canvas?.fit = $0; store.imageEstimate = nil })) {
                Text("Fit · preserve image").tag("fit"); Text("Fill · crop edges").tag("fill")
              }.frame(maxWidth: 190)
              Button("Clear") { store.imageDraft?.canvas = nil; store.imageEstimate = nil }
            }
            Spacer()
          }.font(.caption)
          TextEditor(text: binding(\.prompt, fallback: "")).font(.system(size: 14))
            .scrollContentBackground(.hidden).padding(10).background(Theme.raised, in: RoundedRectangle(cornerRadius: 8))
            .frame(height: 155).overlay(alignment: .topLeading) {
              if draft?.prompt.isEmpty != false { Text("Describe the image or edit…").foregroundStyle(.secondary).padding(15).allowsHitTesting(false) }
            }
        }.padding(16).frame(minWidth: 470, maxWidth: .infinity)
        moodboard.frame(minWidth: 205, idealWidth: 235, maxWidth: 290)
      }
      Divider()
      HStack {
        if store.bridge.busy {
          ProgressView().controlSize(.small)
          Text(store.bridge.message).font(.caption)
          Button("Cancel") { store.bridge.cancel() }
        } else { Text("Each generation is saved as a new image asset.").font(.caption).foregroundStyle(.secondary) }
        Spacer()
        Button("Export Headless Job…") { store.exportDrawThingsImageJob() }.disabled(store.bridge.busy || draft?.modelID.isEmpty != false)
        Button("Check Settings & CU") { Task { await store.prepareImageGeneration() } }.disabled(store.bridge.busy || draft?.modelID.isEmpty != false)
        Button("Generate Image") { Task { await store.generateImageAsset() } }.buttonStyle(.borderedProminent)
          .disabled(store.bridge.busy || store.imageEstimate?["eligibility"] as? String != "allowed")
      }.padding(16)
    }.background(Theme.background).frame(maxWidth: .infinity, maxHeight: .infinity)
      .onChange(of: store.imagePreviewPath) { _, path in if path != nil { showResult = true } }
      .sheet(isPresented: Binding(get: { referencePreview != nil }, set: { if !$0 { referencePreview = nil } })) {
        VStack {
          if let path = referencePreview { WorkspaceImage(path: path).frame(width: 720, height: 580) }
          Button("Done") { referencePreview = nil }
        }.padding()
      }
  }
  var settings: some View {
    Form {
      Section("Draw Things") {
        Picker("Connection", selection: binding(\.profileID, fallback: "")) {
          Text("Choose a connection").tag("")
          ForEach(store.drawThingsConnections) { Text($0.name).tag($0.id) }
        }.onChange(of: draft?.profileID) { _, _ in
          store.imageDraft?.modelID = ""; store.imageDraft?.loras = []
        }
        HStack {
          Button("Connections…") { store.showDrawThings = true }
          Button("Refresh") {
            if let connection = store.drawThingsConnections.first(where: { $0.id == draft?.profileID }) { Task { await store.testDrawThings(connection) } }
          }.disabled(store.bridge.busy)
        }
        let models = store.imageModelsForInputs()
        Picker("Model", selection: binding(\.modelID, fallback: "")) {
          Text("Choose an image model").tag("")
          if let id = draft?.modelID, !id.isEmpty, !models.contains(where: { $0.id == id }) {
            Text("\(id) · refresh / check inputs").tag(id).disabled(true)
          }
          ForEach(models, id: \.id) { Text($0.name).tag($0.id) }
        }.onChange(of: draft?.modelID) { _, _ in
          let ids = Set(store.drawThingsLoRAs(profileID: draft?.profileID ?? "", modelID: draft?.modelID ?? "").map(\.id))
          store.imageDraft?.loras.removeAll { !ids.contains($0.modelID) }
        }
        Text("Models are filtered by the enabled canvas and mood-board inputs.").font(.caption2).foregroundStyle(.secondary)
      }
      Section("Generation") {
        TextField("Width", value: binding(\.width, fallback: 512), format: .number.grouping(.never))
        TextField("Height", value: binding(\.height, fallback: 512), format: .number.grouping(.never))
        Text("Dimensions: multiples of 64").font(.caption2).foregroundStyle(.secondary)
        TextField("Steps", value: binding(\.steps, fallback: 4), format: .number.grouping(.never))
        TextField("CFG", value: binding(\.guidance, fallback: 1), format: .number)
        if draft?.canvas?.enabled == true {
          Text("Generation strength · \(Int((draft?.strength ?? 1) * 100))%")
          Slider(value: binding(\.strength, fallback: 1), in: 0...1)
          Text("Higher values allow more regeneration. Editing models also use the canvas as a reference.").font(.caption2).foregroundStyle(.secondary)
        }
        TextField("Seed", value: binding(\.seed, fallback: 42), format: .number.grouping(.never))
        Button("Randomize seed") { store.imageDraft?.seed = Int.random(in: 0...Int(UInt32.max)); store.imageEstimate = nil }
        Picker("Sampler", selection: binding(\.sampler, fallback: nil)) {
          Text("Server default").tag(nil as Int?)
          ForEach(Array(Self.samplers.enumerated()), id: \.offset) { index, name in Text(name).tag(Optional(index)) }
        }
        Toggle("Override Shift", isOn: Binding(get: { draft?.shift != nil }, set: { store.imageDraft?.shift = $0 ? 1 : nil; store.imageEstimate = nil }))
        if draft?.shift != nil { TextField("Shift", value: Binding(get: { draft?.shift ?? 1 }, set: { store.imageDraft?.shift = $0; store.imageEstimate = nil }), format: .number) }
        DisclosureGroup("Negative prompt") { TextField("Negative prompt", text: binding(\.negativePrompt, fallback: ""), axis: .vertical) }
      }
      Section("LoRAs & Groups") { loraControls }
      Section("Preflight") {
        if let estimate = store.imageEstimate {
          Text("Estimated CU: \(number(estimate["estimateCU"]))")
          Text(estimate["limitMode"] as? String == "notApplicable" ? "Self-hosted · no cloud CU limit" : "CU eligibility checked for this request").font(.caption)
          ForEach(Array((estimate["issues"] as? [[String: Any]] ?? []).enumerated()), id: \.offset) { _, issue in Text(issue["message"] as? String ?? "Connection needs attention").font(.caption) }
        } else { Text("Check Settings & CU after changing inputs or settings.").font(.caption).foregroundStyle(.secondary) }
      }
    }.formStyle(.grouped)
  }
  var moodboard: some View {
    VStack(alignment: .leading, spacing: 12) {
      HStack { Text("MOOD BOARD").font(.caption.bold()); Spacer(); Button { store.chooseImageInputs(canvas: false) } label: { Image(systemName: "plus") } }
      assetsMenu(canvas: false)
      Text("Ordered references · up to 8").font(.caption2).foregroundStyle(.secondary)
      ScrollView {
        VStack(spacing: 14) {
          ForEach(Array((draft?.moodboard ?? []).enumerated()), id: \.element.id) { index, item in
            VStack(spacing: 6) {
              WorkspaceImage(path: item.path).frame(height: 115).background(.black.opacity(0.15)).clipped()
                .onTapGesture { referencePreview = item.path }
                .help("Click to inspect this reference")
              HStack {
                Toggle("Reference \(index + 1)", isOn: referenceBinding(item.id, \.enabled, item.enabled))
                Spacer()
                Button { store.imageDraft?.moodboard.removeAll { $0.id == item.id }; store.imageEstimate = nil } label: { Image(systemName: "xmark") }
              }
              Text(URL(fileURLWithPath: item.path).lastPathComponent).font(.caption2).lineLimit(1)
              Button("Replace…") { store.replaceImageReference(item.id) }
              HStack {
                Text("Strength \(Int(item.strength * 100))%")
                Spacer()
                Button { moveReference(index, -1) } label: { Image(systemName: "arrow.up") }.disabled(index == 0)
                Button { moveReference(index, 1) } label: { Image(systemName: "arrow.down") }.disabled(index == (draft?.moodboard.count ?? 0) - 1)
              }
              Slider(value: referenceBinding(item.id, \.strength, item.strength), in: 0...1)
            }.font(.caption).padding(9).background(Theme.raised, in: RoundedRectangle(cornerRadius: 8)).opacity(item.enabled ? 1 : 0.5)
              .draggable("mood:" + item.id.uuidString)
              .onDrop(of: [UTType.text.identifier], isTargeted: nil) { providers in
                guard let provider = providers.first else { return false }
                provider.loadObject(ofClass: NSString.self) { value, _ in
                  let text = value as? String ?? ""
                  Task { @MainActor in
                    if text.hasPrefix("mood:"), let id = UUID(uuidString: String(text.dropFirst(5))),
                      let from = store.imageDraft?.moodboard.firstIndex(where: { $0.id == id }),
                      let to = store.imageDraft?.moodboard.firstIndex(where: { $0.id == item.id }) {
                      store.imageDraft?.moodboard.move(fromOffsets: IndexSet(integer: from), toOffset: to > from ? to + 1 : to)
                      store.imageEstimate = nil
                    } else { _ = store.dropImageInputs(providers, canvas: false) }
                  }
                }
                return true
              }
          }
          if draft?.moodboard.isEmpty != false { Text("Drop reference images here").foregroundStyle(.secondary).frame(maxWidth: .infinity, minHeight: 140) }
        }
      }.onDrop(of: [UTType.fileURL.identifier, UTType.text.identifier], isTargeted: nil) { store.dropImageInputs($0, canvas: false) }
      Text("FLUX.2/Klein treats positive reference weights as enabled images. Weight is sent unchanged, but may not scale influence. 0% omits the reference.").font(.caption2).foregroundStyle(.secondary)
    }.padding(14)
  }
  func referenceBinding<T>(_ id: UUID, _ key: WritableKeyPath<ImageWorkspaceInput, T>, _ fallback: T) -> Binding<T> {
    Binding(get: { draft?.moodboard.first { $0.id == id }?[keyPath: key] ?? fallback }, set: { value in
      if let index = store.imageDraft?.moodboard.firstIndex(where: { $0.id == id }) { store.imageDraft?.moodboard[index][keyPath: key] = value; store.imageEstimate = nil }
    })
  }
  func moveReference(_ index: Int, _ offset: Int) {
    store.imageDraft?.moodboard.swapAt(index, index + offset); store.imageEstimate = nil
  }
  func assetsMenu(canvas: Bool) -> some View {
    Menu("From Assets") {
      ForEach(store.allAssets.filter { $0.kind == .image }) { asset in
        Button(asset.name) { store.loadImageInputs([URL(fileURLWithPath: asset.path)], canvas: canvas) }
      }
    }.disabled(!store.allAssets.contains { $0.kind == .image })
  }
  var loraControls: some View {
    let available = store.drawThingsLoRAs(profileID: draft?.profileID ?? "", modelID: draft?.modelID ?? "")
    let groups = store.drawThingsLoRAGroups.filter { $0.profileID == draft?.profileID && $0.compatibleModelIDs.contains(draft?.modelID ?? "") }
    return VStack(alignment: .leading) {
      ForEach(available, id: \.id) { lora in
        Toggle(lora.name, isOn: Binding(get: { draft?.loras.contains { $0.modelID == lora.id } == true }, set: { enabled in
          store.imageDraft?.loras.removeAll { $0.modelID == lora.id }
          if enabled { store.imageDraft?.loras.append(DrawThingsLoRA(modelID: lora.id)) }; store.imageEstimate = nil
        }))
      }
      ForEach(draft?.loras ?? []) { lora in
        HStack {
          Text(lora.modelID).lineLimit(1).help(lora.modelID)
          TextField("Weight", value: Binding(get: { draft?.loras.first { $0.modelID == lora.id }?.weight ?? 1 }, set: { value in
            if let index = store.imageDraft?.loras.firstIndex(where: { $0.modelID == lora.id }) { store.imageDraft?.loras[index].weight = value; store.imageEstimate = nil }
          }), format: .number).frame(width: 55)
          Button { store.imageDraft?.loras.removeAll { $0.modelID == lora.id }; store.imageEstimate = nil } label: { Image(systemName: "xmark") }
        }.font(.caption)
      }
      Menu("Apply group") {
        ForEach(groups) { group in storeGroupButton(group) }
      }.disabled(groups.isEmpty)
      HStack {
        TextField("Group name", text: $groupName)
        Button("Save") { saveGroup(available) }.disabled(groupName.trimmingCharacters(in: .whitespaces).isEmpty || draft?.loras.isEmpty != false)
      }
    }
  }
  func storeGroupButton(_ group: DrawThingsLoRAGroup) -> some View {
    Button(group.name) { store.imageDraft?.loras = group.members; store.imageEstimate = nil }
  }
  func saveGroup(_ available: [StudioStore.DiscoveredDrawThingsLoRA]) {
    guard let draft else { return }
    var ids = Set(available.first(where: { $0.id == draft.loras.first?.modelID })?.compatibleModelIDs ?? [])
    for lora in draft.loras {
      guard let entry = available.first(where: { $0.id == lora.modelID }) else { store.error = "Refresh and select compatible LoRAs before saving a group."; return }
      ids.formIntersection(entry.compatibleModelIDs)
    }
    store.drawThingsLoRAGroups.append(DrawThingsLoRAGroup(name: groupName, profileID: draft.profileID,
      family: store.drawThingsModelFamily(profileID: draft.profileID, modelID: draft.modelID), compatibleModelIDs: Array(ids), members: draft.loras))
    store.saveDrawThingsLoRAGroups(); groupName = ""
  }
  func number(_ value: Any?) -> String { (value as? NSNumber).map { $0.stringValue } ?? "Unknown" }
  static let samplers = ["DPM++ 2M Karras", "Euler A", "DDIM", "PLMS", "DPM++ SDE Karras", "UniPC", "LCM", "Euler A Substep", "DPM++ SDE Substep", "TCD", "Euler A Trailing", "DPM++ SDE Trailing", "DPM++ 2M AYS", "Euler A AYS", "DPM++ SDE AYS", "DPM++ 2M Trailing", "DDIM Trailing", "UniPC Trailing", "UniPC AYS", "TCD Trailing"]
}

struct WorkspaceImage: View {
  var path: String
  var fill = false
  @State private var image: NSImage?
  var body: some View {
    Group {
      if let image { Image(nsImage: image).resizable().aspectRatio(contentMode: fill ? .fill : .fit) }
      else { Image(systemName: "photo").foregroundStyle(.secondary) }
    }.task(id: path) {
      guard let source = CGImageSourceCreateWithURL(URL(fileURLWithPath: path) as CFURL, nil),
        let thumbnail = CGImageSourceCreateThumbnailAtIndex(source, 0, [kCGImageSourceCreateThumbnailFromImageAlways: true,
          kCGImageSourceCreateThumbnailWithTransform: true, kCGImageSourceThumbnailMaxPixelSize: 1200] as CFDictionary) else { image = nil; return }
      image = NSImage(cgImage: thumbnail, size: .zero)
    }
  }
}
