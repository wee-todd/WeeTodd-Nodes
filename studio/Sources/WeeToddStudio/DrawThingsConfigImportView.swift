import AppKit
import StudioCore
import SwiftUI

struct DrawThingsConfigImportView: View {
  @EnvironmentObject var store: StudioStore
  @State private var text = ""
  @State private var parsedText = ""
  @State private var candidates: [DrawThingsConfigImport] = []
  @State private var selection = 0
  @State private var includePrompt = true
  @State private var acknowledgeOmissions = false
  @State private var modelOverride = ""
  @State private var message: String?
  var selected: DrawThingsConfigImport? { candidates.indices.contains(selection) ? candidates[selection] : nil }
  var availableModels: [(id: String, name: String)] {
    let clip = store.project.clips.first { $0.id == store.configImportClipID }
    let profileID = clip?.drawThings?.profileID ?? store.imageDraft?.profileID ?? ""
    return store.drawThingsModels(profileID, operation: clip == nil ? "image" : "video")
  }
  var body: some View {
    VStack(alignment: .leading, spacing: 14) {
      HStack {
        Text("Import Draw Things Config").font(.title2.bold())
        Spacer()
        Link("Draw Things presets ↗", destination: DrawThingsConfigImport.presetsURL)
      }
      Text("Load a JSON export or paste a configuration. Your connection and linked images stay in place.").foregroundStyle(.secondary)
      HStack {
        Button("Open JSON…") {
          let panel = NSOpenPanel(); panel.allowedContentTypes = [.json]
          guard panel.runModal() == .OK, let url = panel.url else { return }
          do {
            guard (try url.resourceValues(forKeys: [.fileSizeKey]).fileSize ?? 0) <= 2 * 1024 * 1024 else {
              throw StudioError.invalid("Choose a config smaller than 2 MiB.")
            }
            text = try String(contentsOf: url, encoding: .utf8); preview()
          } catch { message = error.localizedDescription }
        }
        Button("Preview Import") { preview() }
      }
      TextEditor(text: $text).font(.system(.body, design: .monospaced)).frame(height: 170)
        .onChange(of: text) { _, _ in
          if text != parsedText { candidates = []; acknowledgeOmissions = false }
        }
      if let selected {
        Picker("Preset", selection: $selection) {
          ForEach(candidates.indices, id: \.self) { Text(candidates[$0].name).tag($0) }
        }.onChange(of: selection) { _, _ in acknowledgeOmissions = false; modelOverride = "" }
        Picker("Model to use", selection: $modelOverride) {
          Text("Use model specified by config").tag("")
          ForEach(availableModels, id: \.id) { Text($0.name).tag($0.id) }
        }
        if !modelOverride.isEmpty {
          Text("You are replacing the preset’s model with this connection’s selected model. Prepare will check compatibility.")
            .font(.caption).foregroundStyle(.secondary)
        }
        ScrollView {
          VStack(alignment: .leading, spacing: 6) {
            Text("Model: \(selected.modelID ?? "Keep current model")")
            ForEach(selected.configuration.keys.sorted(), id: \.self) { key in
              Text("\(key): \(display(selected.configuration[key]!))").font(.caption.monospaced())
            }
            if let loras = selected.loras { Text("LoRAs: \(loras.isEmpty ? "clear assignments" : loras.map(\.modelID).joined(separator: ", "))") }
            ForEach(selected.warnings, id: \.self) { Text($0).foregroundStyle(.orange).font(.caption) }
          }.frame(maxWidth: .infinity, alignment: .leading)
        }.frame(height: 180)
        Toggle("Include prompts when present", isOn: $includePrompt)
        if !selected.warnings.isEmpty {
          Toggle("Apply supported settings only; omit the listed settings", isOn: $acknowledgeOmissions)
        }
      }
      if let message { Text(message).foregroundStyle(.red).font(.caption).textSelection(.enabled) }
      HStack {
        Text("Model and LoRA availability are checked against the selected connection.").font(.caption).foregroundStyle(.secondary)
        Spacer()
        Button("Cancel") { store.showDrawThingsConfigImport = false }.keyboardShortcut(.cancelAction)
        Button("Apply Config") { apply() }.buttonStyle(.borderedProminent)
          .disabled(selected == nil || (selected?.warnings.isEmpty == false && !acknowledgeOmissions))
      }
    }.padding(22).frame(width: 740).disabled(store.bridge.busy)
  }
  func display(_ value: JSONValue) -> String {
    (try? String(data: JSONEncoder().encode(value), encoding: .utf8)) ?? ""
  }
  func preview() {
    do {
      parsedText = text
      candidates = try DrawThingsConfigImport.parse(Data(text.utf8), operation: store.configImportClipID == nil ? "image" : "video")
      selection = 0; acknowledgeOmissions = false; modelOverride = ""; message = nil
    } catch { candidates = []; message = error.localizedDescription }
  }
  func apply() {
    guard var result = selected else { return }
    if !modelOverride.isEmpty { result.modelID = modelOverride }
    let clip = store.project.clips.first { $0.id == store.configImportClipID }
    let profileID = clip?.drawThings?.profileID ?? store.imageDraft?.profileID ?? ""
    let operation = clip == nil ? "image" : "video"
    let models = store.drawThingsModels(profileID, operation: operation)
    if let id = result.modelID {
      if let match = models.first(where: { $0.id == id }) { result.modelID = match.id }
      else {
        let matches = models.filter { $0.name == id }
        guard matches.count == 1 else { message = "This connection does not advertise ‘\(id)’. Choose an installed model above, or refresh after installing the preset’s exact model in Draw Things."; return }
        result.modelID = matches[0].id
      }
    }
    let modelID = result.modelID ?? clip?.drawThings?.modelID ?? store.imageDraft?.modelID ?? ""
    if let loras = result.loras {
      let available = Set(store.drawThingsLoRAs(profileID: profileID, modelID: modelID).map(\.id))
      let missing = loras.filter { !available.contains($0.modelID) }
      guard missing.isEmpty else { message = "Missing or incompatible LoRAs: " + missing.map(\.modelID).joined(separator: ", "); return }
    }
    if let clipID = store.configImportClipID {
      guard clip?.engine == .drawThings else { message = "The destination clip no longer exists."; return }
      let family = store.drawThingsModelFamily(profileID: profileID, modelID: modelID)
      store.change { project in
        guard let index = project.clips.firstIndex(where: { $0.id == clipID }) else { return }
        if result.modelID != nil { project.clips[index].drawThings?.modelID = modelID; project.clips[index].drawThings?.modelFamily = family }
        if let loras = result.loras { project.clips[index].drawThings?.loras = loras }
        if includePrompt {
          if let prompt = result.prompt { project.clips[index].prompt = prompt }
          if let negative = result.negativePrompt { project.clips[index].negativePrompt = negative }
        }
        for (key, value) in result.configuration {
          switch (key, value) {
          case ("width", .integer(let n)): project.clips[index].generationWidth = n
          case ("height", .integer(let n)): project.clips[index].generationHeight = n
          case ("seed", .integer(let n)): project.clips[index].seed = n
          default: project.clips[index].drawThings?.configuration[key] = value
          }
        }
        if case .integer(let frames) = result.configuration["numFrames"],
          case .integer(let fps) = result.configuration["fps"] { project.clips[index].duration = Double(frames) / Double(fps) }
      }
    } else if var draft = store.imageDraft {
      result.apply(to: &draft, includePrompt: includePrompt)
      store.imageDraft = draft; store.imageEstimate = nil
    }
    store.notice = "Imported \(result.name). Prepare to validate the complete request."
    store.showDrawThingsConfigImport = false
  }
}
