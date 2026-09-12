import StudioCore
import SwiftUI

struct DrawThingsClipInspector: View {
  @EnvironmentObject var store: StudioStore
  var clip: Clip
  var isH3: Bool { clip.drawThings?.modelFamily.lowercased() == "minimaxh3" }
  var hasCatalog: Bool { store.drawThingsCatalogs[clip.drawThings?.profileID ?? ""] != nil }
  @State private var groupName = ""
  func selection<T>(_ key: WritableKeyPath<DrawThingsSelection, T>, fallback: T) -> Binding<T> {
    Binding(get: { store.selectedClip?.drawThings?[keyPath: key] ?? fallback }, set: { value in
      store.editClip {
        if $0.drawThings == nil { $0.drawThings = DrawThingsSelection(profileID: "", modelID: "", modelFamily: "") }
        let previousProfile = $0.drawThings?.profileID
        $0.drawThings?[keyPath: key] = value
        if previousProfile != $0.drawThings?.profileID {
          $0.drawThings?.modelID = ""
          $0.drawThings?.modelFamily = ""
        }
        if let selected = $0.drawThings {
          $0.drawThings?.modelFamily = store.drawThingsModelFamily(
            profileID: selected.profileID, modelID: selected.modelID)
        }
      }
    })
  }
  func number(_ key: String, fallback: Int) -> Binding<Int> {
    Binding(get: {
      if case .integer(let value) = store.selectedClip?.drawThings?.configuration[key] { return value }
      return fallback
    }, set: { value in
      store.editClip {
        if $0.drawThings == nil { $0.drawThings = DrawThingsSelection(profileID: "", modelID: "", modelFamily: "") }
        $0.drawThings?.configuration[key] = .integer(value)
      }
    })
  }
  func decimal(_ key: String, fallback: Double) -> Binding<Double> {
    Binding(get: {
      switch store.selectedClip?.drawThings?.configuration[key] {
      case .number(let value): return value
      case .integer(let value): return Double(value)
      default: return fallback
      }
    }, set: { value in store.editClip { $0.drawThings?.configuration[key] = .number(value) } })
  }
  var body: some View {
    VStack(alignment: .leading, spacing: 10) {
      SmallLabel(text: "Draw Things")
      Picker("Task", selection: Binding(get: {
        clip.inferredTask
      }, set: { task in store.editClip {
        $0.generationSelection = GenerationSelection(task: task, preset: .custom)
        if let selected = $0.drawThings,
          store.drawThingsCatalogs[selected.profileID] != nil,
          !store.drawThingsModels(selected.profileID, operation: "video", task: task)
            .contains(where: { $0.id == selected.modelID }) {
          $0.drawThings?.modelID = ""
          $0.drawThings?.modelFamily = ""
        }
      } })) {
        Text("Text to video").tag("t2v")
        Text("Image to video").tag("i2v")
        Text("First and last frames").tag("fflf")
      }
      Text(clip.drawThings?.modelID.isEmpty != false
        ? "Choose your task and add frame images first, then select a connection and model. First and last frames requires H3 FL2VA."
        : clip.inferredTask == "fflf" && !isH3
          ? "Select an H3 FL2VA model for this task. Your frame images are preserved."
          : "Custom server settings · connection capability is validated before generation.")
        .font(.caption2).foregroundStyle(.secondary)
      let connections = store.drawThingsConnections(for: clip.inferredTask)
      Picker("Connection", selection: selection(\.profileID, fallback: "")) {
        Text("Choose a connection").tag("")
        if let saved = store.drawThingsConnections.first(where: { $0.id == clip.drawThings?.profileID }),
          !connections.contains(where: { $0.id == saved.id }) {
          Text("\(saved.name) · no models for this task").tag(saved.id).disabled(true)
        }
        ForEach(connections) { connection in
          Text(connection.name + (store.drawThingsCatalogs[connection.id] == nil ? " · refresh to verify" : ""))
            .tag(connection.id)
        }
      }
      HStack {
        Button("Connections…") { store.showDrawThings = true }
        Button("Import Config…") { store.configImportClipID = clip.id; store.showDrawThingsConfigImport = true }
        Button("Refresh") {
          if let connection = store.drawThingsConnections.first(where: { $0.id == clip.drawThings?.profileID }) {
            Task { await store.testDrawThings(connection) }
          }
        }.disabled(store.bridge.busy)
      }.font(.caption)
      let models = store.drawThingsModels(clip.drawThings?.profileID ?? "", operation: "video", task: clip.inferredTask)
      Picker("Model", selection: selection(\.modelID, fallback: "")) {
        Text("Choose a video model").tag("")
        if let savedModelID = clip.drawThings?.modelID, !savedModelID.isEmpty,
          !models.contains(where: { $0.id == savedModelID }) {
          Text("\(savedModelID) (\(hasCatalog ? "unavailable for this task/connection" : "saved · connect to verify"))").tag(savedModelID).disabled(true)
        }
        ForEach(models, id: \.id) {
          Text($0.name).tag($0.id)
        }
      }
      if hasCatalog && models.isEmpty {
        Text("This connection advertises no models for the selected task. Choose another connection or Refresh after installing a compatible model.")
          .font(.caption2).foregroundStyle(.orange)
      }
      HStack {
        Text("Steps")
        TextField("Steps", value: number("steps", fallback: isH3 ? 50 : 8), format: .number.grouping(.never))
      }
      HStack {
        Text("CFG")
        TextField("CFG", value: decimal("guidanceScale", fallback: 1), format: .number)
      }
      Toggle("Override Shift", isOn: Binding(get: { clip.drawThings?.configuration["shift"] != nil },
        set: { enabled in store.editClip { $0.drawThings?.configuration["shift"] = enabled ? .number(isH3 ? 12 : 1) : nil } }))
      if clip.drawThings?.configuration["shift"] != nil {
        HStack {
          Text("Shift")
          TextField("Shift", value: decimal("shift", fallback: isH3 ? 12 : 1), format: .number)
        }
      }
      if isH3 {
        HStack {
          Text("Audio Shift")
          TextField("Audio Shift", value: decimal("audioShift", fallback: 3), format: .number)
        }
        Text("H3 defaults: DDIM Trailing, Shift 12, Audio Shift 3. Generation uses 24 FPS.")
          .font(.caption2).foregroundStyle(.secondary)
      }
      HStack {
        Text("Generation FPS")
        TextField("Generation FPS", value: number("fps", fallback: isH3 ? 24 : Int(clip.settings(in: store.project).fps)),
                  format: .number.grouping(.never))
      }
      Text("Frame count rounds up to the model’s valid duration. Movie finishing applies the project frame rate.")
        .font(.caption2).foregroundStyle(.secondary)
      if isH3 && clip.attachments.contains(where: { $0.role == .last }) {
        Text("After generation, the clip adopts the rounded duration to preserve its last frame.")
          .font(.caption2).foregroundStyle(.secondary)
      }
      Divider()
      Text("Server LoRAs").font(.caption.bold())
      let available = store.drawThingsLoRAs(
        profileID: clip.drawThings?.profileID ?? "", modelID: clip.drawThings?.modelID ?? "")
      ForEach(available, id: \.id) { lora in
        HStack {
          Toggle(lora.name, isOn: Binding(get: {
            store.selectedClip?.drawThings?.loras.contains { $0.modelID == lora.id } == true
          }, set: { enabled in
            store.editClip { selected in
              guard var value = selected.drawThings else { return }
              value.loras.removeAll { $0.modelID == lora.id }
              if enabled { value.loras.append(DrawThingsLoRA(modelID: lora.id)) }
              selected.drawThings = value
            }
          }))
          if let index = clip.drawThings?.loras.firstIndex(where: { $0.modelID == lora.id }) {
            TextField("Strength", value: Binding(get: {
              store.selectedClip?.drawThings?.loras[safe: index]?.weight ?? 1
            }, set: { weight in store.editClip { $0.drawThings?.loras[safe: index]?.weight = weight } }),
              format: .number).frame(width: 58)
          }
        }
      }
      let unavailable = clip.drawThings?.unavailableLoRAs(
        availableIDs: hasCatalog ? Set(available.map(\.id)) : nil) ?? []
      let saved = hasCatalog ? unavailable : clip.drawThings?.loras ?? []
      if !saved.isEmpty {
        VStack(alignment: .leading, spacing: 6) {
          Text(hasCatalog ? "Unavailable for this connection or model" : "Saved LoRAs · connection not verified")
            .font(.caption.bold()).foregroundStyle(hasCatalog ? .orange : .secondary)
          ForEach(saved) { lora in
            HStack {
              VStack(alignment: .leading, spacing: 2) {
                Text(lora.modelID).lineLimit(1)
              }
              Spacer()
              TextField("Strength", value: Binding(get: {
                store.selectedClip?.drawThings?.loras.first { $0.modelID == lora.modelID }?.weight ?? lora.weight
              }, set: { weight in
                store.editClip { selected in
                  if let index = selected.drawThings?.loras.firstIndex(where: { $0.modelID == lora.modelID }) {
                    selected.drawThings?.loras[index].weight = weight
                  }
                }
              }), format: .number).frame(width: 58)
              Button("Remove") { removeLoRA(lora.modelID) }
            }
          }
          if hasCatalog { Button("Remove all unavailable") {
            let ids = Set(unavailable.map(\.modelID))
            store.editClip { $0.drawThings?.loras.removeAll { ids.contains($0.modelID) } }
          }.controlSize(.small) }
        }.padding(8).background(.secondary.opacity(0.08), in: RoundedRectangle(cornerRadius: 6))
      }
      if available.isEmpty {
        Text(hasCatalog
          ? "No compatible LoRAs were advertised for this model. Install them in Draw Things, enable Model Browsing for a local server, then Refresh."
          : "Connect and Refresh to load compatible LoRAs. For a local connection, enable Draw Things’ gRPC API server and Model Browsing. Saved assignments remain editable; generation verifies compatibility.")
          .font(.caption2).foregroundStyle(.secondary)
      }
      HStack {
        TextField("Group name", text: $groupName)
        Button("Save") { saveGroup() }.disabled(!hasCatalog || groupName.trimmingCharacters(in: .whitespaces).isEmpty || clip.drawThings?.loras.isEmpty != false)
      }
      Menu("Apply LoRA group") {
        ForEach(compatibleGroups) { group in Button(group.name) { apply(group) } }
      }.disabled(compatibleGroups.isEmpty)
      DrawThingsCUStatus(clip: clip)
    }.font(.caption).textFieldStyle(.roundedBorder)
  }
  var compatibleGroups: [DrawThingsLoRAGroup] {
    guard let selection = clip.drawThings else { return [] }
    return store.drawThingsLoRAGroups.filter {
      $0.profileID == selection.profileID && $0.family == selection.modelFamily
        && $0.compatibleModelIDs.contains(selection.modelID)
    }
  }
  func saveGroup() {
    guard let selection = clip.drawThings else { return }
    let discovered = store.drawThingsLoRAs(profileID: selection.profileID, modelID: selection.modelID)
    guard let family = discovered.first(where: { candidate in selection.loras.contains { $0.modelID == candidate.id } })?.family,
      selection.loras.allSatisfy({ member in discovered.contains { $0.id == member.modelID && $0.family == family } })
    else { store.error = "A Draw Things LoRA group cannot mix model families."; return }
    let compatible = discovered.filter { candidate in selection.loras.contains { $0.modelID == candidate.id } }
      .reduce(Set([selection.modelID])) { $0.intersection(Set($1.compatibleModelIDs)) }
    store.drawThingsLoRAGroups.append(DrawThingsLoRAGroup(name: groupName, profileID: selection.profileID,
      family: family, compatibleModelIDs: Array(compatible).sorted(), members: selection.loras))
    store.saveDrawThingsLoRAGroups(); groupName = ""
  }
  func apply(_ group: DrawThingsLoRAGroup) {
    store.editClip { selected in
      guard var selection = selected.drawThings,
        (try? group.validate(profileID: selection.profileID, family: selection.modelFamily, modelID: selection.modelID)) != nil else { return }
      selection.apply(group); selected.drawThings = selection
    }
  }
  func removeLoRA(_ modelID: String) {
    store.editClip { $0.drawThings?.loras.removeAll { $0.modelID == modelID } }
  }
}

private extension Array {
  subscript(safe index: Index) -> Element? {
    get { indices.contains(index) ? self[index] : nil }
    set { if indices.contains(index), let value = newValue { self[index] = value } }
  }
}

struct DrawThingsCUStatus: View {
  @EnvironmentObject var store: StudioStore
  var clip: Clip
  var body: some View {
    if let estimate = store.drawThingsClipEstimates[clip.id],
      estimate["studioSignature"] as? String == store.signature(for: clip) {
      VStack(alignment: .leading, spacing: 5) {
        Text("Estimated CU: \((estimate["estimateCU"] as? NSNumber)?.stringValue ?? "Unknown")")
        Text(estimate["limitMode"] as? String == "notApplicable" ? "Self-hosted · no cloud limit" :
          estimate["limitEnforcement"] as? String == "server" ? "CU limit checked by Draw Things on submission" :
          "Per-job limit: \((estimate["limitCU"] as? NSNumber)?.stringValue ?? "Unknown")")
        if let message = estimate["accountMessage"] as? String { Text(message) }
        if estimate["eligibility"] as? String == "blocked" {
          Text("Adjust size, duration, or steps, then prepare again.")
        }
      }.font(.caption).foregroundStyle(.secondary)
    } else {
      Text("Prepare this clip to calculate CU and check the connection.").font(.caption).foregroundStyle(.secondary)
    }
  }
}
