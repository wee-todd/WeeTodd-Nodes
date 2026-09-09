import AppKit
import Foundation
import StudioCore

struct DrawThingsImageDraft: Equatable {
  var destination: ImageAssetDestination
  var name = "Generated image"
  var profileID = ""
  var modelID = ""
  var prompt = ""
  var negativePrompt = ""
  var width = 512
  var height = 512
  var steps = 4
  var seed = 42
  var guidance = 1.0
  var configuration: [String: Any] {
    ["width": width, "height": height, "steps": steps, "seed": seed, "guidanceScale": guidance]
  }
  func request(id: String) -> [String: Any] {
    ["schema": "weetodd-drawthings-request-v1", "requestID": id, "operation": "image",
     "profileID": profileID, "modelID": modelID, "prompt": prompt, "negativePrompt": negativePrompt,
     "configuration": configuration, "inputs": [], "loras": [], "billingPolicy": "freeOnly"]
  }
}

struct PreparedDrawThingsClip {
  var projectID: UUID
  var clipID: UUID
  var signature: String
  var connection: DrawThingsConnection
}

extension StudioStore {
  struct DiscoveredDrawThingsLoRA {
    var id: String; var name: String; var family: String; var compatibleModelIDs: [String]
  }
  var canGenerateSelected: Bool {
    guard let clip = selectedClip else { return false }
    guard clip.engine == .drawThings else { return preparedRecipe != nil }
    guard let prepared = preparedDrawThingsClip else { return false }
    return prepared.projectID == project.id && prepared.clipID == clip.id
      && prepared.signature == signature(for: clip)
      && drawThingsConnections.contains(prepared.connection)
  }
  func drawThingsModelFamily(profileID: String, modelID: String) -> String {
    (drawThingsCatalogs[profileID]?["models"] as? [[String: Any]])?
      .first(where: { $0["id"] as? String == modelID })?["family"] as? String ?? ""
  }
  func prepareDrawThingsClip() async {
    guard let clip = selectedClip,
      let connection = drawThingsConnections.first(where: { $0.id == clip.drawThings?.profileID }) else {
      error = "Choose a Draw Things connection in the clip inspector."
      return
    }
    preparedDrawThingsClip = nil
    let projectID = project.id
    let snapshot = signature(for: clip)
    do {
      var body = try payload()
      body["connection"] = try connection.object()
      var result = try await bridge.invoke("dt-prepare-clip", runtime: runtime, payload: body)
      guard project.id == projectID, selectedClipID == clip.id,
        selectedClip.map({ signature(for: $0) }) == snapshot,
        drawThingsConnections.contains(connection) else {
        throw StudioError.invalid("Clip or connection changed during preflight. Prepare it again.")
      }
      result["studioSignature"] = snapshot
      drawThingsClipEstimates[clip.id] = result
      preparedPrompt = clip.prompt
      preparedReport = String(decoding: try JSONSerialization.data(withJSONObject: result,
        options: [.prettyPrinted, .sortedKeys]), as: UTF8.self)
      if result["eligibility"] as? String == "allowed" {
        preparedDrawThingsClip = PreparedDrawThingsClip(projectID: projectID, clipID: clip.id,
          signature: snapshot, connection: connection)
        if let i = project.clips.firstIndex(where: { $0.id == clip.id }) {
          project.clips[i].validatedSignature = snapshot
        }
        notice = "Draw Things preflight passed. Review the prompt and CU, then generate."
      } else {
        notice = "Draw Things needs attention. Review the preflight details."
      }
    } catch {
      self.error = error.localizedDescription
      if project.id == projectID { validationErrors[clip.id] = error.localizedDescription }
    }
  }
  func renderDrawThingsClip() async {
    guard let clip = selectedClip, let prepared = preparedDrawThingsClip,
      canGenerateSelected else { error = "Prepare the current clip before generating."; return }
    do {
      var body = try payload()
      body["connection"] = try prepared.connection.object()
      let output = Self.supportDirectory.appendingPathComponent("Jobs/\(UUID().uuidString)/render")
      let result = try await bridge.invoke("dt-generate-clip", runtime: runtime, payload: body, output: output)
      guard let video = result["video"] as? String, FileManager.default.fileExists(atPath: video),
        let manifest = result["manifestPath"] as? String else {
        throw StudioError.invalid("Draw Things did not return a completed video with audio.")
      }
      guard project.id == prepared.projectID, project.clips.contains(where: { $0.id == clip.id }) else {
        throw StudioError.invalid("The destination project or clip changed. The completed video is saved at \(video)")
      }
      let stillCurrent = project.clips.first(where: { $0.id == clip.id })
        .map { signature(for: $0) == prepared.signature } ?? false
      change { p in
        guard let i = p.clips.firstIndex(where: { $0.id == clip.id }) else { return }
        p.clips[i].versions.append(RenderVersion(path: video, seed: clip.seed, prompt: clip.prompt,
          recipePath: manifest))
        if stillCurrent {
          p.clips[i].sourcePath = video
          p.clips[i].sourceIn = 0
          p.clips[i].renderedSignature = prepared.signature
        }
        var asset = MediaAsset(name: clip.name + " render", kind: .video, path: video,
          scope: .clip, owner: clip.id)
        asset.duration = clip.duration
        asset.width = clip.generationWidth; asset.height = clip.generationHeight
        p.assets.append(asset)
      }
      preparedDrawThingsClip = nil
      showPrompt = false
      refreshPreview()
      notice = stillCurrent ? "Draw Things video and audio saved to clip versions and Clip Assets."
        : "Render saved as a version. The clip changed during generation; prepare its new settings."
    } catch { self.error = error.localizedDescription }
  }
  func loadDrawThingsConnections() {
    let url = Self.supportDirectory.appendingPathComponent("drawthings-connections.json")
    if let data = try? Data(contentsOf: url), let values = try? JSONDecoder().decode([DrawThingsConnection].self, from: data) {
      drawThingsConnections = values
    }
  }
  func saveDrawThingsConnections() {
    do {
      try JSONEncoder().encode(drawThingsConnections).write(
        to: Self.supportDirectory.appendingPathComponent("drawthings-connections.json"), options: .atomic)
    } catch { self.error = error.localizedDescription }
  }
  func testDrawThings(_ connection: DrawThingsConnection) async {
    do {
      let result = try await bridge.invoke("dt-discover", runtime: runtime,
        payload: ["connection": try connection.object()])
      drawThingsCatalogs[connection.id] = result
      notice = "Connected to \(connection.name)."
    } catch { self.error = error.localizedDescription }
  }
  func drawThingsModels(_ profileID: String, operation: String) -> [(id: String, name: String)] {
    guard let catalog = drawThingsCatalogs[profileID], let rules = catalog["capabilities"] as? [String: Any] else { return [] }
    let names = (catalog["models"] as? [[String: Any]] ?? []).reduce(into: [String: String]()) { result, item in
      if let id = item["id"] as? String, let name = item["name"] as? String { result[id] = name }
    }
    return rules.keys.filter { id in
      ((rules[id] as? [String: Any])?["operations"] as? [String: Any])?[operation] != nil
    }.sorted().map { (id: $0, name: names[$0] ?? $0) }
  }
  func drawThingsLoRAs(profileID: String, modelID: String) -> [DiscoveredDrawThingsLoRA] {
    (drawThingsCatalogs[profileID]?["loras"] as? [[String: Any]] ?? []).compactMap { item in
      guard let id = item["id"] as? String, let name = item["name"] as? String,
        let family = item["family"] as? String,
        let compatible = item["compatibleModelIDs"] as? [String], compatible.contains(modelID)
      else { return nil }
      return DiscoveredDrawThingsLoRA(id: id, name: name, family: family, compatibleModelIDs: compatible)
    }.sorted { $0.name.localizedCaseInsensitiveCompare($1.name) == .orderedAscending }
  }
  func loadDrawThingsLoRAGroups() {
    let url = Self.supportDirectory.appendingPathComponent("drawthings-lora-groups.json")
    if let data = try? Data(contentsOf: url),
      let groups = try? JSONDecoder().decode([DrawThingsLoRAGroup].self, from: data) {
      drawThingsLoRAGroups = groups
    }
  }
  func saveDrawThingsLoRAGroups() {
    do {
      try JSONEncoder().encode(drawThingsLoRAGroups).write(
        to: Self.supportDirectory.appendingPathComponent("drawthings-lora-groups.json"), options: .atomic)
    } catch { self.error = error.localizedDescription }
  }
  func beginImageGeneration(scope: AssetScope) {
    guard scope != .clip || selectedClipID != nil else { return }
    let destination = ImageAssetDestination(scope: scope, projectID: project.id,
      owner: scope == .clip ? selectedClipID : nil)
    var draft = DrawThingsImageDraft(destination: destination)
    draft.profileID = drawThingsConnections.first?.id ?? ""
    draft.modelID = drawThingsModels(draft.profileID, operation: "image").first?.id ?? ""
    imageDraft = draft; imageEstimate = nil; imagePreviewPath = nil
  }
  func prepareImageGeneration() async {
    guard let draft = imageDraft,
      let connection = drawThingsConnections.first(where: { $0.id == draft.profileID }) else { return }
    do {
      let result = try await bridge.invoke("dt-estimate", runtime: runtime, payload: [
        "connection": try connection.object(), "drawThingsRequest": draft.request(id: UUID().uuidString)])
      guard imageDraft == draft else { return }
      imageEstimate = result
    } catch { self.error = error.localizedDescription }
  }
  func generateImageAsset() async {
    guard let draft = imageDraft,
      let connection = drawThingsConnections.first(where: { $0.id == draft.profileID }) else { return }
    do {
      let output = Self.supportDirectory.appendingPathComponent("Generated Images/\(UUID().uuidString)")
      try FileManager.default.createDirectory(at: output.deletingLastPathComponent(), withIntermediateDirectories: true)
      var payload: [String: Any] = ["connection": try connection.object(),
        "drawThingsRequest": draft.request(id: UUID().uuidString), "name": draft.name,
        "scope": draft.destination.scope.rawValue, "project": try project.object()]
      if let owner = draft.destination.owner { payload["owner"] = owner.uuidString }
      let result = try await bridge.invoke("dt-generate-image", runtime: runtime, payload: payload, output: output)
      guard let value = result["asset"] as? [String: Any], let path = value["path"] as? String else {
        throw StudioError.invalid("Draw Things did not return a saved image.")
      }
      var asset = try draft.destination.asset(name: draft.name, path: path, in: project)
      asset.width = value["width"] as? Int ?? draft.width
      asset.height = value["height"] as? Int ?? draft.height
      var provenance = ImageGeneration(provider: "drawThings", requestFingerprint: result["fingerprint"] as? String ?? "",
        modelID: draft.modelID, prompt: draft.prompt)
      provenance.profileID = draft.profileID; provenance.negativePrompt = draft.negativePrompt
      provenance.configuration = try JSONDecoder().decode([String: JSONValue].self,
        from: JSONSerialization.data(withJSONObject:
          (result["normalizedRequest"] as? [String: Any])?["configuration"] ?? draft.configuration))
      provenance.inputIDs = []; provenance.generatedAt = Date(); asset.generation = provenance
      if asset.scope == .global { globalAssets.append(asset); saveGlobals() }
      else { change { $0.assets.append(asset) } }
      selectedAssetID = asset.id
      if imageDraft == draft { imagePreviewPath = path }
      notice = "Generated \(draft.name) in \(asset.scope.rawValue) assets."
    } catch { self.error = error.localizedDescription }
  }
}
