import Foundation
import StudioCore

@MainActor final class ModelSetupState: ObservableObject {
  @Published var presets: [ModelSetupPreset] = []
  @Published var downloads: [ModelSetupDownload] = []
  @Published var loadingCatalog = false
  @Published var catalogError = ""
  @Published var selectedPreset: ModelSetupPreset?
  @Published var roots: [String] = []
  @Published var selection = ModelSetupSelection()
  @Published var memoryMode = ModelSetupMemoryMode.automatic
  @Published var warnings: [String] = []
  @Published var error = ""
  @Published var resultPath = ""
  @Published var selectedDownloadID = ""
  @Published var downloadDestination = ""
  @Published var downloadMessage = ""
  @Published var setupLog = ""
  private let catalogBridge = Bridge()

  static func rendererAvailable(_ runtime: RuntimeSettings) -> Bool {
    FileManager.default.isExecutableFile(atPath: runtime.pythonPath)
      && FileManager.default.fileExists(atPath: runtime.root + "/scripts/studio_bridge.py")
  }

  func loadCatalog(runtime: RuntimeSettings) async {
    guard Self.rendererAvailable(runtime) else {
      presets = []
      downloads = []
      return
    }
    guard !loadingCatalog else { return }
    loadingCatalog = true
    catalogError = ""
    defer { loadingCatalog = false }
    do {
      let catalog = try await catalogBridge.invoke("setup-catalog", runtime: runtime, payload: [:])
      presets = try decode([ModelSetupPreset].self, from: catalog["presets"] ?? [])
      let downloadCatalog = try await catalogBridge.invoke(
        "setup-downloads", runtime: runtime, payload: [:])
      downloads = try decode([ModelSetupDownload].self, from: downloadCatalog["downloads"] ?? [])
    } catch {
      catalogError = "Model setup requires the current renderer. Use Set Up Managed Renderer above or update the connected repository. " + error.localizedDescription
    }
  }

  func begin(_ preset: ModelSetupPreset) {
    selectedPreset = preset
    roots = []
    selection = ModelSetupSelection()
    memoryMode = .automatic
    warnings = []
    error = ""
    resultPath = ""
    downloadMessage = ""
    selectedDownloadID = ""
    setupLog = ""
  }

  func scan(store: StudioStore) async {
    guard let preset = selectedPreset, !roots.isEmpty else { return }
    error = ""
    resultPath = ""
    defer { captureLog(store.bridge) }
    do {
      let response = try await store.bridge.invoke(
        "setup-scan", runtime: store.runtime, payload: ["presetID": preset.id, "roots": roots])
      selection.applyScan(response["candidates"] as? [String: [String]] ?? [:])
      warnings = response["warnings"] as? [String] ?? []
    } catch { self.error = error.localizedDescription }
  }

  func createRecipe(store: StudioStore) async {
    guard let preset = selectedPreset, selection.missingComponents(for: preset).isEmpty else {
      return
    }
    error = ""
    do {
      let response = try await store.bridge.invoke(
        "setup-create", runtime: store.runtime,
        payload: [
          "presetID": preset.id, "components": selection.components,
          "memoryMode": memoryMode.rawValue,
          "memoryGB": Double(ProcessInfo.processInfo.physicalMemory) / 1_073_741_824,
        ])
      guard let recipePath = response["recipePath"] as? String, !recipePath.isEmpty else {
        throw StudioError.invalid("Setup did not return a recipe. Open the log for details.")
      }
      resultPath = recipePath
      warnings = response["warnings"] as? [String] ?? []
      captureLog(store.bridge)
      await store.reloadProfiles()
      store.notice =
        "Model recipe created. Select it for a compatible clip, then prepare the clip to validate its media and settings."
    } catch {
      captureLog(store.bridge)
      self.error = error.localizedDescription
    }
  }

  func useRecipeForSelectedClip(store: StudioStore) {
    guard let preset = selectedPreset, let clip = store.selectedClip,
      preset.supports(clip), !resultPath.isEmpty, !store.bridge.busy
    else { return }
    guard FileManager.default.fileExists(atPath: resultPath) else {
      error = "The created recipe is missing. Create the recipe again before selecting it."
      resultPath = ""
      return
    }
    let recipePath = resultPath
    store.editClip { $0.profileID = recipePath }
    error = ""
    store.notice =
      "Recipe selected for \(clip.name). Prepare the clip to validate its media and settings."
  }

  func download(store: StudioStore) async {
    guard !selectedDownloadID.isEmpty, !downloadDestination.isEmpty else { return }
    error = ""
    downloadMessage = ""
    defer { captureLog(store.bridge) }
    do {
      let response = try await store.bridge.invoke(
        "setup-download", runtime: store.runtime,
        payload: [
          "downloadID": selectedDownloadID, "destination": downloadDestination,
          "existingRoots": roots,
        ])
      guard let path = response["path"] as? String, !path.isEmpty else {
        throw StudioError.invalid("Download did not return a model path. Open the log for details.")
      }
      if !roots.contains(path) { roots.append(path) }
      downloadMessage =
        response["message"] as? String ?? "Model prepared. Scan the selected folders to use it."
    } catch { self.error = error.localizedDescription }
  }

  private func captureLog(_ bridge: Bridge) {
    setupLog = String((setupLog + "\n" + bridge.log).suffix(60000))
  }

  private func decode<T: Decodable>(_ type: T.Type, from object: Any) throws -> T {
    try JSONDecoder().decode(type, from: JSONSerialization.data(withJSONObject: object))
  }
}
