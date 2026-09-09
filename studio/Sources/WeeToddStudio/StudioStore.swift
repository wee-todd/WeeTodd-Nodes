import AVFoundation
import AppKit
import Combine
import Foundation
import StudioCore
import UniformTypeIdentifiers

struct RuntimeSettings: Codable {
  var root: String
  var pythonPath: String
  var profilesDirectory: String
  var ffmpegPath = ""
  var ffprobePath = ""
  var rifePath = ""
  var rifeWeights = ""
  var metalPath = ""
  var drawThingsHelperPath: String?
  static func defaults() -> Self {
    let root =
      ProcessInfo.processInfo.environment["WEETODD_ROOT"] ?? Bundle.main.object(
        forInfoDictionaryKey: "WeeToddRuntimeRoot") as? String ?? ""
    let support = StudioStore.supportDirectory
    var value = Self(
      root: root, pythonPath: root.isEmpty ? "" : root + "/.venv/bin/python",
      profilesDirectory: support.appendingPathComponent("Profiles").path)
    value.metalPath =
      Bundle.main.bundleURL.appendingPathComponent("Contents/MacOS/StudioMetal").path
    value.drawThingsHelperPath = Bundle.main.bundleURL.appendingPathComponent("Contents/MacOS/WeeToddDrawThings").path
    return value
  }
}
struct ModelProfile: Identifiable {
  var id: String
  var name: String
  var engine: String
  var task: String
}

@MainActor final class Bridge: ObservableObject {
  @Published var busy = false
  @Published var log = ""
  @Published var message = "Ready"
  @Published var fraction: Double = 0
  private var process: Process?
  func cancel() {
    message = "Cancelling and releasing render resources…"
    process?.interrupt()
  }
  func invoke(
    _ command: String, runtime: RuntimeSettings, payload: [String: Any], output: URL? = nil
  ) async throws -> [String: Any] {
    guard !busy else {
      throw StudioError.invalid("Another job is active. Wait or cancel it first.")
    }
    guard FileManager.default.isExecutableFile(atPath: runtime.pythonPath),
      FileManager.default.fileExists(atPath: runtime.root + "/scripts/studio_bridge.py")
    else {
      throw StudioError.invalid(
        "Select the WeeTodd repository and its Python environment in Runtime Settings.")
    }
    var env = ProcessInfo.processInfo.environment
    if command == "setup-download" {
      env = try ModelDownloadToken.environment(env, savedToken: ModelDownloadToken.read())
    }
    if command.hasPrefix("dt-"), let connection = payload["connection"] as? [String: Any],
      let reference = connection["credentialRef"] as? String,
      let secret = try DrawThingsCredential.read(reference) {
      env["WEETODD_DT_CREDENTIAL"] = secret
    }
    env["PYTHONUNBUFFERED"] = "1"
    let input = StudioStore.supportDirectory.appendingPathComponent(
      "Requests/\(UUID().uuidString).json")
    try FileManager.default.createDirectory(
      at: input.deletingLastPathComponent(), withIntermediateDirectories: true)
    var body = payload
    body["runtime"] = try runtime.object()
    try JSONSerialization.data(withJSONObject: body, options: [.prettyPrinted, .sortedKeys]).write(
      to: input, options: .atomic)
    busy = true
    fraction = 0
    message = command.capitalized + "…"
    log = ""
    let task = Process()
    task.executableURL = URL(fileURLWithPath: runtime.pythonPath)
    task.arguments = [runtime.root + "/scripts/studio_bridge.py", command, "--request", input.path]
    if let output { task.arguments! += ["--output", output.path] }
    task.currentDirectoryURL = URL(fileURLWithPath: runtime.root)
    task.environment = env
    let pipe = Pipe()
    task.standardOutput = pipe
    task.standardError = pipe
    process = task
    do { try task.run() } catch {
      busy = false
      process = nil
      throw error
    }
    defer {
      busy = false
      process = nil
      try? FileManager.default.removeItem(at: input)
    }
    let response: [String: Any] = try await withCheckedThrowingContinuation { continuation in
      DispatchQueue.global(qos: .userInitiated).async {
        var bytes = Data()
        while true {
          let part = pipe.fileHandleForReading.availableData
          if part.isEmpty { break }
          bytes.append(part)
          if bytes.count > 2_000_000 { bytes = Data(bytes.suffix(1_000_000)) }
          let text = String(decoding: part, as: UTF8.self)
          DispatchQueue.main.async {
            self.log = String((self.log + text).suffix(30000))
            for line in text.split(separator: "\n") {
              if let data = String(line).data(using: .utf8),
                let event = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                event["event"] as? String == "progress"
              {
                self.message = event["message"] as? String ?? self.message
                self.fraction = event["fraction"] as? Double ?? self.fraction
              }
            }
          }
        }
        task.waitUntilExit()
        let lines = String(decoding: bytes, as: UTF8.self).split(separator: "\n")
        let last = lines.reversed().compactMap { line -> [String: Any]? in
          guard let data = String(line).data(using: .utf8),
            let d = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            d["status"] != nil
          else { return nil }
          return d
        }.first
        if task.terminationStatus == 0, let result = last?["result"] as? [String: Any] {
          continuation.resume(returning: result)
        } else {
          continuation.resume(
            throwing: StudioError.invalid(
              last?["error"] as? String
                ?? (last?["status"] as? String == "cancelled"
                  ? "Job cancelled." : "The job failed. Open the log for details.")))
        }
      }
    }
    message = "Ready"
    fraction = 1
    return response
  }
}
extension Encodable {
  func object() throws -> [String: Any] {
    try JSONSerialization.jsonObject(with: JSONEncoder().encode(self)) as! [String: Any]
  }
}

@MainActor final class StudioStore: ObservableObject {
  nonisolated static var supportDirectory: URL {
    if let override = ProcessInfo.processInfo.environment["WEETODD_STUDIO_DATA"] {
      return URL(fileURLWithPath: override)
    }
    return FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
      .appendingPathComponent("WeeTodd Studio")
  }
  @Published var project = StudioProject()
  @Published var globalAssets: [MediaAsset] = []
  @Published var loraGroups: [LoRAGroup] = []
  @Published var showLoRALibrary = false
  @Published var selectedClipID: UUID?
  @Published var selectedAssetID: UUID?
  @Published var selectedTitleID: UUID?
  @Published var selectedAudioID: UUID?
  @Published var playhead: Double = 0
  @Published var zoom: Double = 42
  @Published var runtime = RuntimeSettings.defaults()
  @Published var profiles: [ModelProfile] = []
  @Published var projectURL: URL?
  @Published var showPrompt = false
  @Published var showMotionPrompt = false
  @Published var showRuntime = false
  @Published var showDrawThings = false
  @Published var drawThingsConnections: [DrawThingsConnection] = []
  @Published var drawThingsCatalogs: [String: [String: Any]] = [:]
  @Published var drawThingsLoRAGroups: [DrawThingsLoRAGroup] = []
  @Published var imageDraft: DrawThingsImageDraft?
  @Published var imageEstimate: [String: Any]?
  @Published var imagePreviewPath: String?
  @Published var drawThingsClipEstimates: [UUID: [String: Any]] = [:]
  var preparedDrawThingsClip: PreparedDrawThingsClip?
  @Published var showProjectSettings = false
  @Published var showLog = false
  @Published var showActions = false
  @Published var selectedTrackID: UUID?
  @Published var error: String?
  @Published var validationErrors: [UUID: String] = [:]
  @Published var notice = "Create a clip or drop a movie onto the timeline."
  @Published var preparedPrompt = ""
  @Published var preparedRecipe: String?
  @Published var preparedReport = ""
  @Published var motionPromptDraft = ""
  @Published var motionRecipePrompt = ""
  @Published var motionPromptClipName = ""
  @Published var motionPromptUsingOverride = false
  @Published var motionPromptLoading = false
  @Published var motionPromptEditorError: String?
  @Published var dirty = false
  @Published var player = AVPlayer()
  @Published var isPlaying = false
  @Published var queue: [UUID] = []
  @Published var previewMode = "Clip"
  let bridge = Bridge()
  private var preparedFingerprint: String?
  var motionPromptSession: MotionPromptEditorSession?
  private var undoStates: [StudioProject] = []
  private var redoStates: [StudioProject] = []
  private var lastUndoGroup: UUID?
  private var autosaveTask: Task<Void, Never>?
  private var observer: Any?
  private var bridgeObservation: AnyCancellable?
  var selectedClip: Clip? { project.clips.first { $0.id == selectedClipID } }
  var allAssets: [MediaAsset] { globalAssets + project.assets }
  var selectedAsset: MediaAsset? { allAssets.first { $0.id == selectedAssetID } }
  var canUndo: Bool { !undoStates.isEmpty }
  var canRedo: Bool { !redoStates.isEmpty }

  init() {
    bridgeObservation = bridge.objectWillChange.sink { [weak self] _ in
      self?.objectWillChange.send()
    }
    try? FileManager.default.createDirectory(
      at: Self.supportDirectory.appendingPathComponent("Profiles"),
      withIntermediateDirectories: true)
    if let data = try? Data(
      contentsOf: Self.supportDirectory.appendingPathComponent("runtime.json")),
      let value = try? JSONDecoder().decode(RuntimeSettings.self, from: data)
    {
      runtime = value
    }
    if let data = try? Data(
      contentsOf: Self.supportDirectory.appendingPathComponent("global-assets.json")),
      let value = try? JSONDecoder().decode([MediaAsset].self, from: data)
    {
      globalAssets = value
    }
    loadLoRAGroups()
    loadDrawThingsConnections()
    loadDrawThingsLoRAGroups()
    let args = CommandLine.arguments
    if let i = args.firstIndex(of: "--project"), args.count > i + 1 {
      load(URL(fileURLWithPath: args[i + 1]))
    } else if let p = try? ProjectStorage.read(
      Self.supportDirectory.appendingPathComponent("Autosave.weetodd"))
    {
      project = p
      selectedClipID = p.clips.first?.id
      notice = "Recovered your autosaved project."
    }
    observer = player.addPeriodicTimeObserver(
      forInterval: CMTime(seconds: 0.05, preferredTimescale: 600), queue: .main
    ) { [weak self] time in
      Task { @MainActor in
        guard let self, self.isPlaying else { return }
        let local =
          time.seconds - (self.previewMode == "Movie" ? 0 : self.selectedClip?.playbackIn ?? 0)
        let duration =
          self.previewMode == "Movie" ? self.project.duration : self.selectedClip?.duration ?? 0
        if local >= duration {
          self.player.pause()
          self.isPlaying = false
        } else if local.isFinite {
          self.playhead = max(0, local)
        }
      }
    }
    Task {
      await reloadProfiles()
      refreshPreview()
    }
  }
  func change(undoGroup: UUID? = nil, _ body: (inout StudioProject) -> Void) {
    var updated = project
    body(&updated)
    guard updated != project else { return }
    if undoGroup == nil || undoGroup != lastUndoGroup { undoStates.append(project) }
    lastUndoGroup = undoGroup
    if undoStates.count > 80 { undoStates.removeFirst() }
    redoStates.removeAll()
    project = updated
    changed()
  }
  func editClip(undoGroup: UUID? = nil, _ body: (inout Clip) -> Void) {
    guard let i = project.clips.firstIndex(where: { $0.id == selectedClipID }) else { return }
    change(undoGroup: undoGroup) { body(&$0.clips[i]) }
  }
  func changed() {
    preparedDrawThingsClip = nil
    validationErrors.removeAll()
    if previewMode == "Movie" {
      previewMode = "Clip"
      refreshPreview()
    }
    dirty = true
    preparedRecipe = nil
    preparedFingerprint = nil
    preparedPrompt = ""
    preparedReport = ""
    autosaveTask?.cancel()
    autosaveTask = Task { [weak self] in
      try? await Task.sleep(nanoseconds: 600_000_000)
      guard !Task.isCancelled, let self else { return }
      do {
        try ProjectStorage.write(
          self.project, to: Self.supportDirectory.appendingPathComponent("Autosave.weetodd"))
      } catch { self.notice = "Autosave needs attention: \(error.localizedDescription)" }
    }
  }
  func undo() {
    lastUndoGroup = nil
    guard let value = undoStates.popLast() else { return }
    redoStates.append(project)
    project = value
    changed()
    refreshPreview()
  }
  func redo() {
    lastUndoGroup = nil
    guard let value = redoStates.popLast() else { return }
    undoStates.append(project)
    project = value
    changed()
    refreshPreview()
  }
  func select(_ id: UUID) {
    preparedDrawThingsClip = nil
    lastUndoGroup = nil
    selectedClipID = id
    selectedTitleID = nil
    selectedAudioID = nil
    selectedTrackID = nil
    playhead = 0
    preparedRecipe = nil
    refreshPreview()
  }
  func addClip(_ engine: Engine = .ltx25) {
    var c = Clip(name: "Shot \(project.clips.count + 1)", engine: engine)
    if engine == .drawThings {
      c.drawThings = DrawThingsSelection(profileID: drawThingsConnections.first?.id ?? "",
        modelID: "", modelFamily: "", configuration: ["steps": .integer(8)])
    }
    change { $0.clips.append(c) }
    select(c.id)
  }
  func deleteClip() {
    guard let id = selectedClipID else { return }
    change {
      $0.clips.removeAll { $0.id == id }
      $0.assets.removeAll { $0.scope == .clip && $0.owner == id }
    }
    selectedClipID = project.clips.first?.id
    refreshPreview()
  }
  func duplicateClip() {
    guard var c = selectedClip else { return }
    let oldID = c.id
    c.id = UUID()
    c.name += " copy"
    // Duplicate clip-store links, retaining the same underlying media files.
    var links: [MediaAsset] = []
    for old in project.assets where old.owner == oldID && old.scope == .clip {
      var linked = old
      linked.id = UUID()
      linked.owner = c.id
      for i in c.attachments.indices where c.attachments[i].assetID == old.id {
        c.attachments[i].assetID = linked.id
      }
      links.append(linked)
    }
    change { p in
      let i = p.clips.firstIndex { $0.id == oldID } ?? p.clips.count - 1
      p.clips.insert(c, at: i + 1)
      p.assets.append(contentsOf: links)
    }
    select(c.id)
  }
  func split() {
    guard let id = selectedClipID else { return }
    do {
      var p = project
      let next = try p.split(id, at: playhead)
      change { $0 = p }
      select(next)
    } catch { self.error = error.localizedDescription }
  }
  func seek(_ seconds: Double) {
    playhead = min(
      max(0, seconds), previewMode == "Movie" ? project.duration : selectedClip?.duration ?? 0)
    player.seek(
      to: CMTime(
        seconds: (previewMode == "Movie" ? 0 : selectedClip?.playbackIn ?? 0) + playhead,
        preferredTimescale: 600), toleranceBefore: .zero, toleranceAfter: .zero)
  }
  func togglePlayback() {
    guard player.currentItem != nil else { return }
    let duration = previewMode == "Movie" ? project.duration : selectedClip?.duration ?? 0
    if isPlaying {
      player.pause()
    } else {
      if playhead >= duration - 0.05 { seek(0) }
      player.play()
    }
    isPlaying.toggle()
  }
  func refreshPreview() {
    player.pause()
    isPlaying = false
    previewMode = "Clip"
    guard let c = selectedClip, !c.sourcePath.isEmpty else {
      player.replaceCurrentItem(with: nil)
      return
    }
    player.replaceCurrentItem(with: AVPlayerItem(url: URL(fileURLWithPath: c.playbackPath)))
    seek(playhead)
  }
  func save(asNew: Bool = false) {
    var target = asNew ? nil : projectURL
    if target == nil {
      let panel = NSSavePanel()
      panel.title = "Save WeeTodd Project"
      panel.nameFieldStringValue = project.name + ".weetodd"
      panel.canCreateDirectories = true
      guard panel.runModal() == .OK else { return }
      target = panel.url
    }
    guard let target else { return }
    do {
      try ProjectStorage.write(project, to: target)
      projectURL = target
      dirty = false
      notice = "Saved \(target.lastPathComponent)"
    } catch { self.error = error.localizedDescription }
  }
  func openProject() {
    let panel = NSOpenPanel()
    panel.allowedContentTypes = [.json, UTType(filenameExtension: "weetodd") ?? .data]
    guard panel.runModal() == .OK, let url = panel.url else { return }
    load(url)
  }
  func load(_ url: URL) {
    do {
      let p = try ProjectStorage.read(url)
      cancelMotionPromptEditor()
      change { $0 = p }
      projectURL = url
      selectedClipID = p.clips.first?.id
      dirty = false
      refreshPreview()
    } catch { self.error = error.localizedDescription }
  }
  func newProject() {
    cancelMotionPromptEditor()
    change { $0 = StudioProject() }
    projectURL = nil
    selectedClipID = nil
    refreshPreview()
  }
  func chooseImports(scope: AssetScope = .project, addToTimeline: Bool = false) {
    let panel = NSOpenPanel()
    panel.allowsMultipleSelection = true
    panel.canChooseDirectories = false
    guard panel.runModal() == .OK else { return }
    Task { await importURLs(panel.urls, scope: scope, addToTimeline: addToTimeline) }
  }
  func importURLs(
    _ urls: [URL], scope: AssetScope, addToTimeline: Bool = false, loraModel: LoRAModel? = nil
  ) async {
    if scope == .clip && selectedClipID == nil && !addToTimeline {
      error = "Select a clip before importing into its asset store."
      return
    }
    for url in urls {
      do {
        let info = try await bridge.invoke("inspect", runtime: runtime, payload: ["path": url.path])
        let kind = AssetKind(rawValue: info["kind"] as? String ?? "video") ?? .video
        var asset = MediaAsset(
          name: url.deletingPathExtension().lastPathComponent, kind: kind, path: url.path,
          scope: scope, owner: scope == .clip ? selectedClipID : nil)
        asset.duration = info["duration"] as? Double ?? 0
        asset.width = info["width"] as? Int ?? 0
        asset.height = info["height"] as? Int ?? 0
        asset.fps = info["fps"] as? Double ?? 0
        asset.text = info["text"] as? String ?? ""
        if kind == .lora {
          asset.loraModel =
            (info["loraModel"] as? String).flatMap(LoRAModel.init(rawValue:)) ?? loraModel
        }
        if addToTimeline && (kind == .video || kind == .image) {
          var c = Clip(name: asset.name, engine: .movie)
          c.sourcePath = asset.path
          c.duration = kind == .image ? 5 : max(0.1, asset.duration)
          asset.scope = .clip
          asset.owner = c.id
          change {
            $0.clips.append(c)
            $0.assets.append(asset)
          }
          select(c.id)
        } else if scope == .global {
          globalAssets.append(asset)
          saveGlobals()
        } else {
          change { $0.assets.append(asset) }
        }
        selectedAssetID = asset.id
        notice = "Linked \(asset.name) to \(asset.scope.rawValue) assets."
      } catch {
        self.error = error.localizedDescription
        break
      }
    }
  }
  func saveGlobals() {
    do {
      try JSONEncoder().encode(globalAssets).write(
        to: Self.supportDirectory.appendingPathComponent("global-assets.json"), options: .atomic)
    } catch { self.error = error.localizedDescription }
  }
  func useAsset(_ asset: MediaAsset, role: MediaRole, time: Double = 0) {
    if asset.kind == .text {
      editClip { $0.prompt = asset.text }
      return
    }
    guard selectedClip != nil else {
      error = "Select or create a clip first."
      return
    }
    if role == .lora {
      applyLoRAMembers([LoRAMember(asset: asset)])
      return
    }
    editClip { c in
      if [.first, .last, .audioDriver].contains(role) {
        c.attachments.removeAll { $0.role == role }
      }
      c.attachments.append(Attachment(assetID: asset.id, role: role, time: time))
    }
  }
  func addAssetToTimeline(_ asset: MediaAsset) {
    if asset.kind == .audio {
      var a = AudioRegion(assetID: asset.id, path: asset.path)
      a.duration = max(0.1, asset.duration)
      a.trackID = selectedTrackID ?? project.audioTracks.first?.id
      change { $0.audio.append(a) }
      selectedAudioID = a.id
      selectedTitleID = nil
    } else if [.video, .image, .sequence].contains(asset.kind) {
      var c = Clip(name: asset.name, engine: .movie)
      c.sourcePath = asset.path
      c.duration = asset.kind == .image ? 5 : max(0.1, asset.duration)
      var linked = asset
      linked.id = UUID()
      linked.scope = .clip
      linked.owner = c.id
      change {
        $0.clips.append(c)
        $0.assets.append(linked)
      }
      select(c.id)
    }
  }
  func addTitle() {
    var t = TitleOverlay()
    t.start =
      selectedClipID.flatMap { id in project.clips.firstIndex { $0.id == id } }.map {
        project.start(of: $0)
      } ?? 0
    change { $0.titles.append(t) }
    selectedTitleID = t.id
    selectedAudioID = nil
  }
  func relink(_ asset: MediaAsset) {
    let panel = NSOpenPanel()
    guard panel.runModal() == .OK, let url = panel.url else { return }
    let old = asset.path
    if let i = globalAssets.firstIndex(where: { $0.id == asset.id }) {
      globalAssets[i].path = url.path
      saveGlobals()
    }
    change { p in
      for i in p.assets.indices where p.assets[i].id == asset.id { p.assets[i].path = url.path }
      for i in p.clips.indices where p.clips[i].sourcePath == old {
        p.clips[i].sourcePath = url.path
      }
      for i in p.audio.indices where p.audio[i].path == old { p.audio[i].path = url.path }
    }
    refreshPreview()
  }
  func collectMedia() {
    let panel = NSOpenPanel()
    panel.title = "Choose a folder for the portable project"
    panel.canChooseDirectories = true
    panel.canChooseFiles = false
    panel.canCreateDirectories = true
    guard panel.runModal() == .OK, let folder = panel.url else { return }
    do {
      let bundle = folder.appendingPathComponent(
        project.name + "-" + String(UUID().uuidString.prefix(6)))
      let media = bundle.appendingPathComponent("Media")
      try FileManager.default.createDirectory(at: media, withIntermediateDirectories: true)
      var p = project
      var copied: [String: String] = [:]
      let used = Set(p.clips.flatMap { $0.attachments.map(\.assetID) } + p.audio.map(\.assetID))
      for var asset in globalAssets where used.contains(asset.id) {
        asset.scope = .project
        p.assets.append(asset)
      }
      func collect(_ value: String) throws -> String {
        if value.isEmpty { return value }
        if let known = copied[value] { return known }
        let source = URL(fileURLWithPath: value)
        let target = media.appendingPathComponent(
          UUID().uuidString + "-" + source.lastPathComponent)
        try FileManager.default.copyItem(at: source, to: target)
        let relative = "Media/" + target.lastPathComponent
        copied[value] = relative
        return relative
      }
      for i in p.assets.indices where p.assets[i].kind != .lora {
        p.assets[i].path = try collect(p.assets[i].path)
        if p.assets[i].kind == .sequence { p.assets[i].text = try collect(p.assets[i].text) }
      }
      for i in p.clips.indices {
        p.clips[i].sourcePath = try collect(p.clips[i].sourcePath)
        if p.clips[i].motionResult != nil {
          p.clips[i].motionResult!.path = try collect(p.clips[i].motionResult!.path)
          p.clips[i].motionResult!.sourcePath = try collect(p.clips[i].motionResult!.sourcePath)
          p.clips[i].motionResult!.report = try collect(p.clips[i].motionResult!.report)
          if let recipe = p.clips[i].motionResult!.recipePath {
            p.clips[i].motionResult!.recipePath = try collect(recipe)
          }
        }
        p.clips[i].extensionSource = try collect(p.clips[i].extensionSource)
        p.clips[i].depthDirectory = try collect(p.clips[i].depthDirectory)
        p.clips[i].motionDirectory = try collect(p.clips[i].motionDirectory)
        for j in p.clips[i].versions.indices {
          p.clips[i].versions[j].path = try collect(p.clips[i].versions[j].path)
        }
      }
      for i in p.audio.indices { p.audio[i].path = try collect(p.audio[i].path) }
      let target = bundle.appendingPathComponent(project.name + ".weetodd")
      try ProjectStorage.write(p, to: target)
      notice = "Collected \(copied.count) media files. Model weights remain shared."
      NSWorkspace.shared.activateFileViewerSelecting([target])
    } catch { self.error = error.localizedDescription }
  }
  func saveRuntime(reloadProfiles: Bool = true) {
    do {
      try JSONEncoder().encode(runtime).write(
        to: Self.supportDirectory.appendingPathComponent("runtime.json"), options: .atomic)
      if reloadProfiles { Task { await self.reloadProfiles() } }
    } catch { self.error = error.localizedDescription }
  }
  func reloadProfiles() async {
    guard !runtime.root.isEmpty, !bridge.busy else { return }
    do {
      let r = try await bridge.invoke("catalog", runtime: runtime, payload: [:])
      profiles = (r["profiles"] as? [[String: Any]] ?? []).compactMap { d in
        guard let id = d["id"] as? String, let name = d["name"] as? String,
          let engine = d["engine"] as? String, let task = d["task"] as? String
        else { return nil }
        return ModelProfile(id: id, name: name, engine: engine, task: task)
      }
    } catch { notice = error.localizedDescription }
  }
  func importRecipes() {
    let panel = NSOpenPanel()
    panel.allowedContentTypes = [.json]
    panel.allowsMultipleSelection = true
    guard panel.runModal() == .OK else { return }
    do {
      let folder = URL(fileURLWithPath: runtime.profilesDirectory)
      try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
      for url in panel.urls {
        let d = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any]
        guard d?["format"] as? String == "weetodd-headless-v2" else {
          throw StudioError.invalid("Import a WeeTodd headless v2 recipe, not a ComfyUI graph.")
        }
        var target = folder.appendingPathComponent(url.lastPathComponent)
        if FileManager.default.fileExists(atPath: target.path) {
          target = folder.appendingPathComponent(UUID().uuidString + "-" + url.lastPathComponent)
        }
        try FileManager.default.copyItem(at: url, to: target)
      }
      Task { await reloadProfiles() }
    } catch { self.error = error.localizedDescription }
  }
  func payload() throws -> [String: Any] {
    [
      "project": try project.object(),
      "globalAssets": try JSONSerialization.jsonObject(with: JSONEncoder().encode(globalAssets)),
      "clipID": selectedClipID?.uuidString ?? "",
    ]
  }
  func prepareSelected() async {
    guard selectedClip != nil else { return }
    if selectedClip?.engine == .drawThings { await prepareDrawThingsClip(); return }
    do {
      let snapshot = signature(for: selectedClip!)
      let destination = Self.supportDirectory.appendingPathComponent(
        "Jobs/\(UUID().uuidString)/prepared")
      let r = try await bridge.invoke(
        "prepare", runtime: runtime, payload: try payload(), output: destination)
      guard selectedClip.map({ signature(for: $0) }) == snapshot else {
        throw StudioError.invalid("Clip changed during preflight. Prepare it again.")
      }
      preparedRecipe = r["recipePath"] as? String
      preparedPrompt = r["prompt"] as? String ?? ""
      preparedReport =
        String(
          data: try JSONSerialization.data(
            withJSONObject: r["report"] ?? [:], options: [.prettyPrinted, .sortedKeys]),
          encoding: .utf8) ?? ""
      if let i = project.clips.firstIndex(where: { $0.id == selectedClipID }) {
        project.clips[i].validatedSignature = snapshot
      }
      preparedFingerprint = snapshot
      notice = "Preflight passed. Review the exact prompt, then render."
    } catch {
      self.error = error.localizedDescription
      if let id = selectedClipID { validationErrors[id] = error.localizedDescription }
    }
  }
  func renderPrepared() async {
    if selectedClip?.engine == .drawThings { await renderDrawThingsClip(); return }
    guard let path = preparedRecipe, let c = selectedClip else { return }
    do {
      guard signature(for: c) == preparedFingerprint else {
        throw StudioError.invalid("Clip changed. Prepare it again before rendering.")
      }
      let destination = URL(fileURLWithPath: path).deletingLastPathComponent()
        .deletingLastPathComponent().appendingPathComponent("render")
      let prompt = preparedPrompt
      let r = try await bridge.invoke(
        "render", runtime: runtime, payload: ["recipePath": path], output: destination)
      guard let video = r["video"] as? String else {
        throw StudioError.invalid("Renderer did not return a movie.")
      }
      let info = try await bridge.invoke("inspect", runtime: runtime, payload: ["path": video])
      var renderedDuration = info["duration"] as? Double ?? c.duration
      var renderedStart = 0.0
      if !c.extensionSource.isEmpty {
        let source = try await bridge.invoke(
          "inspect", runtime: runtime, payload: ["path": c.extensionSource])
        let sourceDuration = source["duration"] as? Double ?? 0
        renderedDuration -= sourceDuration
        if c.extensionDirection == "after" { renderedStart = sourceDuration }
      }
      guard renderedDuration > 0 else {
        throw StudioError.invalid("The extension returned no new frames.")
      }
      var finishedClip = c
      finishedClip.duration = min(c.duration, renderedDuration)
      let finishedSignature = signature(for: finishedClip)
      change { p in
        guard let i = p.clips.firstIndex(where: { $0.id == c.id }) else { return }
        p.clips[i].versions.append(
          RenderVersion(path: video, seed: c.seed, prompt: prompt, recipePath: path))
        p.clips[i].sourcePath = video
        p.clips[i].sourceIn = renderedStart
        p.clips[i].duration = min(c.duration, renderedDuration)
        p.clips[i].renderedSignature = finishedSignature
        var asset = MediaAsset(
          name: c.name + " render", kind: .video, path: video, scope: .clip, owner: c.id)
        asset.duration = p.clips[i].duration
        p.assets.append(asset)
      }
      showPrompt = false
      refreshPreview()
      notice = "Render complete. Added to clip versions and Clip Assets."
    } catch { self.error = error.localizedDescription }
  }
  func extend(_ direction: String) {
    guard let old = selectedClip, !old.sourcePath.isEmpty else {
      error = "Render or import a movie before extending it."
      return
    }
    var c = Clip(
      name: old.name + " · extension", engine: old.engine == .movie ? .ltx25 : old.engine)
    c.prompt = old.prompt
    c.extensionDirection = direction
    c.extensionSource = old.sourcePath
    c.extensionClipID = old.id
    c.generationWidth = old.generationWidth
    c.generationHeight = old.generationHeight
    change { p in
      let i = p.clips.firstIndex(where: { $0.id == old.id })!
      p.clips.insert(c, at: direction == "before" ? i : i + 1)
    }
    select(c.id)
    showPrompt = true
  }
  func exportMovie() {
    let panel = NSSavePanel()
    let f = project.settings.format
    panel.nameFieldStringValue =
      project.name + (f == .pngSequence ? "-frames" : f == .mp4 ? ".mp4" : ".mov")
    guard panel.runModal() == .OK, let url = panel.url else { return }
    Task { await export(to: url) }
  }
  func export(to url: URL) async {
    do {
      _ = try await bridge.invoke("export", runtime: runtime, payload: try payload(), output: url)
      notice = "Exported \(url.lastPathComponent)"
      NSWorkspace.shared.activateFileViewerSelecting([url])
    } catch { self.error = error.localizedDescription }
  }
}
