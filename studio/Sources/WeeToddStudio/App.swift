import AppKit
import StudioCore
import SwiftUI

@main struct WeeToddStudioApp: App {
  @StateObject private var store = StudioStore()
  @AppStorage("appearance") private var appearance = "system"
  var body: some Scene {
    WindowGroup("WeeTodd Studio") {
      StudioView().environmentObject(store).frame(minWidth: 1200, minHeight: 760)
        .preferredColorScheme(appearance == "light" ? .light : appearance == "dark" ? .dark : nil)
        .onOpenURL { store.load($0) }
        .onAppear {
          NSApp.setActivationPolicy(.regular)
          NSApp.activate(ignoringOtherApps: true)
        }
    }
    .defaultSize(width: 1600, height: 960)
    .windowStyle(.hiddenTitleBar)
    .commands {
      CommandGroup(replacing: .newItem) {
        Button("New Movie") { store.newProject() }.keyboardShortcut("n")
        Button("Open Project…") { store.openProject() }.keyboardShortcut("o")
      }
      CommandGroup(replacing: .saveItem) {
        Button("Save Project") { store.save() }.keyboardShortcut("s")
        Button("Save Project As…") { store.save(asNew: true) }.keyboardShortcut(
          "s", modifiers: [.command, .shift])
        Button("Collect Media…") { store.collectMedia() }
      }
      CommandGroup(replacing: .undoRedo) {
        Button("Undo") { store.undo() }.keyboardShortcut("z").disabled(!store.canUndo)
        Button("Redo") { store.redo() }.keyboardShortcut("z", modifiers: [.command, .shift])
          .disabled(!store.canRedo)
      }
      CommandMenu("Clip") {
        Button("Add Generated Clip") { store.addClip() }.keyboardShortcut(
          "n", modifiers: [.command, .shift])
        Button("Import Image Sequence…") { store.importSequence() }
        Button("Import Movie…") { store.chooseImports(addToTimeline: true) }.keyboardShortcut("i")
        Button("Edit Prompt") { store.showPrompt = true }.keyboardShortcut(
          .return, modifiers: .command)
        Button("Split at Playhead") { store.split() }.keyboardShortcut("b")
        Button("Insert Bridge to Next Clip…") { Task { await store.insertBridge() } }
        Button("Duplicate Clip") { store.duplicateClip() }.keyboardShortcut("d")
        Button("Delete Clip") { store.deleteClip() }
      }
      CommandMenu("Movie") {
        Button("Export Movie Headless Job…") { store.exportJob(clipOnly: false) }
        Button("Export Clip Headless Job…") { store.exportJob(clipOnly: true) }
        Button("Add Audio Track") { store.addAudioTrack() }
        Button("Add Title") { store.addTitle() }
        Button("Export Movie…") { store.exportMovie() }.keyboardShortcut("e")
        Button("Draw Things Connections…") { store.showDrawThings = true }
        Button("Runtime Settings…") { store.showRuntime = true }.keyboardShortcut(",")
      }
    }
  }
}

enum Theme {
  static let background = Color(nsColor: .windowBackgroundColor)
  static let panel = Color(nsColor: .controlBackgroundColor)
  static let raised = Color(nsColor: .quaternaryLabelColor).opacity(0.12)
  static let line = Color(nsColor: .separatorColor)
  static let mint = Color.accentColor
  static let violet = Color(red: 0.70, green: 0.64, blue: 0.94)
  static let text = Color.primary
  static func engine(_ e: Engine) -> Color {
    switch e {
    case .h3: return .orange
    case .ltx23: return .cyan
    case .ltx25: return violet
    case .drawThings: return .purple
    case .movie: return mint
    }
  }
}
struct PanelHeading: View {
  var title: String
  var icon: String
  var trailing: String = ""
  var body: some View {
    HStack(spacing: 8) {
      Image(systemName: icon).foregroundStyle(Theme.mint)
      Text(title).font(.system(size: 11, weight: .semibold))
      Spacer()
      Text(trailing).font(.system(size: 10, design: .monospaced)).foregroundStyle(.secondary)
    }
    .padding(.horizontal, 16).frame(height: 40).background(Theme.panel)
  }
}
struct SmallLabel: View {
  var text: String
  var body: some View {
    Text(text.uppercased()).font(.system(size: 9, weight: .semibold)).tracking(1.3).foregroundStyle(
      .secondary)
  }
}
struct StudioView: View {
  @EnvironmentObject var store: StudioStore
  @State private var leftWidth: CGFloat = 280
  var body: some View {
    ZStack {
      VStack(spacing: 0) {
        topbar
        Divider().overlay(Theme.line)
        HSplitView {
          InspectorColumn().frame(minWidth: 250, idealWidth: 280, maxWidth: 340)
          VStack(spacing: 0) {
            PreviewPane().frame(maxHeight: .infinity)
            TimelineView().frame(
              height: min(424, CGFloat(222 + store.project.audioTracks.count * 38)))
            statusbar
          }.frame(minWidth: 570, maxWidth: .infinity)
          AssetBrowser().frame(minWidth: 245, idealWidth: 290, maxWidth: 370)
        }
      }
      .disabled(store.showPrompt || store.showMotionPrompt || store.imageDraft != nil)
      if store.imageDraft != nil { ImageGenerationEditor().transition(.opacity).zIndex(10) }
      if store.showPrompt { PromptEditor().transition(.opacity).zIndex(10) }
      if store.showMotionPrompt {
        MotionPromptEditor().transition(.opacity).zIndex(10)
      }
    }
    .background(Theme.background).foregroundStyle(Theme.text).tint(.accentColor)
    .sheet(isPresented: $store.showProjectSettings) {
      VStack(alignment: .leading, spacing: 20) {
        HStack {
          Text("Movie Settings").font(.title2)
          Spacer()
          Button("Done") { store.showProjectSettings = false }
        }
        MovieSettingsForm(
          settings: Binding(
            get: { store.project.settings }, set: { value in store.change { $0.settings = value } })
        )
      }.padding(24).frame(width: 480)
    }
    .sheet(isPresented: $store.showDrawThings) { DrawThingsSettings().environmentObject(store) }
    .sheet(isPresented: $store.showDrawThingsConfigImport) { DrawThingsConfigImportView().environmentObject(store) }
    .sheet(isPresented: $store.showRuntime) { RuntimeView().environmentObject(store) }
    .sheet(isPresented: $store.showLog) {
      VStack(alignment: .leading) {
        HStack {
          Text("Job log").font(.title2)
          Spacer()
          Button("Done") { store.showLog = false }
        }
        ScrollView {
          Text(store.bridge.log.isEmpty ? "No job output yet." : store.bridge.log).font(
            .system(size: 11, design: .monospaced)
          ).textSelection(.enabled).frame(maxWidth: .infinity, alignment: .leading)
        }
      }.padding(24).frame(width: 900, height: 620)
    }
    .alert(
      "WeeTodd Studio",
      isPresented: Binding(
        get: { store.error != nil && !store.showLoRALibrary }, set: { if !$0 { store.error = nil } }
      )
    ) {
      Button("OK") { store.error = nil }
      Button("Show Log") {
        store.error = nil
        store.showLog = true
      }
    } message: {
      Text(store.error ?? "")
    }
  }
  var topbar: some View {
    HStack(spacing: 14) {
      HStack(spacing: 9) {
        Image(systemName: "waveform.path").font(.system(size: 20, weight: .bold)).foregroundStyle(
          Theme.mint)
        Text("WEETODD").tracking(2).font(.system(size: 12, weight: .heavy))
        Text("STUDIO").tracking(2).font(.system(size: 10, weight: .medium)).foregroundStyle(
          .secondary)
      }
      Rectangle().fill(Theme.line).frame(width: 1, height: 22).padding(.horizontal, 8)
      TextField(
        "Movie name",
        text: Binding(get: { store.project.name }, set: { v in store.change { $0.name = v } })
      ).textFieldStyle(.plain).font(.system(size: 13, weight: .medium)).frame(maxWidth: 230)
      Circle().fill(store.dirty ? Color.orange : Theme.mint).frame(width: 5, height: 5).help(
        store.dirty ? "Changes are autosaved; save to update your project file." : "Project saved")
      Spacer()
      Text(
        "\(store.project.settings.width) × \(store.project.settings.height)  ·  \(store.project.settings.fps,specifier:"%.0f") FPS"
      ).font(.system(size: 10, design: .monospaced)).foregroundStyle(.secondary)
      Button {
        store.save()
      } label: {
        Image(systemName: "square.and.arrow.down")
      }.help("Save project · ⌘S")
      Button {
        store.showRuntime = true
      } label: {
        Image(systemName: "slider.horizontal.3")
      }.help("Runtime settings")
      Button {
        store.exportMovie()
      } label: {
        Label("Export movie", systemImage: "arrow.up.right")
      }.buttonStyle(.borderedProminent).foregroundStyle(.white).disabled(
        store.bridge.busy || store.project.clips.isEmpty)
    }.buttonStyle(.borderless).padding(.horizontal, 20).frame(height: 62)
  }
  var statusbar: some View {
    StatusBar().environmentObject(store).frame(height: 43).background(Theme.panel)
  }
}
struct StatusBar: View {
  @EnvironmentObject var store: StudioStore
  var body: some View { LiveStatus(bridge: store.bridge).environmentObject(store) }
}
struct LiveStatus: View {
  @EnvironmentObject var store: StudioStore
  @ObservedObject var bridge: Bridge
  var body: some View {
    HStack(spacing: 9) {
      if bridge.busy {
        ProgressView().controlSize(.small)
        Text(bridge.message).lineLimit(1)
        if bridge.fraction > 0 {
          ProgressView(value: bridge.fraction).frame(width: 70)
            .help("Progress within the current stage")
        }
        SwiftUI.TimelineView(.periodic(from: .now, by: 1)) { context in
          if let started = bridge.startedAt {
            Text(RenderStats.duration(context.date.timeIntervalSince(started)))
              .monospacedDigit().help("Elapsed time for this operation")
          }
          if let updated = bridge.lastOutputAt, context.date.timeIntervalSince(updated) >= 15 {
            Text("Last output \(RenderStats.duration(context.date.timeIntervalSince(updated))) ago")
              .foregroundStyle(.secondary).help("Time since renderer output; a long step may still be running")
          }
        }
        Button("Cancel") { bridge.cancel() }
      } else {
        Circle().fill(Theme.mint).frame(width: 5, height: 5)
        Text(store.notice).lineLimit(1)
      }
      Spacer(minLength: 4)
      Button {
        store.showActions.toggle()
      } label: {
        Label("Actions \(store.actionItems.count)", systemImage: "checklist")
      }.popover(isPresented: $store.showActions) { ActionList().environmentObject(store) }
      Button {
        store.showLog = true
      } label: {
        Image(systemName: "text.alignleft")
      }.help("Job log")
      Button {
        store.showPrompt = true
      } label: {
        Label("Prompt", systemImage: "text.bubble")
      }.disabled(store.selectedClip == nil || store.selectedClip?.engine == .movie)
    }.font(.system(size: 10)).foregroundStyle(.secondary).buttonStyle(.borderless).padding(
      .horizontal, 16)
  }
}
