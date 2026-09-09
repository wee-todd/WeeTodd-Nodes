import AppKit
import StudioCore
import SwiftUI

struct ModelSetupCatalogView: View {
  @EnvironmentObject var store: StudioStore
  @ObservedObject var state: ModelSetupState
  @ObservedObject var bridge: Bridge
  var rendererInstalling: Bool

  var body: some View {
    VStack(alignment: .leading, spacing: 12) {
      HStack {
        Text("Model setup").font(.headline)
        Spacer()
        if state.loadingCatalog { ProgressView().controlSize(.small) }
        Button("Refresh presets") { Task { await state.loadCatalog(runtime: store.runtime) } }
          .disabled(state.loadingCatalog || !ModelSetupState.rendererAvailable(store.runtime))
      }
      Text(
        "Choose a preset, locate compatible models, then create a recipe. Existing model files can stay in their current folders."
      )
      .font(.caption).foregroundStyle(.secondary)
      if !ModelSetupState.rendererAvailable(store.runtime) {
        Label(
          "Set up or connect the renderer above to load H3, LTX 2.3 and LTX 2.5 presets.",
          systemImage: "arrow.up.circle"
        )
        .font(.caption).foregroundStyle(.secondary)
      }
      if !state.catalogError.isEmpty {
        Text(state.catalogError).font(.caption).foregroundStyle(.red).textSelection(.enabled)
      }
      ForEach(state.presets) { preset in
        HStack(alignment: .top, spacing: 12) {
          VStack(alignment: .leading, spacing: 4) {
            Text(preset.name).font(.subheadline.weight(.medium))
            Text(preset.description).font(.caption).foregroundStyle(.secondary)
          }
          Spacer()
          Button("Set Up…") { state.begin(preset) }
            .disabled(
              bridge.busy || rendererInstalling || !ModelSetupState.rendererAvailable(store.runtime)
            )
        }.padding(12).background(.quaternary.opacity(0.4), in: RoundedRectangle(cornerRadius: 8))
      }
    }
    .task(id: store.runtime.root + "|" + store.runtime.pythonPath) {
      await state.loadCatalog(runtime: store.runtime)
    }
    .sheet(item: $state.selectedPreset) { preset in
      ModelSetupView(state: state, bridge: bridge, preset: preset).environmentObject(store)
    }
  }
}

struct ModelSetupView: View {
  @EnvironmentObject var store: StudioStore
  @ObservedObject var state: ModelSetupState
  @ObservedObject var bridge: Bridge
  var preset: ModelSetupPreset
  @State private var showLog = false
  @State private var sourceTermsReviewed = false

  var body: some View {
    VStack(alignment: .leading, spacing: 16) {
      HStack {
        VStack(alignment: .leading, spacing: 4) {
          Text(preset.name).font(.title2)
          Text("Set up compatible models").font(.caption).foregroundStyle(.secondary)
        }
        Spacer()
        Button("Done") { state.selectedPreset = nil }.disabled(bridge.busy)
      }
      Divider()
      ScrollView {
        VStack(alignment: .leading, spacing: 18) {
          existingModels
          componentChoices
          memoryPolicy
          if !compatibleDownloads.isEmpty { downloads }
          if !state.error.isEmpty {
            Label(state.error, systemImage: "exclamationmark.triangle")
              .foregroundStyle(.red).font(.callout).textSelection(.enabled)
            Text(
              "Choose a replacement component below its label or add another model folder and scan again."
            )
            .font(.caption).foregroundStyle(.secondary)
          }
          ForEach(Array(state.warnings.enumerated()), id: \.offset) { _, warning in
            Label(warning, systemImage: "info.circle").font(.caption).foregroundStyle(.secondary)
          }
          if !state.resultPath.isEmpty {
            Label("Recipe created", systemImage: "checkmark.circle.fill").foregroundStyle(.green)
            Text(state.resultPath).font(.caption).textSelection(.enabled)
            if let clip = store.selectedClip, preset.supports(clip) {
              Button("Use Recipe for Selected Clip") {
                state.useRecipeForSelectedClip(store: store)
              }
              .disabled(bridge.busy || clip.profileID == state.resultPath)
              Text(
                clip.profileID == state.resultPath
                  ? "Selected for \(clip.name). Prepare the clip to validate its media and settings."
                  : "Apply this recipe to \(clip.name)."
              )
              .font(.caption).foregroundStyle(.secondary)
            } else {
              Text("Select a clip with a compatible engine and media task to use this recipe.")
                .font(.caption).foregroundStyle(.secondary)
            }
            Button("Show recipe in Finder") {
              NSWorkspace.shared.activateFileViewerSelecting([
                URL(fileURLWithPath: state.resultPath)
              ])
            }
          }
          DisclosureGroup("Setup log", isExpanded: $showLog) {
            Text(displayLog.isEmpty ? "Setup activity appears here." : displayLog)
              .font(.system(size: 10, design: .monospaced)).textSelection(.enabled)
              .frame(maxWidth: .infinity, alignment: .leading)
          }
        }.padding(.trailing, 6)
      }
      Divider()
      HStack(spacing: 12) {
        if bridge.busy {
          if bridge.fraction > 0 {
            ProgressView(value: bridge.fraction).frame(width: 80)
          } else {
            ProgressView().controlSize(.small)
          }
          Text(bridge.message).font(.caption).lineLimit(2)
          Button("Cancel") { bridge.cancel() }
        } else if !state.resultPath.isEmpty {
          Label("Recipe created", systemImage: "checkmark.circle.fill")
            .font(.caption).foregroundStyle(.green)
        } else {
          Text("Creates a new validated recipe in your model recipes folder.")
            .font(.caption).foregroundStyle(.secondary)
        }
        Spacer()
        Button("Create Recipe") { Task { await state.createRecipe(store: store) } }
          .buttonStyle(.borderedProminent)
          .disabled(
            bridge.busy || !state.selection.missingComponents(for: preset).isEmpty
              || !state.resultPath.isEmpty)
      }
    }.padding(24).frame(width: 720, height: 760)
      .interactiveDismissDisabled(bridge.busy)
  }

  private var displayLog: String {
    bridge.busy ? state.setupLog + "\n" + bridge.log : state.setupLog
  }

  private var existingModels: some View {
    VStack(alignment: .leading, spacing: 10) {
      Text("Use Existing Models").font(.headline)
      Text(
        "Only folders you choose are scanned. Component headers and manifests determine compatibility."
      )
      .font(.caption).foregroundStyle(.secondary)
      ForEach(state.roots, id: \.self) { root in
        HStack {
          Image(systemName: "folder")
          Text(root).font(.caption).lineLimit(2).textSelection(.enabled)
          Spacer()
          Button {
            state.roots.removeAll { $0 == root }
          } label: {
            Image(systemName: "minus.circle")
          }
          .help("Remove folder from scan")
        }
      }
      HStack {
        Button("Choose Model Folders…") {
          let panel = NSOpenPanel()
          panel.canChooseFiles = false
          panel.canChooseDirectories = true
          panel.allowsMultipleSelection = true
          guard panel.runModal() == .OK else { return }
          for url in panel.urls where !state.roots.contains(url.path) {
            state.roots.append(url.path)
          }
        }
        Button("Scan Selected Folders") { Task { await state.scan(store: store) } }
          .disabled(state.roots.isEmpty)
      }
    }.disabled(bridge.busy)
  }

  private var componentChoices: some View {
    VStack(alignment: .leading, spacing: 12) {
      Text("Compatible components").font(.headline)
      ForEach(preset.components) { component in
        componentRow(component)
      }
    }.disabled(bridge.busy)
  }

  private func componentRow(_ component: ModelSetupComponent) -> some View {
    let paths = state.selection.candidates[component.key] ?? []
    let selected = state.selection.components[component.key] ?? ""
    let choices = Array(Set(paths + (selected.isEmpty ? [] : [selected]))).sorted()
    return VStack(alignment: .leading, spacing: 5) {
      HStack {
        Text(component.label).font(.subheadline.weight(.medium))
        Spacer()
        if selected.isEmpty {
          Text(paths.count > 1 ? "Choose from \(paths.count) matches" : "Required")
            .font(.caption).foregroundStyle(.orange)
        }
        Button("Browse…") {
          let panel = NSOpenPanel()
          let accepts = component.accepts ?? ["file", "directory"]
          panel.canChooseDirectories = accepts.contains("directory")
          panel.canChooseFiles = accepts.contains("file")
          guard panel.runModal() == .OK, let url = panel.url else { return }
          state.selection.components[component.key] = url.path
          state.resultPath = ""
        }
      }
      if !choices.isEmpty {
        Picker(
          component.label,
          selection: Binding(
            get: { state.selection.components[component.key] ?? "" },
            set: {
              state.selection.components[component.key] = $0
              state.resultPath = ""
            }
          )
        ) {
          Text("Choose a component…").tag("")
          ForEach(choices, id: \.self) { path in
            Text(componentLabel(path, choices: choices)).tag(path).help(path)
          }
        }.labelsHidden()
        if !selected.isEmpty {
          Text(selected).font(.caption2).foregroundStyle(.secondary).textSelection(.enabled)
        }
      } else {
        Text("Scan a model folder or browse for this component.").font(.caption).foregroundStyle(
          .secondary)
      }
    }
  }

  private func componentLabel(_ path: String, choices: [String]) -> String {
    let url = URL(fileURLWithPath: path)
    let filename = url.lastPathComponent
    let matches = choices.filter { URL(fileURLWithPath: $0).lastPathComponent == filename }
    guard matches.count > 1 else { return filename }
    let parentParts = url.deletingLastPathComponent().pathComponents
    for depth in 1...max(1, parentParts.count) {
      let suffix = parentParts.suffix(depth).joined(separator: "/")
      let duplicates = matches.filter {
        URL(fileURLWithPath: $0).deletingLastPathComponent().pathComponents.suffix(depth)
          .joined(separator: "/") == suffix
      }
      if duplicates.count == 1 { return filename + " — " + suffix }
    }
    return filename + " — " + url.deletingLastPathComponent().path
  }

  private var memoryPolicy: some View {
    VStack(alignment: .leading, spacing: 8) {
      Text("Memory policy").font(.headline)
      Picker("Memory policy", selection: $state.memoryMode) {
        ForEach(ModelSetupMemoryMode.allCases) { Text($0.label).tag($0) }
      }.pickerStyle(.segmented)
      Text(state.memoryMode.detail).font(.caption).foregroundStyle(.secondary)
      Text(
        "This Mac: \(Double(ProcessInfo.processInfo.physicalMemory) / 1_073_741_824, specifier: "%.0f") GB unified memory"
      )
      .font(.caption).foregroundStyle(.secondary)
    }.disabled(bridge.busy).onChange(of: state.memoryMode) { _, _ in state.resultPath = "" }
  }

  private var compatibleDownloads: [ModelSetupDownload] {
    state.downloads.filter { $0.supports(engine: preset.engine) }
  }

  private var downloads: some View {
    DisclosureGroup("Download or prepare a model") {
      VStack(alignment: .leading, spacing: 10) {
        Picker("Available model", selection: $state.selectedDownloadID) {
          Text("Choose a model…").tag("")
          ForEach(compatibleDownloads) { Text($0.name).tag($0.id) }
        }
        if let download = compatibleDownloads.first(where: { $0.id == state.selectedDownloadID }) {
          Text(download.description).font(.caption)
          ModelDownloadAccessView()
          HStack {
            if let source = URL(string: download.sourceURL) {
              Link("Model source", destination: source)
            }
            if let license = URL(string: download.licenseURL) {
              Link("License terms", destination: license)
            }
          }.font(.caption)
          Text(
            "Download: \(bytes(download.downloadBytes)) · Space required: \(bytes(download.requiredDiskBytes))"
          )
          .font(.caption).foregroundStyle(.secondary)
          PathPicker(
            label: "Destination model library", value: $state.downloadDestination, directory: true)
          if !state.downloadDestination.isEmpty {
            Text(
              "Prepared model: "
                + URL(fileURLWithPath: state.downloadDestination).appendingPathComponent(
                  download.id
                ).path
            )
            .font(.caption).foregroundStyle(.secondary).textSelection(.enabled)
          }
          if let notice = download.licenseNotice, !notice.isEmpty {
            Text(notice).font(.caption).foregroundStyle(.secondary)
          }
          Toggle(
            "I have reviewed the source terms and am eligible to download and use this model",
            isOn: $sourceTermsReviewed
          )
          .font(.caption)
          Button("Download and Prepare") { Task { await state.download(store: store) } }
            .disabled(
              !sourceTermsReviewed
                || state.downloadDestination.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            )
          Text(
            "Review the source, license and destination before downloading. Exact source files in your selected model folders are verified and reused. When preparation finishes, scan the folders to find its compatible components."
          )
          .font(.caption).foregroundStyle(.secondary)
        }
        if !state.downloadMessage.isEmpty {
          Text(state.downloadMessage).font(.caption).textSelection(.enabled)
        }
      }.padding(.top, 10).disabled(bridge.busy)
        .onChange(of: state.selectedDownloadID) { _, _ in sourceTermsReviewed = false }
    }
  }

  private func bytes(_ value: Int64) -> String {
    ByteCountFormatter.string(fromByteCount: value, countStyle: .file)
  }
}
