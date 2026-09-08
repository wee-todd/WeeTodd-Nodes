import AppKit
import StudioCore
import SwiftUI
import UniformTypeIdentifiers

struct PromptEditor: View {
  @EnvironmentObject var store: StudioStore
  var body: some View {
    VStack(spacing: 0) {
      HStack(spacing: 14) {
        Button {
          store.showPrompt = false
        } label: {
          Label("Back to movie", systemImage: "arrow.left")
        }.keyboardShortcut(.escape, modifiers: [])
        Divider().frame(height: 20)
        Text(store.selectedClip?.name ?? "Prompt").font(.headline)
        Spacer()
        Text(store.selectedClip?.displayTask ?? "").font(.system(size: 11)).foregroundStyle(
          .secondary)
      }.padding(.horizontal, 24).frame(height: 64)
      Divider()
      if let clip = store.selectedClip {
        HSplitView {
          VStack(alignment: .leading, spacing: 16) {
            HStack {
              SmallLabel(text: "Shot direction")
              Spacer()
              Text("⌘ Return to open · Esc to close").font(.caption2).foregroundStyle(.tertiary)
            }
            Text("Describe what happens.").font(.system(size: 28, weight: .light))
            Text(
              "Action, camera, lighting, dialogue and sound. The complete prompt is shown before generation."
            ).font(.system(size: 12)).foregroundStyle(.secondary)
            TextEditor(
              text: Binding(
                get: { store.selectedClip?.prompt ?? "" },
                set: { v in store.editClip { $0.prompt = v } })
            ).font(.system(size: 15)).scrollContentBackground(.hidden).padding(12).background(
              Theme.raised, in: RoundedRectangle(cornerRadius: 8)
            ).overlay(RoundedRectangle(cornerRadius: 8).strokeBorder(Theme.line)).frame(
              minHeight: 250)
            if clip.engine == .h3 {
              HStack(alignment: .top, spacing: 16) {
                VStack(alignment: .leading, spacing: 7) {
                  SmallLabel(text: "Soundscape")
                  TextField(
                    "Ambience and physical sounds",
                    text: Binding(
                      get: { store.selectedClip?.soundscape ?? "" },
                      set: { v in store.editClip { $0.soundscape = v } }), axis: .vertical
                  ).lineLimit(2...4)
                }
                VStack(alignment: .leading, spacing: 7) {
                  SmallLabel(text: "Music")
                  TextField(
                    "N/A or music direction",
                    text: Binding(
                      get: { store.selectedClip?.music ?? "N/A" },
                      set: { v in store.editClip { $0.music = v } }), axis: .vertical
                  ).lineLimit(2...4)
                }
              }.textFieldStyle(.roundedBorder)
              if ["ref2va", "a2v", "extension"].contains(clip.inferredTask) {
                Label(
                  "Paste the complete native H3 reference prompt for this task. Reference descriptions are never invented.",
                  systemImage: "info.circle"
                ).font(.caption).foregroundStyle(.secondary)
              }
            } else {
              DisclosureGroup("Negative prompt") {
                TextEditor(
                  text: Binding(
                    get: { store.selectedClip?.negativePrompt ?? "" },
                    set: { v in store.editClip { $0.negativePrompt = v } })
                ).font(.system(size: 12)).frame(height: 65)
              }.font(.caption)
            }
            HStack {
              Button("Save as prompt asset") {
                var a = MediaAsset(
                  name: clip.name + " prompt", kind: .text, scope: .clip, owner: clip.id)
                a.text = clip.prompt
                store.change { $0.assets.append(a) }
                store.notice = "Prompt saved in Clip Assets."
              }
              Spacer()
              Text("\(clip.prompt.count) characters").font(.caption).foregroundStyle(.secondary)
            }
          }.padding(28).frame(minWidth: 520, maxWidth: .infinity)
          VStack(alignment: .leading, spacing: 16) {
            SmallLabel(text: "Shot context")
            HStack {
              Text(clip.engine.label).font(.headline)
              Spacer()
              Text("\(clip.duration,specifier:"%.1f") sec").font(
                .system(size: 12, design: .monospaced))
            }
            Text(
              "\(clip.generationWidth) × \(clip.generationHeight) generation · movie settings applied at finishing"
            ).font(.caption).foregroundStyle(.secondary)
            ScrollView {
              VStack(alignment: .leading, spacing: 10) {
                ForEach(clip.attachments) { a in AttachmentRow(attachment: a) }
                if clip.attachments.isEmpty {
                  Text(
                    "No media conditioning. Add assets in the main window, then choose their role."
                  ).font(.caption).foregroundStyle(.secondary)
                }
                Divider().padding(.vertical, 8)
                SmallLabel(text: "Exact render prompt")
                Text(
                  store.preparedPrompt.isEmpty
                    ? "Prepare this clip to validate its model, inputs and task. The resolved prompt appears here."
                    : store.preparedPrompt
                ).font(.system(size: 11, design: .monospaced)).textSelection(.enabled)
                  .foregroundStyle(store.preparedPrompt.isEmpty ? .secondary : .primary)
                if !store.preparedReport.isEmpty {
                  DisclosureGroup("Resolved settings and validation") {
                    Text(store.preparedReport).font(.system(size: 9, design: .monospaced))
                      .textSelection(.enabled)
                  }.font(.caption)
                }
              }
            }
          }.padding(24).frame(minWidth: 310, idealWidth: 390, maxWidth: 490).background(Theme.panel)
        }
        Divider()
        PromptActions(bridge: store.bridge).environmentObject(store).padding(.horizontal, 28).frame(
          height: 68)
      } else {
        Spacer()
        Text("Select a generated clip first.")
        Spacer()
      }
    }.background(Theme.background).frame(maxWidth: .infinity, maxHeight: .infinity)
  }
}
struct PromptActions: View {
  @EnvironmentObject var store: StudioStore
  @ObservedObject var bridge: Bridge
  var body: some View {
    HStack {
      if bridge.busy {
        ProgressView().controlSize(.small)
        Text(bridge.message).font(.caption)
        Button("Cancel") { bridge.cancel() }
      } else {
        Image(systemName: store.preparedRecipe == nil ? "checklist" : "checkmark.circle.fill")
          .foregroundStyle(store.preparedRecipe == nil ? Color.secondary : Color.green)
        Text(
          store.preparedRecipe == nil
            ? "Prepare → review → render" : "Preflight passed. This exact recipe will run."
        ).font(.caption).foregroundStyle(.secondary)
      }
      Spacer()
      Button("View log") { store.showLog = true }
      Button("Prepare clip") { Task { await store.prepareSelected() } }.disabled(bridge.busy)
      Button {
        Task { await store.renderPrepared() }
      } label: {
        Label("Generate clip", systemImage: "play.fill")
      }.buttonStyle(.borderedProminent).disabled(bridge.busy || store.preparedRecipe == nil)
    }
  }
}
struct PathPicker: View {
  var label: String
  @Binding var value: String
  var directory = false
  var body: some View {
    VStack(alignment: .leading, spacing: 5) {
      Text(label).font(.caption).foregroundStyle(.secondary)
      HStack {
        TextField(label, text: $value).textFieldStyle(.roundedBorder)
        Button("Choose…") {
          let panel = NSOpenPanel()
          panel.canChooseDirectories = directory
          panel.canChooseFiles = !directory
          guard panel.runModal() == .OK, let url = panel.url else { return }
          value = url.path
        }
      }
    }
  }
}
struct RuntimeView: View {
  @EnvironmentObject var store: StudioStore
  @AppStorage("appearance") var appearance = "system"
  @StateObject private var installer = RuntimeInstaller()
  var body: some View {
    VStack(alignment: .leading, spacing: 20) {
      HStack {
        Text("Studio Settings").font(.title2)
        Spacer()
        Button("Done") {
          store.saveRuntime()
          store.showRuntime = false
        }.keyboardShortcut(.defaultAction)
      }
      ScrollView {
        VStack(alignment: .leading, spacing: 16) {
          Picker("Appearance", selection: $appearance) {
            Text("System").tag("system")
            Text("Light").tag("light")
            Text("Dark").tag("dark")
          }.pickerStyle(.segmented)
          Text("Renderer").font(.headline)
          Text(
            "Studio can install a private native renderer with its own Python and verified dependencies. Existing model files stay shared. Advanced users can connect an existing environment."
          ).font(.caption).foregroundStyle(.secondary)
          HStack {
            Button(installer.busy ? "Setting up…" : "Set Up Managed Renderer") {
              Task {
                do {
                  let source = Bundle.main.resourceURL?.appendingPathComponent("RendererSource")
                  let root =
                    source.flatMap { FileManager.default.fileExists(atPath: $0.path) ? $0 : nil }
                    ?? URL(fileURLWithPath: store.runtime.root)
                  let result = try await installer.install(source: root)
                  if let root = result["root"], let python = result["pythonPath"] {
                    store.runtime.root = root
                    store.runtime.pythonPath = python
                    store.saveRuntime()
                  }
                } catch { store.error = error.localizedDescription }
              }
            }.disabled(installer.busy || store.bridge.busy)
            if installer.busy {
              ProgressView().controlSize(.small)
              Button("Cancel") { installer.cancel() }
            }
          }
          if !installer.message.isEmpty { Text(installer.message).font(.caption) }
          if !installer.log.isEmpty {
            DisclosureGroup("Setup log") {
              ScrollView {
                Text(installer.log).font(.system(size: 10, design: .monospaced)).textSelection(
                  .enabled)
              }.frame(height: 120)
            }
          }
          PathPicker(label: "WeeTodd-Nodes repository", value: $store.runtime.root, directory: true)
          PathPicker(label: "Python executable", value: $store.runtime.pythonPath)
          PathPicker(
            label: "Model recipes folder", value: $store.runtime.profilesDirectory, directory: true)
          HStack {
            Button("Import model recipes…") { store.importRecipes() }
            Button("Refresh") { store.saveRuntime() }
            Spacer()
            Text("\(store.profiles.count) recipes").foregroundStyle(.secondary)
          }
          Text(
            "A recipe identifies a compatible component set and sampling policy. Automatic selection matches the clip’s engine and media roles."
          ).font(.caption).foregroundStyle(.secondary)
          Divider()
          Text("Finishing tools").font(.headline)
          PathPicker(
            label: "FFmpeg (blank uses installed executable)", value: $store.runtime.ffmpegPath)
          PathPicker(
            label: "FFprobe (blank uses installed executable)", value: $store.runtime.ffprobePath)
          PathPicker(label: "RIFE MLX executable", value: $store.runtime.rifePath)
          PathPicker(
            label: "RIFE weights folder", value: $store.runtime.rifeWeights, directory: true)
          PathPicker(label: "StudioMetal helper", value: $store.runtime.metalPath)
          Text(
            "MetalFX spatial upscaling uses the local GPU. Experimental MetalFX frame interpolation additionally requires matched depth and motion guides in the clip’s advanced settings."
          ).font(.caption).foregroundStyle(.secondary)
        }
      }
    }.padding(26).frame(width: 690, height: 730)
  }
}
