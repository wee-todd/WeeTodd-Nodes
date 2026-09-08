import SwiftUI

struct MotionPromptEditor: View {
  @EnvironmentObject var store: StudioStore

  var saveDisabled: Bool {
    store.motionPromptLoading || store.motionPromptEditorError != nil
      || store.motionPromptDraft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
  }

  var body: some View {
    VStack(spacing: 0) {
      HStack(spacing: 14) {
        Button {
          store.cancelMotionPromptEditor()
        } label: {
          Label("Cancel", systemImage: "arrow.left")
        }.keyboardShortcut(.escape, modifiers: [])
        Divider().frame(height: 20)
        Text("Repair Prompt").font(.headline)
        Spacer()
        Text(store.motionPromptClipName.isEmpty ? "Clip unavailable" : store.motionPromptClipName)
          .font(.system(size: 11)).foregroundStyle(.secondary)
      }.padding(.horizontal, 24).frame(height: 64).background(Theme.panel)
      Divider()

      if store.motionPromptLoading {
        Spacer()
        ProgressView("Resolving and validating the repair recipe…")
        Spacer()
      } else if let message = store.motionPromptEditorError {
        Spacer()
        VStack(spacing: 14) {
          Image(systemName: "exclamationmark.triangle").font(.title).foregroundStyle(.orange)
          Text("Repair prompt unavailable").font(.title2)
          Text(message).foregroundStyle(.secondary).multilineTextAlignment(.center).frame(
            maxWidth: 520)
          HStack {
            Button("Close") { store.cancelMotionPromptEditor() }
            Button("Use Recipe Prompt") { store.useRecipeMotionPrompt() }
              .disabled(store.motionPromptSession?.clipID != store.selectedClipID)
            Button("Try Again") { Task { await store.retryMotionPromptEditor() } }
              .buttonStyle(.borderedProminent)
          }
        }
        Spacer()
      } else {
        VStack(alignment: .leading, spacing: 18) {
          HStack {
            VStack(alignment: .leading, spacing: 5) {
              SmallLabel(
                text: store.motionPromptUsingOverride ? "Custom override" : "Recipe prompt")
              Text("Direct the motion repair").font(.system(size: 28, weight: .light))
            }
            Spacer()
            Text("\(store.motionPromptDraft.count) characters").font(.caption.monospacedDigit())
              .foregroundStyle(.secondary)
          }
          Text(
            "Keep the source framing, subjects and scene recognizable. Describe the action and camera timing across the expanded motion interval so the refinement has a complete direction."
          ).font(.system(size: 12)).foregroundStyle(.secondary)
          TextEditor(text: $store.motionPromptDraft)
            .font(.system(size: 15)).scrollContentBackground(.hidden).padding(12)
            .background(Theme.raised, in: RoundedRectangle(cornerRadius: 8))
            .overlay(RoundedRectangle(cornerRadius: 8).strokeBorder(Theme.line))
            .frame(maxWidth: .infinity, maxHeight: .infinity)
          if store.motionPromptUsingOverride {
            DisclosureGroup("Recipe prompt") {
              ScrollView {
                Text(store.motionRecipePrompt).font(.system(size: 11, design: .monospaced))
                  .textSelection(.enabled).frame(maxWidth: .infinity, alignment: .leading)
              }.frame(maxHeight: 130)
            }.font(.caption)
          }
          Text(
            "Save keeps this text exactly as written. Use Recipe Prompt removes the clip override and returns to the resolved recipe text."
          ).font(.caption).foregroundStyle(.secondary)
        }.padding(28).frame(maxWidth: .infinity, maxHeight: .infinity)
        Divider()
        HStack {
          Button("Cancel") { store.cancelMotionPromptEditor() }
          Button("Use Recipe Prompt") { store.useRecipeMotionPrompt() }
            .disabled(store.motionPromptLoading)
          Spacer()
          Button("Save") { store.saveMotionPromptEditor() }
            .keyboardShortcut(.defaultAction).buttonStyle(.borderedProminent).disabled(saveDisabled)
        }.padding(.horizontal, 28).frame(height: 68).background(Theme.panel)
      }
    }.background(Theme.background).foregroundStyle(Theme.text)
      .frame(maxWidth: .infinity, maxHeight: .infinity)
  }
}
