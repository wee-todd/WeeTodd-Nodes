import AppKit
import StudioCore
import SwiftUI

struct MotionFidelityInspector: View {
  @EnvironmentObject var store: StudioStore
  let clip: Clip
  func setting<T>(_ key: WritableKeyPath<MotionFidelitySettings, T>) -> Binding<T> {
    Binding(get: { (store.selectedClip?.motionFidelity ?? MotionFidelitySettings())[keyPath: key] },
      set: { value in
        store.editClip { c in
          var s = c.motionFidelity ?? MotionFidelitySettings()
          s[keyPath: key] = value
          c.motionFidelity = s
        }
        store.refreshPreview()
      })
  }
  var body: some View {
    DisclosureGroup("Motion Fidelity · Experimental") {
      VStack(alignment: .leading, spacing: 10) {
        Text("De-Roping expands motion, refines it with H3, and restores the original timing and audio. Results vary; the original is retained.")
          .font(.caption).foregroundStyle(.secondary)
        Toggle("Use enhanced motion", isOn: setting(\.enabled)).disabled(clip.engine != .h3)
        if clip.engine != .h3 {
          Text("H3 clips only in this release.").font(.caption)
        } else if clip.motionFidelity?.enabled == true {
          Picker("Repair recipe", selection: Binding(get: { clip.motionRecipeID ?? "" },
            set: { value in store.editClip { $0.motionRecipeID = value } })) {
            Text("Use base render recipe").tag("")
            ForEach(store.profiles.filter { $0.engine == "h3" && $0.task == "t2v" }) {
              Text($0.name).tag($0.id)
            }
          }
          Text("Requires a plain H3 recipe with at least 16 steps. Accelerators, LoRAs and additional conditioning are not qualified.")
            .font(.caption).foregroundStyle(.secondary)
          Picker("Expansion", selection: setting(\.mode)) {
            Text("Adaptive motion analysis").tag("adaptive")
            Text("Uniform").tag("uniform")
          }
          Stepper("Maximum hold: \(clip.motionFidelity?.maxHold ?? 2)×", value: setting(\.maxHold), in: 2...4)
          Text("Refinement strength: \((clip.motionFidelity?.strength ?? 0.5).formatted(.number.precision(.fractionLength(2))))")
          Slider(value: setting(\.strength), in: 0.05...1, step: 0.01)
          Text("Initial video noise. Higher values allow larger changes to the source.")
            .font(.caption).foregroundStyle(.secondary)
          Toggle("Set refinement evaluations", isOn: Binding(
            get: { clip.motionFidelity?.evaluations != nil },
            set: { setting(\.evaluations).wrappedValue = $0 ? 14 : nil }))
          if clip.motionFidelity?.evaluations != nil {
            Stepper("Evaluations: \(clip.motionFidelity?.evaluations ?? 14)", value: Binding(
              get: { clip.motionFidelity?.evaluations ?? 14 },
              set: { setting(\.evaluations).wrappedValue = $0 }), in: 1...64)
            Text("Fixed work at every strength. More evaluations take longer; improvement is not guaranteed.")
              .font(.caption).foregroundStyle(.secondary)
          } else {
            Text("Uses the recipe’s evaluation count multiplied by strength, rounded up.")
              .font(.caption).foregroundStyle(.secondary)
          }
          Text("Motion sensitivity")
          Slider(value: setting(\.sensitivity), in: 0...1, step: 0.05)
          TextField("Seed", value: setting(\.seed), format: .number).textFieldStyle(.roundedBorder)
          TextField("Expanded frame budget (73–345)", value: setting(\.maxFrames), format: .number)
            .textFieldStyle(.roundedBorder)
          HStack {
            Button("Analyze") { Task { await store.enhanceMotion(analyze: true) } }
            Button("Enhance") { Task { await store.enhanceMotion(analyze: false) } }
          }.disabled(store.bridge.busy || clip.sourcePath.isEmpty)
          Text(clip.motionIsCurrent ? "Enhanced version active. Turn off to compare the original." : "Enhancement pending. Headless jobs can process it with the app closed.")
            .font(.caption).foregroundStyle(.secondary)
          if let result = clip.motionResult {
            Button("Open analysis report") { NSWorkspace.shared.open(URL(fileURLWithPath: result.report)) }
          }
        }
      }.padding(.top, 8)
    }
  }
}

@MainActor extension StudioStore {
  func enhanceMotion(analyze: Bool) async {
    guard let clip = selectedClip else { return }
    do {
      let folder = Self.supportDirectory.appendingPathComponent("Motion Fidelity")
        .appendingPathComponent(UUID().uuidString)
      let result = try await bridge.invoke(analyze ? "motion-analyze" : "motion-enhance",
        runtime: runtime, payload: try payload(), output: folder)
      if let value = result["motionResult"] {
        let saved = try JSONDecoder().decode(MotionFidelityResult.self,
          from: JSONSerialization.data(withJSONObject: value))
        guard let current = project.clips.first(where: { $0.id == clip.id }), current == clip else {
          notice = "Clip changed during enhancement. The result is saved in Motion Fidelity; it was not applied."
          return
        }
        change { p in
          guard let index = p.clips.firstIndex(where: { $0.id == clip.id }) else { return }
          p.clips[index].motionResult = saved
          p.assets.append(MediaAsset(name: clip.name + " · Motion Fidelity", kind: .video,
            path: saved.path, scope: .clip, owner: clip.id))
        }
        refreshPreview()
      }
      let plan = result["plan"] as? [String: Any] ?? [:]
      notice = "Motion Fidelity: \(plan["sourceFrames"] ?? 0) → \(plan["paddedFrames"] ?? 0) expanded frames. "
        + (analyze ? "Analysis saved; choose Enhance to refine." : "Original preserved.")
      if analyze, let report = result["report"] as? String {
        NSWorkspace.shared.open(URL(fileURLWithPath: report))
      }
    } catch { self.error = error.localizedDescription }
  }
}
