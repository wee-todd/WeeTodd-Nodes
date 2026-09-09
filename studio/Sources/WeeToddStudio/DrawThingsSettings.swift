import AppKit
import StudioCore
import SwiftUI

struct DrawThingsSettings: View {
  @EnvironmentObject var store: StudioStore
  @State private var draft = DrawThingsConnection()
  @State private var credential = ""
  @State private var message = ""
  var body: some View {
    VStack(alignment: .leading, spacing: 18) {
      HStack {
        Text("Draw Things Connections").font(.title2)
        Spacer()
        Button("Done") { store.showDrawThings = false }
      }
      HStack(alignment: .top, spacing: 22) {
        VStack(alignment: .leading) {
          ForEach(store.drawThingsConnections) { connection in
            Button(connection.name) { draft = connection; credential = ""; message = "" }
              .buttonStyle(.borderless)
          }
          Divider()
          Button("Add Connection") { draft = DrawThingsConnection(); credential = ""; message = "" }
        }.frame(width: 170, alignment: .leading)
        Form {
          TextField("Name", text: $draft.name)
          Picker("Route", selection: $draft.route) {
            Text("Self-hosted gRPC").tag("grpc")
            Text("DT+ App Bridge").tag("dtBridge")
            Text("Draw Things Cloud API").tag("dtCloud")
          }.onChange(of: draft.route) { _, route in
            draft.selfHostedConfirmed = false
            if route == "dtCloud" { draft.host = "compute.drawthings.ai"; draft.port = 443; draft.useTLS = true }
            else { draft.host = "127.0.0.1"; draft.port = 7859; draft.useTLS = false }
          }
          TextField("Host", text: $draft.host).disabled(draft.route == "dtCloud")
          TextField("Port", value: $draft.port, format: .number.grouping(.never)).disabled(draft.route == "dtCloud")
          Toggle("TLS with hostname verification", isOn: $draft.useTLS).disabled(draft.route == "dtCloud")
          if draft.route == "grpc" {
            Toggle("This server has cloud offload disabled", isOn: Binding(
              get: { draft.selfHostedConfirmed ?? false }, set: { draft.selfHostedConfirmed = $0 }))
          }
          SecureField(draft.route == "dtCloud" ? "API key" : "Shared secret (optional)", text: $credential)
          Text("Credentials are stored in this Mac’s Keychain. Leave blank to keep the saved credential.")
            .font(.caption).foregroundStyle(.secondary)
          if draft.route == "dtCloud" {
            Link("Open Draw Things API dashboard", destination: URL(string: "https://api.drawthings.ai/dashboard")!)
          } else if draft.route == "dtBridge" {
            Text("Draw Things must stay open in Bridge Mode. Free-only generation remains unavailable until the bridge exposes a verifiable billing route.")
              .font(.caption).foregroundStyle(.secondary)
          }
          HStack {
            Button("Save") { save() }
            Button("Test Connection") { if save() { Task { await store.testDrawThings(draft) } } }
              .disabled(store.bridge.busy)
            if let reference = draft.credentialRef {
              Button("Remove Credential") {
                do { try DrawThingsCredential.remove(reference); draft.credentialRef = nil; credential = ""; _ = save() }
                catch { message = error.localizedDescription }
              }
            }
          }
          if let catalog = store.drawThingsCatalogs[draft.id] {
            Text("\((catalog["capabilities"] as? [String: Any])?.count ?? 0) supported models · connection verified")
              .font(.caption).foregroundStyle(.secondary)
            if let account = catalog["account"] as? [String: Any] {
              if let quota = account["monthlyQuota"] as? [String: Any],
                let remaining = quota["remainingRequests"] as? NSNumber {
                Text("Free requests remaining: \(remaining.stringValue)").font(.caption)
              }
              if let reason = account["reason"] as? String { Text(reason).font(.caption).foregroundStyle(.secondary) }
            }
          }
          if !message.isEmpty { Text(message).font(.caption).foregroundStyle(.secondary) }
        }.formStyle(.grouped).frame(maxWidth: .infinity)
      }
      Divider()
      HStack {
        Text("Transport helper").font(.caption)
        TextField("WeeToddDrawThings", text: Binding(get: { store.runtime.drawThingsHelperPath ?? "" },
          set: { store.runtime.drawThingsHelperPath = $0 })).textFieldStyle(.roundedBorder)
        Button("Import…") {
          let panel = NSOpenPanel(); panel.canChooseDirectories = false; panel.allowsMultipleSelection = false
          if panel.runModal() == .OK, let url = panel.url {
            store.runtime.drawThingsHelperPath = url.path
            store.saveRuntime(reloadProfiles: false)
          }
        }
      }
    }.padding(24).frame(width: 840, height: 660)
      .onAppear { if let first = store.drawThingsConnections.first { draft = first } }
  }
  @discardableResult func save() -> Bool {
    guard !draft.name.trimmingCharacters(in: .whitespaces).isEmpty,
      !draft.host.isEmpty, (1...65535).contains(draft.port) else { message = "Enter a name, host, and valid port."; return false }
    do {
      if !credential.isEmpty {
        let reference = draft.credentialRef ?? draft.id
        try DrawThingsCredential.save(credential, reference: reference)
        draft.credentialRef = reference; credential = ""
      }
      if let index = store.drawThingsConnections.firstIndex(where: { $0.id == draft.id }) {
        if store.drawThingsConnections[index] != draft { store.drawThingsCatalogs.removeValue(forKey: draft.id) }
        store.drawThingsConnections[index] = draft
      } else { store.drawThingsConnections.append(draft) }
      store.saveDrawThingsConnections()
      store.saveRuntime(reloadProfiles: false)
      message = "Connection saved."
      return true
    } catch { message = error.localizedDescription; return false }
  }
}
