import StudioCore
import SwiftUI

struct ModelDownloadAccessView: View {
  @State private var token = ""
  @State private var configured = false
  @State private var status = ""
  @State private var error = ""

  var body: some View {
    DisclosureGroup("Hugging Face access (optional)") {
      VStack(alignment: .leading, spacing: 8) {
        Text(
          "For models requiring account access, accept the model’s source terms, then save a read token. An existing Hugging Face login also works."
        )
        .font(.caption).foregroundStyle(.secondary)
        Link(
          "Create or manage read tokens",
          destination: URL(string: "https://huggingface.co/settings/tokens")!
        )
        .font(.caption)
        SecureField("Hugging Face read token", text: $token).textFieldStyle(.roundedBorder)
        HStack {
          Button("Save to Keychain") { save() }
            .disabled(token.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
          Button("Remove Saved Token") { remove() }.disabled(!configured)
          Spacer()
          Text(configured ? "Token configured" : "No Studio token saved")
            .font(.caption).foregroundStyle(.secondary)
        }
        Text("Stored in this Mac’s Keychain and used only for model downloads.")
          .font(.caption).foregroundStyle(.secondary)
        if !status.isEmpty { Text(status).font(.caption).foregroundStyle(.secondary) }
        if !error.isEmpty { Text(error).font(.caption).foregroundStyle(.red) }
      }.padding(.top, 8)
    }.onAppear { refresh() }.onDisappear { token = "" }
  }

  private func refresh() {
    do { configured = try ModelDownloadToken.isConfigured() } catch {
      self.error = error.localizedDescription
    }
  }

  private func save() {
    error = ""
    status = ""
    do {
      try ModelDownloadToken.save(token)
      token = ""
      configured = true
      status = "Token saved to Keychain."
    } catch { self.error = error.localizedDescription }
  }

  private func remove() {
    error = ""
    status = ""
    do {
      try ModelDownloadToken.remove()
      token = ""
      configured = false
      status = "Studio token removed. Existing Hugging Face logins remain available."
    } catch { self.error = error.localizedDescription }
  }
}
