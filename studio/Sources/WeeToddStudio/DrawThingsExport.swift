import AppKit
import StudioCore

@MainActor extension StudioStore {
  func exportDrawThingsImageJob() {
    guard let draft = imageDraft,
      let connection = drawThingsConnections.first(where: { $0.id == draft.profileID }) else { return }
    let panel = NSSavePanel()
    panel.title = "Export Image Headless Job"
    panel.nameFieldStringValue = draft.name + ".weetodd-job.json"
    guard panel.runModal() == .OK, let url = panel.url else { return }
    Task {
      do {
        var imageProject = StudioProject()
        imageProject.name = draft.name
        let body: [String: Any] = ["project": try imageProject.object(), "generateIDs": [],
          "globalAssets": [], "drawThingsImageJobs": [["id": UUID().uuidString, "kind": "image",
          "request": draft.request(id: UUID().uuidString), "connection": try connection.object(),
          "dependsOn": []]]]
        _ = try await bridge.invoke("export-job", runtime: runtime, payload: body, output: url)
        notice = "Exported an image job for WeeToddCLI. You can close Studio before running it."
        NSWorkspace.shared.activateFileViewerSelecting([url])
      } catch { self.error = error.localizedDescription }
    }
  }
}
