import AppKit
import CryptoKit
import Foundation
import StudioCore

@MainActor final class RuntimeInstaller: ObservableObject {
  @Published var busy = false
  @Published var message = ""
  @Published var log = ""
  private var process: Process?
  func cancel() { process?.interrupt() }
  private func run(_ executable: URL, _ arguments: [String], environment: [String: String])
    async throws -> String
  {
    let task = Process()
    task.executableURL = executable
    task.arguments = arguments
    task.environment = environment
    let pipe = Pipe()
    task.standardOutput = pipe
    task.standardError = pipe
    process = task
    try task.run()
    defer { process = nil }
    return try await withCheckedThrowingContinuation { continuation in
      DispatchQueue.global(qos: .userInitiated).async {
        var all = Data()
        while true {
          let part = pipe.fileHandleForReading.availableData
          if part.isEmpty { break }
          all.append(part)
          let text = String(decoding: part, as: UTF8.self)
          DispatchQueue.main.async { self.log = String((self.log + text).suffix(24000)) }
        }
        task.waitUntilExit()
        if task.terminationStatus == 0 {
          continuation.resume(returning: String(decoding: all, as: UTF8.self))
        } else {
          continuation.resume(
            throwing: StudioError.invalid(
              "Runtime setup stopped. Your previous runtime remains available. See the setup log."))
        }
      }
    }
  }
  func install(source: URL) async throws -> [String: String] {
    guard !busy else { throw StudioError.invalid("Runtime setup is already active.") }
    busy = true
    log = ""
    defer { busy = false }
    let base = StudioStore.supportDirectory.appendingPathComponent("RuntimeDownloads")
    try FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
    var env = ProcessInfo.processInfo.environment
    env["UV_PYTHON_INSTALL_DIR"] = base.appendingPathComponent("Python").path
    env["UV_CACHE_DIR"] = base.appendingPathComponent("Cache").path
    env["UV_PYTHON_BIN_DIR"] = base.appendingPathComponent("bin").path
    env["PYTHONUNBUFFERED"] = "1"
    let archive = base.appendingPathComponent("uv-0.12.8.tar.gz")
    message = "Downloading the runtime installer…"
    let url = URL(
      string:
        "https://github.com/astral-sh/uv/releases/download/0.12.8/uv-aarch64-apple-darwin.tar.gz")!
    let (data, response) = try await URLSession.shared.data(from: url)
    guard (response as? HTTPURLResponse)?.statusCode == 200 else {
      throw StudioError.invalid("The runtime download was unavailable. Try setup again later.")
    }
    let checksum = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    guard checksum == "8ce083658dbff20143607ca7af8e0c1d64b6fd7bf03a5cdcb62bf3d47d991b5f" else {
      throw StudioError.invalid("Runtime installer verification failed.")
    }
    try data.write(to: archive, options: .atomic)
    _ = try await run(
      URL(fileURLWithPath: "/usr/bin/tar"), ["-xzf", archive.path, "-C", base.path],
      environment: env)
    let uv = base.appendingPathComponent("uv-aarch64-apple-darwin/uv")
    message = "Installing private Python…"
    _ = try await run(uv, ["python", "install", "3.12.13", "--no-bin"], environment: env)
    let found = try await run(
      uv, ["python", "find", "--managed-python", "3.12.13"], environment: env)
    let python = URL(fileURLWithPath: found.trimmingCharacters(in: .whitespacesAndNewlines))
    let destination = StudioStore.supportDirectory.appendingPathComponent(
      "Runtimes/\(UUID().uuidString)")
    message = "Installing and verifying the MLX renderer…"
    _ = try await run(
      python,
      [
        source.appendingPathComponent("scripts/install_studio_runtime.py").path,
        "--source", source.path, "--destination", destination.path, "--uv", uv.path,
      ], environment: env)
    let result = try JSONDecoder().decode(
      [String: String].self,
      from: Data(contentsOf: destination.appendingPathComponent("runtime.json")))
    message = "Native renderer installed and verified."
    return result
  }
}
