import DataModels
import Foundation
import GRPC
import GRPCImageServiceModels
import GRPCServer
import ModelZoo
import NIOHPACK

public enum Generation {
  // Authorization is resolved before submission; a failed call is never automatically retried.
  public static func run(_ request: [String: Any], authorize: (Data) throws -> String?,
                         progress: @escaping ([String: Any]) -> Void) throws -> [String: Any] {
    guard let id = request["requestID"] as? String, !id.isEmpty,
      let prompt = request["prompt"] as? String,
      let outputPath = request["outputDirectory"] as? String, outputPath.hasPrefix("/"),
      let operation = request["operation"] as? String,
      let profile = request["profile"] as? [String: Any],
      let host = profile["host"] as? String, !host.isEmpty,
      let useTLS = profile["useTLS"] as? Bool else { throw TransportError.invalidRequest }
    let port = Int(try Configuration.number(profile["port"], min: 1, max: 65535, integer: true))
    let secret = (request["credentials"] as? [String: String])?["sharedSecret"]
    let catalog = try Discovery.fetch(request, inspectAccount: false)
    let configuration = try Configuration.resolve(request)
    var payload = ImageGenerationRequest()
    payload.prompt = prompt
    payload.negativePrompt = request["negativePrompt"] as? String ?? ""
    payload.configuration = configuration.toData()
    payload.scaleFactor = 1; payload.chunked = true; payload.device = .laptop
    payload.user = "WeeTodd Studio"
    if let secret { payload.sharedSecret = secret }
    guard let files = catalog["files"] as? [String],
      files.contains(configuration.model ?? "") else { throw TransportError.unsupportedModel }
    try Conditioning.validateAvailability(request, catalog: catalog)
    try Conditioning.apply(request, to: &payload, width: Int(configuration.startWidth) * 64,
                           height: Int(configuration.startHeight) * 64)
    let client = ImageGenerationClientWrapper(deviceName: "WeeTodd Studio")
    try client.connect(host: host, port: port, TLS: useTLS, hostnameVerification: useTLS,
                       sharedSecret: secret)
    defer { try? client.disconnect() }
    guard let rpc = client.client else { throw TransportError.connectionFailed }
    let root = URL(fileURLWithPath: outputPath, isDirectory: true)
    let requiresAudio = Configuration.requiresAudio(configuration)
    let writer = try ArtifactWriter(root: root, requestID: id, operation: operation,
      expectedFrames: operation == "video" ? Int(configuration.numFrames) : 1,
      fps: operation == "video" ? Int(configuration.fpsId) : 1,
      sampleRate: requiresAudio ? ModelZoo.audioSampleRateForModel(configuration.model ?? "") : 0,
      requiresAudio: requiresAudio,
      ltxAudio: [.ltx2, .ltx2_3].contains(ModelZoo.versionForModel(configuration.model ?? "")))
    let stream = TensorStream(writer: writer)
    // All local, inventory and connection checks precede any quota reservation.
    // From this marker onward cancellation may leave a reserved/submitted request.
    progress(["stage": "submitting"])
    let bearer = try authorize(payload.serializedData())
    var headers = HPACKHeaders()
    if let bearer { headers.add(name: "authorization", value: "bearer \(bearer)") }
    let lock = NSLock()
    var decodeFailure: Error?
    let call = rpc.generateImage(payload, callOptions: CallOptions(customMetadata: headers)) { response in
      lock.lock(); defer { lock.unlock() }
      guard decodeFailure == nil else { return }
      do {
        if response.hasScaleFactor, response.scaleFactor != 1 { throw TransportError.invalidMedia }
        try stream.receive(response)
        if response.hasCurrentSignpost {
          progress(["stage": "generating", "framesReceived": writer.frameCount])
        }
      } catch { decodeFailure = error }
    }
    let status: GRPCStatus
    do { status = try call.status.wait() }
    catch { throw TransportError.submissionUncertain }
    guard status.code == .ok else {
      if status.code == .permissionDenied || status.code == .unauthenticated {
        throw TransportError.authenticationRequired
      }
      throw TransportError.submissionUncertain
    }
    lock.lock(); defer { lock.unlock() }
    if let decodeFailure { throw decodeFailure }
    let expectedWidth = Int(configuration.startWidth) * 64
    let expectedHeight = Int(configuration.startHeight) * 64
    guard writer.width == expectedWidth, writer.height == expectedHeight else {
      throw TransportError.invalidMedia
    }
    _ = try stream.finish(configuration: request["configuration"] as? [String: Any] ?? [:])
    return ["manifestPath": root.appendingPathComponent("manifest.json").path]
  }
}
