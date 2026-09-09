import Foundation
import GRPCServer
import ModelZoo

public enum Discovery {
  public static func fetch(_ request: [String: Any], inspectAccount: Bool = true) throws -> [String: Any] {
    guard let profile = request["profile"] as? [String: Any],
      let route = profile["route"] as? String, ["grpc", "dtBridge", "dtCloud"].contains(route),
      let host = profile["host"] as? String, !host.isEmpty,
      let useTLS = profile["useTLS"] as? Bool else { throw TransportError.invalidRequest }
    let port = Int(try Configuration.number(profile["port"], min: 1, max: 65535, integer: true))
    if route == "dtCloud", (host != "compute.drawthings.ai" || port != 443 || !useTLS) {
      throw TransportError.invalidRequest
    }
    let credentials = request["credentials"] as? [String: String]
    let client = ImageGenerationClientWrapper(deviceName: "WeeTodd Studio")
    try client.connect(host: host, port: port, TLS: useTLS, hostnameVerification: useTLS,
                       sharedSecret: credentials?["sharedSecret"])
    defer { try? client.disconnect() }
    let semaphore = DispatchSemaphore(value: 0)
    let lock = NSLock()
    var result: [String: Any]?
    var failure: TransportError = .connectionFailed
    client.echo { success, authenticated, resources, limits, serverID in
      lock.lock()
      defer { lock.unlock(); semaphore.signal() }
      guard success, authenticated else {
        failure = success ? .authenticationRequired : .connectionFailed
        return
      }
      // Whitelist fields: model metadata can also include third-party API credentials.
      var value: [String: Any] = [
        "authenticated": true, "serverIdentifier": String(serverID),
        "files": resources.files,
        "capabilities": Capabilities.catalog(files: resources.files),
        "transport": ["tls": useTLS, "hostnameVerified": useTLS, "authenticated": true],
        "models": resources.files.compactMap { file -> [String: String]? in
          guard let model = ModelZoo.specificationForModel(file), Capabilities.rules(for: file) != nil else { return nil }
          return ["id": file, "name": model.name, "family": String(describing: model.version)]
        },
        "loras": resources.LoRAs.filter {
          resources.files.contains($0.file) && ($0.modifier.map { $0 == .none } ?? true)
            && $0.isConsistencyModel != true && $0.isLoHa != true && $0.alternativeDecoder == nil
        }.map { lora -> [String: Any] in
          ["id": lora.file, "name": lora.name, "family": String(describing: lora.version),
           "compatibleModelIDs": resources.files.filter {
             Capabilities.rules(for: $0) != nil && ModelZoo.versionForModel($0) == lora.version
           }]
        }
      ]
      if let limits {
        value["thresholds"] = ["community": limits.community, "plus": limits.plus,
                                "expiresAt": limits.expireAt.timeIntervalSince1970]
      }
      result = value
    }
    guard semaphore.wait(timeout: .now() + 10) == .success else {
      throw TransportError.connectionFailed
    }
    lock.lock()
    defer { lock.unlock() }
    guard var result else { throw failure }
    if route == "dtCloud" && inspectAccount {
      if let key = credentials?["apiKey"], let thresholds = result["thresholds"] as? [String: Any] {
        do {
          result["account"] = try CloudSession(apiKey: key).inspect(thresholds: thresholds,
            now: Date().timeIntervalSince1970)
        } catch {
          result["account"] = ["limitMode": "cloud",
            "routeVerified": false, "billingRoute": "unknown",
            "reason": "Free allowance could not be verified. Check the API key, available free requests, and PAYG setting in the Draw Things dashboard."]
        }
      }
    }
    return result
  }
}
