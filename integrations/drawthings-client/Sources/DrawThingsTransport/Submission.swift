import Foundation

public enum Submission {
  public static func run(_ request: [String: Any], progress: @escaping ([String: Any]) -> Void) throws -> [String: Any] {
    guard request["billingPolicy"] as? String == "freeOnly",
      let profile = request["profile"] as? [String: Any],
      let route = profile["route"] as? String else { throw TransportError.billingUnverified }
    switch route {
    case "grpc":
      // Stored connection configuration must identify a server with cloud offload disabled.
      guard profile["selfHostedConfirmed"] as? Bool == true else { throw TransportError.billingUnverified }
      return try Generation.run(request, authorize: { _ in nil }, progress: progress)
    case "dtCloud":
      let catalog = try Discovery.fetch(request, inspectAccount: false)
      let estimate = try ComputeEstimate.evaluate(request)
      guard let cu = estimate["cu"] as? NSNumber,
        let apiKey = (request["credentials"] as? [String: String])?["apiKey"] else {
        throw TransportError.authenticationRequired
      }
      let thresholds = catalog["thresholds"] as? [String: Any] ?? [:]
      let session = CloudSession(apiKey: apiKey)
      return try Generation.run(request, authorize: { blob in
        let auth = try session.authorize(blob: blob, estimatedCU: cu.doubleValue,
          thresholds: thresholds, now: Date().timeIntervalSince1970)
        progress(["stage": "authorized", "account": auth.account, "estimateCU": cu.doubleValue])
        return auth.token
      }, progress: progress)
    case "dtBridge":
      // Echo has no billing-route/account proof. Do not treat localhost as a free cloud grant.
      throw TransportError.billingUnverified
    default: throw TransportError.invalidRequest
    }
  }
}
