import CoreFoundation
import Foundation

// This client deliberately omits consumables and paid amounts. Credentials and issued
// tokens remain inside the helper; neither is returned in JSON events or saved manifests.
final class CloudSession {
  struct Authorization {
    let token: String
    let account: [String: Any]
  }
  private let apiKey: String
  private let send: (URLRequest) throws -> [String: Any]
  init(apiKey: String, send: @escaping (URLRequest) throws -> [String: Any] = CloudHTTP.send) {
    self.apiKey = apiKey; self.send = send
  }

  private func request(_ path: String, body: [String: Any], token: String? = nil) throws -> [String: Any] {
    var request = URLRequest(url: URL(string: "https://api.drawthings.ai" + path)!)
    request.httpMethod = "POST"; request.timeoutInterval = 30
    request.setValue("application/json", forHTTPHeaderField: "Content-Type")
    if let token { request.setValue(token, forHTTPHeaderField: "Authorization") }
    request.httpBody = try JSONSerialization.data(withJSONObject: body)
    return try send(request)
  }

  private func login() throws -> String {
    guard !apiKey.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
      throw TransportError.authenticationRequired
    }
    let response = try request("/sdk/token", body: ["apiKey": apiKey, "appCheckType": "none"])
    guard let token = response["shortTermToken"] as? String, !token.isEmpty else {
      throw TransportError.authenticationRequired
    }
    return token
  }

  private func freeQuota(token: String, now: Double) throws -> [String: Any] {
    var query = URLRequest(url: URL(string: "https://api.drawthings.ai/billing/stripe/payg")!)
    query.httpMethod = "GET"; query.timeoutInterval = 30
    query.setValue(token, forHTTPHeaderField: "Authorization")
    let value = try send(query)
    guard let paid = value["paygEnabled"] as? NSNumber,
      CFGetTypeID(paid) == CFBooleanGetTypeID(), !paid.boolValue,
      let quota = value["freeQuota"] as? [String: Any],
      let month = quota["monthKey"] as? String, !month.isEmpty else {
      throw TransportError.billingUnverified
    }
    let limit = try Configuration.number(quota["limitRequests"], min: 1, max: 1000000, integer: true)
    let used = try Configuration.number(quota["usedRequests"], min: 0, max: limit, integer: true)
    let remaining = try Configuration.number(quota["remainingRequests"], min: 1, max: limit, integer: true)
    var calendar = Calendar(identifier: .gregorian); calendar.timeZone = TimeZone(secondsFromGMT: 0)!
    let date = calendar.dateComponents([.year, .month], from: Date(timeIntervalSince1970: now))
    let currentMonth = String(format: "%04d-%02d", date.year!, date.month!)
    guard used + remaining == limit, month == currentMonth else { throw TransportError.billingUnverified }
    return ["limitRequests": limit, "usedRequests": used, "remainingRequests": remaining, "monthKey": month]
  }

  // Read-only inspection never reserves a generation and never changes billing settings.
  func inspect(thresholds: [String: Any], now: Double) throws -> [String: Any] {
    guard now.isFinite else { throw TransportError.invalidRequest }
    let expires = try Configuration.number(thresholds["expiresAt"], min: now, max: Double.greatestFiniteMagnitude)
    let community = try Configuration.number(thresholds["community"], min: 0, max: Double.greatestFiniteMagnitude)
    let plus = try Configuration.number(thresholds["plus"], min: 0, max: Double.greatestFiniteMagnitude)
    guard expires > now else { throw TransportError.billingUnverified }
    let quota = try freeQuota(token: login(), now: now)
    return ["authenticated": true, "routeVerified": true, "billingRoute": "free",
      "limitMode": "cloud", "limitCU": min(community, plus),
      "policyExpiresAt": min(expires, now + 30), "monthlyQuota": quota,
      "limitBasis": "conservativeUntilGenerationAuthorization", "paygEnabled": false]
  }

  func authorize(blob: Data, estimatedCU: Double, thresholds: [String: Any], now: Double) throws -> Authorization {
    guard !apiKey.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
      now.isFinite, estimatedCU.isFinite, estimatedCU >= 0 else { throw TransportError.invalidRequest }
    let expires = try Configuration.number(thresholds["expiresAt"], min: 0, max: Double.greatestFiniteMagnitude)
    let community = try Configuration.number(thresholds["community"], min: 0, max: Double.greatestFiniteMagnitude)
    let plus = try Configuration.number(thresholds["plus"], min: 0, max: Double.greatestFiniteMagnitude)
    guard expires > now, estimatedCU < min(community, plus) else { throw TransportError.billingUnverified }
    let shortToken = try login()
    _ = try freeQuota(token: shortToken, now: now)
    // Authorize only when Generate was explicitly requested. /authenticate may reserve a request.
    let response = try request("/authenticate", body: [
      "blob": blob.base64EncodedString(), "fromBridge": true,
      "attestationSupported": false, "isSandbox": false
    ], token: shortToken)
    guard let token = response["gRPCToken"] as? String else { throw TransportError.authenticationRequired }
    // Claims are inspected only on a fresh response from the fixed HTTPS authority, never
    // on a token provided by a project, user JSON, cache or an untrusted endpoint.
    let claims = try Self.claims(token)
    let expiry = try Configuration.number(claims["exp"], min: 0, max: Double.greatestFiniteMagnitude)
    guard expiry > now, claims["fromBridge"] as? Bool == true,
      let userClass = claims["userClass"] as? String, ["community", "plus"].contains(userClass) else {
      throw TransportError.billingUnverified
    }
    if let kind = claims["consumableType"], !(kind is NSNull) {
      // Even free PAYG is kept out of the subscription/free-request path until separately verified.
      throw TransportError.billingUnverified
    }
    if let amount = claims["amount"], !(amount is NSNull) {
      guard try Configuration.number(amount, min: 0, max: 0) == 0 else { throw TransportError.billingUnverified }
    }
    let limit = userClass == "plus" ? plus : community
    guard estimatedCU < limit else { throw TransportError.billingUnverified }
    return Authorization(token: token, account: [
      "authenticated": true, "routeVerified": true, "billingRoute": "free",
      "limitMode": "cloud", "limitCU": limit, "policyExpiresAt": min(expires, expiry)
    ])
  }

  private static func claims(_ token: String) throws -> [String: Any] {
    let parts = token.split(separator: ".", omittingEmptySubsequences: false)
    guard parts.count == 3, token.utf8.count <= 65536 else { throw TransportError.authenticationRequired }
    var body = String(parts[1]).replacingOccurrences(of: "-", with: "+").replacingOccurrences(of: "_", with: "/")
    body += String(repeating: "=", count: (4-body.count%4)%4)
    guard let data = Data(base64Encoded: body),
      let claims = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
      throw TransportError.authenticationRequired
    }
    return claims
  }
}

private final class CloudHTTP: NSObject, URLSessionTaskDelegate {
  func urlSession(_ session: URLSession, task: URLSessionTask, willPerformHTTPRedirection response: HTTPURLResponse,
                  newRequest request: URLRequest, completionHandler: @escaping (URLRequest?) -> Void) {
    completionHandler(nil)
  }

  static func send(_ request: URLRequest) throws -> [String: Any] {
    let configuration = URLSessionConfiguration.ephemeral
    configuration.httpCookieStorage = nil; configuration.urlCredentialStorage = nil
    let session = URLSession(configuration: configuration, delegate: CloudHTTP(), delegateQueue: nil)
    defer { session.invalidateAndCancel() }
    let semaphore = DispatchSemaphore(value: 0)
    let lock = NSLock()
    var result: [String: Any]?
    let task = session.dataTask(with: request) { data, response, error in
      lock.lock(); defer { lock.unlock(); semaphore.signal() }
      guard error == nil, let response = response as? HTTPURLResponse, response.statusCode == 200,
        let data, data.count <= 1024 * 1024,
        let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return }
      result = object
    }
    task.resume()
    guard semaphore.wait(timeout: .now() + 31) == .success else { throw TransportError.authenticationRequired }
    lock.lock(); defer { lock.unlock() }
    guard let result else { throw TransportError.authenticationRequired }
    return result
  }
}
