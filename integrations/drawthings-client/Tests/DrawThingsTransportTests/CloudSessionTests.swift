import Foundation
import XCTest
@testable import DrawThingsTransport

final class CloudSessionTests: XCTestCase {
  let free: [String: Any] = ["paygEnabled": false,
    "freeQuota": ["limitRequests": 20, "usedRequests": 1, "remainingRequests": 19, "monthKey": "1970-01"]]
  func testMissingLimitsStillChecksSavedKeyAndFreeAllowance() throws {
    var paths: [String] = []
    let session = CloudSession(apiKey: "fixture-key", send: { request in
      paths.append(request.url!.path)
      return request.url!.path == "/sdk/token" ? ["shortTermToken": "fixture-secret"] : self.free
    })
    let account = try session.inspect(thresholds: [:], now: 100)
    XCTAssertEqual(paths, ["/sdk/token", "/billing/stripe/payg"])
    XCTAssertEqual(account["authenticated"] as? Bool, true)
    XCTAssertEqual(account["billingRoute"] as? String, "free")
    XCTAssertNil(account["limitCU"])
    XCTAssertEqual(account["limitEnforcement"] as? String, "server")
    XCTAssertNotNil(account["reason"])
  }
  func testInspectionIsReadOnlyAndDoesNotExposeTokens() throws {
    var calls: [URLRequest] = []
    let session = CloudSession(apiKey: "fixture-key", send: { request in
      calls.append(request)
      return request.url!.path == "/sdk/token" ? ["shortTermToken": "fixture-secret"] : self.free
    })
    let account = try session.inspect(thresholds: ["community": 100, "plus": 400, "expiresAt": 200], now: 100)
    XCTAssertEqual(calls.map { $0.url!.path }, ["/sdk/token", "/billing/stripe/payg"])
    XCTAssertEqual(calls[1].httpMethod, "GET")
    XCTAssertNil(calls[1].httpBody)
    XCTAssertEqual(account["limitCU"] as? Double, 100)
    XCTAssertEqual(account["policyExpiresAt"] as? Double, 130)
    XCTAssertEqual((account["monthlyQuota"] as? [String: Any])?["remainingRequests"] as? Double, 19)
    let json = String(decoding: try JSONSerialization.data(withJSONObject: account), as: UTF8.self)
    XCTAssertFalse(json.contains("fixture-secret")); XCTAssertFalse(json.contains("fixture-key"))
  }
  func testUnknownExhaustedOrPaidAccountNeverReservesGeneration() throws {
    for policy: [String: Any] in [[:], free.merging(["paygEnabled": true]) { _, new in new },
      free.merging(["paygEnabled": 0]) { _, new in new },
      ["paygEnabled": false, "freeQuota": ["limitRequests": 20, "usedRequests": 20, "remainingRequests": 0, "monthKey": "1970-01"]]] {
      var calls: [String] = []
      let session = CloudSession(apiKey: "fixture", send: { request in
        calls.append(request.url!.path)
        return request.url!.path == "/sdk/token" ? ["shortTermToken": "fixture"] : policy
      })
      XCTAssertThrowsError(try session.authorize(blob: Data(), estimatedCU: 1,
        thresholds: ["community": 100, "plus": 400, "expiresAt": 200], now: 100))
      XCTAssertFalse(calls.contains("/authenticate"))
    }
  }
  func token(_ claims: [String: Any]) throws -> String {
    let payload = try JSONSerialization.data(withJSONObject: claims).base64EncodedString()
      .replacingOccurrences(of: "+", with: "-").replacingOccurrences(of: "/", with: "_")
      .replacingOccurrences(of: "=", with: "")
    return "fixture.\(payload).fixture"
  }
  func testFreeAuthorizationNeverRequestsPAYGOrBoostAndChecksIssuedClaims() throws {
    var calls: [URLRequest] = []
    let jwt = try token(["userClass": "community", "fromBridge": true, "exp": 300,
                         "checksum": "fixture", "nonce": "fixture"])
    let session = CloudSession(apiKey: "fixture-key", send: { request in
      calls.append(request)
      switch request.url!.path {
      case "/sdk/token": return ["shortTermToken": "fixture-session", "expiresIn": 3600]
      case "/billing/stripe/payg": return self.free
      case "/authenticate": return ["gRPCToken": jwt]
      default: throw TransportError.invalidRequest
      }
    })
    let result = try session.authorize(blob: Data("fixture".utf8), estimatedCU: 208,
      thresholds: ["community": 10000, "plus": 40000, "expiresAt": 200], now: 100)
    XCTAssertEqual(result.token, jwt)
    XCTAssertEqual(result.account["limitCU"] as? Double, 10000)
    XCTAssertEqual(result.account["billingRoute"] as? String, "free")
    XCTAssertEqual(calls.count, 3)
    let body = try JSONSerialization.jsonObject(with: calls[2].httpBody!) as! [String: Any]
    XCTAssertNil(body["consumableType"])
    XCTAssertNil(body["amount"])
    XCTAssertEqual(body["fromBridge"] as? Bool, true)
    XCTAssertEqual(calls[2].value(forHTTPHeaderField: "Authorization"), "fixture-session")
  }
  func testPaidBoostUnknownAndExpiredClaimsNeverAuthorizeGeneration() throws {
    for change: [String: Any] in [
      ["consumableType": "payg"], ["consumableType": "boost"], ["consumableType": "future"],
      ["amount": 1], ["userClass": "banned"], ["userClass": "unknown"], ["exp": 99],
      ["fromBridge": false]
    ] {
      var claims: [String: Any] = ["userClass": "community", "fromBridge": true, "exp": 300]
      claims.merge(change) { _, new in new }
      let jwt = try token(claims)
      let session = CloudSession(apiKey: "fixture", send: { request in
        if request.url!.path == "/billing/stripe/payg" { return self.free }
        return request.url!.path == "/sdk/token"
          ? ["shortTermToken": "fixture-session", "expiresIn": 3600] : ["gRPCToken": jwt]
      })
      XCTAssertThrowsError(try session.authorize(blob: Data(), estimatedCU: 208,
        thresholds: ["community": 10000, "plus": 40000, "expiresAt": 200], now: 100))
    }
  }
  func testMissingExpiredOrInsufficientPolicyFailsBeforeAuthenticationReservation() throws {
    for thresholds: [String: Any] in [["community": 208], ["community": 208, "plus": 40000, "expiresAt": 200], ["community": 208, "plus": 208, "expiresAt": 200],
      ["community": 10000, "plus": 40000, "expiresAt": 99]] {
      var calls = 0
      let session = CloudSession(apiKey: "fixture", send: { _ in calls += 1; return [:] })
      XCTAssertThrowsError(try session.authorize(blob: Data(), estimatedCU: 208,
        thresholds: thresholds, now: 100))
      XCTAssertEqual(calls, 0)
    }
  }
  func testServerAuthorizationWithoutPublishedLimitsStillRequiresFreeIssuedClaims() throws {
    for paid in [false, true] {
      var paths: [String] = []
      var claims: [String: Any] = ["userClass": "community", "fromBridge": true, "exp": 300]
      if paid { claims["consumableType"] = "payg" }
      let jwt = try token(claims)
      let session = CloudSession(apiKey: "fixture-key", send: { request in
        paths.append(request.url!.path)
        if request.url!.path == "/sdk/token" { return ["shortTermToken": "fixture-session"] }
        if request.url!.path == "/billing/stripe/payg" { return self.free }
        let body = try JSONSerialization.jsonObject(with: request.httpBody!) as! [String: Any]
        XCTAssertNil(body["consumableType"]); XCTAssertNil(body["amount"])
        return ["gRPCToken": jwt]
      })
      if paid {
        XCTAssertThrowsError(try session.authorize(blob: Data(), estimatedCU: 2356, thresholds: [:], now: 100))
      } else {
        let result = try session.authorize(blob: Data(), estimatedCU: 2356, thresholds: [:], now: 100)
        XCTAssertEqual(result.account["limitEnforcement"] as? String, "server")
        XCTAssertNil(result.account["limitCU"])
      }
      XCTAssertEqual(paths, ["/sdk/token", "/billing/stripe/payg", "/authenticate"])
    }
  }
}
