import Foundation
import GRPC
import GRPCImageServiceModels
import NIO
import XCTest
@testable import DrawThingsTransport

private final class EchoFixture: ImageGenerationServiceProvider {
  var includeH3 = false
  var echoHasThresholds = true
  var hoursHasThresholds = true
  var interceptors: ImageGenerationServiceServerInterceptorFactoryProtocol? { nil }
  func echo(request: EchoRequest, context: StatusOnlyCallContext) -> EventLoopFuture<EchoReply> {
    context.eventLoop.makeSucceededFuture(EchoReply.with {
      $0.sharedSecretMissing = request.sharedSecret != "fixture-pass"
      $0.files = ["fixture-model.ckpt", "flux_2_klein_4b_q8p.ckpt"]
      if includeH3 {
        $0.files.append("fixture-h3.ckpt")
        $0.override.models = Data("""
        [{"name":"Fixture H3","file":"fixture-h3.ckpt","prefix":"",
          "version":"minimax_h3","modifier":"fl2va","upcast_attention":false,"default_scale":8}]
        """.utf8)
      }
      $0.serverIdentifier = 123
      if echoHasThresholds {
        $0.thresholds = ComputeUnitThreshold.with {
          $0.community = 10000; $0.plus = 40000; $0.expireAt = 2000000000
        }
      }
    })
  }
  func generateImage(request: ImageGenerationRequest,
                     context: StreamingResponseCallContext<ImageGenerationResponse>) -> EventLoopFuture<GRPCStatus> {
    context.eventLoop.makeFailedFuture(GRPCStatus(code: .unimplemented))
  }
  func filesExist(request: FileListRequest, context: StatusOnlyCallContext) -> EventLoopFuture<FileExistenceResponse> {
    context.eventLoop.makeFailedFuture(GRPCStatus(code: .unimplemented))
  }
  func uploadFile(context: StreamingResponseCallContext<UploadResponse>) -> EventLoopFuture<(StreamEvent<FileUploadRequest>) -> Void> {
    context.eventLoop.makeFailedFuture(GRPCStatus(code: .unimplemented))
  }
  func pubkey(request: PubkeyRequest, context: StatusOnlyCallContext) -> EventLoopFuture<PubkeyResponse> {
    context.eventLoop.makeFailedFuture(GRPCStatus(code: .unimplemented))
  }
  func hours(request: HoursRequest, context: StatusOnlyCallContext) -> EventLoopFuture<HoursResponse> {
    context.eventLoop.makeSucceededFuture(HoursResponse.with {
      if hoursHasThresholds {
        $0.thresholds = ComputeUnitThreshold.with {
          $0.community = 15000; $0.plus = 60000; $0.expireAt = 2000000000
        }
      }
    })
  }
}

final class DiscoveryTests: XCTestCase {
  func testEndpointH3MetadataEnablesFirstLastAndItsComputeEstimate() throws {
    let group = MultiThreadedEventLoopGroup(numberOfThreads: 1)
    defer { try? group.syncShutdownGracefully() }
    let fixture = EchoFixture(); fixture.includeH3 = true
    let server = try Server.insecure(group: group).withServiceProviders([fixture])
      .bind(host: "127.0.0.1", port: 0).wait()
    defer { try? server.close().wait() }
    let result = try Discovery.fetch(["profile": ["route": "grpc", "host": "127.0.0.1",
      "port": try XCTUnwrap(server.channel.localAddress?.port), "useTLS": false],
      "credentials": ["sharedSecret": "fixture-pass"]])
    let models = try XCTUnwrap(result["models"] as? [[String: String]])
    XCTAssertTrue(models.contains { $0["id"] == "fixture-h3.ckpt" && $0["family"] == "minimaxH3" })
    let request: [String: Any] = ["modelID": "fixture-h3.ckpt", "operation": "video",
      "configuration": ["width": 512, "height": 512, "steps": 19, "seed": 42, "fps": 24, "numFrames": 124]]
    let estimate = try ComputeEstimate.evaluate(request)
    XCTAssertGreaterThan(try XCTUnwrap(estimate["cu"] as? Int), 0)
    let config = try Configuration.resolve(request)
    XCTAssertTrue(Configuration.requiresAudio(config))
    XCTAssertEqual(config.shiftForAudio, 3)
    XCTAssertEqual(config.shift, 12)
    XCTAssertEqual(config.guidanceScale, 1)
    var fractionalRequest = request
    var values = try XCTUnwrap(request["configuration"] as? [String: Any])
    values["audioShift"] = 3.1
    fractionalRequest["configuration"] = values
    let fractionalEstimate = try ComputeEstimate.evaluate(fractionalRequest)
    let wireData = try JSONSerialization.data(withJSONObject: fractionalEstimate)
    let wireEstimate = try XCTUnwrap(JSONSerialization.jsonObject(with: wireData) as? [String: Any])
    let wireConfig = try XCTUnwrap(wireEstimate["configuration"] as? [String: Any])
    XCTAssertEqual(wireConfig["audioShift"] as? Double, 3.1)
  }
  func testSeparateHoursEndpointSuppliesMissingEchoLimits() throws {
    let group = MultiThreadedEventLoopGroup(numberOfThreads: 1)
    defer { try? group.syncShutdownGracefully() }
    let fixture = EchoFixture(); fixture.echoHasThresholds = false
    let server = try Server.insecure(group: group).withServiceProviders([fixture])
      .bind(host: "127.0.0.1", port: 0).wait()
    defer { try? server.close().wait() }
    let request: [String: Any] = ["profile": ["route": "dtBridge", "host": "127.0.0.1",
      "port": try XCTUnwrap(server.channel.localAddress?.port), "useTLS": false],
      "credentials": ["sharedSecret": "fixture-pass"]]
    let result = try Discovery.fetch(request)
    let policy = try XCTUnwrap(result["thresholds"] as? [String: Any])
    XCTAssertEqual(policy["community"] as? Int, 15000)
    XCTAssertEqual(policy["plus"] as? Int, 60000)
    XCTAssertEqual(policy["expiresAt"] as? Double, 2000000000)
    fixture.hoursHasThresholds = false
    XCTAssertNil(try Discovery.fetch(request)["thresholds"])
  }

  func testEchoDiscoversThresholdsAndRejectsMissingSecret() throws {
    let group = MultiThreadedEventLoopGroup(numberOfThreads: 1)
    defer { try? group.syncShutdownGracefully() }
    let server = try Server.insecure(group: group).withServiceProviders([EchoFixture()])
      .bind(host: "127.0.0.1", port: 0).wait()
    defer { try? server.close().wait() }
    let port = try XCTUnwrap(server.channel.localAddress?.port)
    var request: [String: Any] = ["profile": ["route": "grpc", "host": "127.0.0.1",
                                              "port": port, "useTLS": false]]
    XCTAssertThrowsError(try Discovery.fetch(request)) {
      XCTAssertEqual($0 as? TransportError, .authenticationRequired)
    }
    request["credentials"] = ["sharedSecret": "fixture-pass"]
    let result = try Discovery.fetch(request)
    XCTAssertEqual(result["files"] as? [String], ["fixture-model.ckpt", "flux_2_klein_4b_q8p.ckpt"])
    let rules = try XCTUnwrap(result["capabilities"] as? [String: Any])
    XCTAssertNotNil(rules["flux_2_klein_4b_q8p.ckpt"])
    XCTAssertNil(rules["fixture-model.ckpt"])
    XCTAssertEqual((result["transport"] as? [String: Bool])?["tls"], false)
    XCTAssertEqual(result["serverIdentifier"] as? String, "123")
    let policy = try XCTUnwrap(result["thresholds"] as? [String: Any])
    XCTAssertEqual(policy["community"] as? Int, 10000)
    XCTAssertEqual(policy["plus"] as? Int, 40000)
    XCTAssertEqual(policy["expiresAt"] as? Double, 2000000000)
  }

  func testTLSRequiresBooleanNotAnInteger() {
    let request: [String: Any] = ["profile": ["route": "grpc", "host": "127.0.0.1",
                                             "port": 1, "useTLS": 1]]
    XCTAssertThrowsError(try Discovery.fetch(request)) {
      XCTAssertEqual($0 as? TransportError, .invalidRequest)
    }
  }
}
