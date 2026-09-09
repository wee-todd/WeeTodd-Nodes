import Foundation
import GRPC
import GRPCImageServiceModels
import NIO
import XCTest
@testable import DrawThingsTransport

private final class EchoFixture: ImageGenerationServiceProvider {
  var interceptors: ImageGenerationServiceServerInterceptorFactoryProtocol? { nil }
  func echo(request: EchoRequest, context: StatusOnlyCallContext) -> EventLoopFuture<EchoReply> {
    context.eventLoop.makeSucceededFuture(EchoReply.with {
      $0.sharedSecretMissing = request.sharedSecret != "fixture-pass"
      $0.files = ["fixture-model.ckpt", "flux_2_klein_4b_q8p.ckpt"]
      $0.serverIdentifier = 123
      $0.thresholds = ComputeUnitThreshold.with {
        $0.community = 10000; $0.plus = 40000; $0.expireAt = 2000000000
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
    context.eventLoop.makeFailedFuture(GRPCStatus(code: .unimplemented))
  }
}

final class DiscoveryTests: XCTestCase {
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
