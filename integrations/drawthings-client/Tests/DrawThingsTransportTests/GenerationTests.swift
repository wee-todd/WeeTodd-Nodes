import Foundation
import GRPC
import GRPCImageServiceModels
import NIO
import NNC
import XCTest
@testable import DrawThingsTransport

private final class GenerationFixture: ImageGenerationServiceProvider {
  var interceptors: ImageGenerationServiceServerInterceptorFactoryProtocol? { nil }
  var calls = 0
  let failsAfterFrames: Bool
  let video: Bool
  let omitAudio: Bool
  init(failsAfterFrames: Bool = false, video: Bool = false, omitAudio: Bool = false) {
    self.failsAfterFrames = failsAfterFrames; self.video = video; self.omitAudio = omitAudio
  }
  func echo(request: EchoRequest, context: StatusOnlyCallContext) -> EventLoopFuture<EchoReply> {
    context.eventLoop.makeSucceededFuture(EchoReply.with {
      $0.files = ["flux_2_klein_4b_q8p.ckpt", "ltx_2.3_22b_distilled_q6p.ckpt"]; $0.serverIdentifier = 123
    })
  }
  func generateImage(request: ImageGenerationRequest,
                     context: StreamingResponseCallContext<ImageGenerationResponse>) -> EventLoopFuture<GRPCStatus> {
    calls += 1
    var tensor = Tensor<Float>(.CPU, .NHWC(1, 64, 64, 3))
    for y in 0..<64 { for x in 0..<64 { for c in 0..<3 { tensor[0,y,x,c] = 0.25 } } }
    var audio = Tensor<Float>(.CPU, .NC(2, 18000))
    for channel in 0..<2 { for sample in 0..<18000 {
      audio[channel, sample] = sin(Float(sample) * 440 * 2 * .pi / 48000) * 0.1
    } }
    return context.sendResponse(ImageGenerationResponse.with {
      $0.generatedImages = Array(repeating: tensor.data(using: [.zip, .fpzip]), count: video ? 9 : 1)
      if video && !omitAudio { $0.generatedAudio = [audio.data(using: [.zip, .fpzip])] }
    }).map { self.failsAfterFrames ? GRPCStatus(code: .unavailable) : GRPCStatus.ok }
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

final class GenerationTests: XCTestCase {
  func testActualGRPCVideoPreservesSeparateAudioAndRefusesSilentCompletion() throws {
    for omitAudio in [false, true] {
      let fixture = GenerationFixture(video: true, omitAudio: omitAudio)
      let group = MultiThreadedEventLoopGroup(numberOfThreads: 1)
      defer { try? group.syncShutdownGracefully() }
      let server = try Server.insecure(group: group).withServiceProviders([fixture])
        .bind(host: "127.0.0.1", port: 0).wait()
      defer { try? server.close().wait() }
      let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
      defer { try? FileManager.default.removeItem(at: root) }
      let request: [String: Any] = [
        "requestID": "av-fixture", "operation": "video", "modelID": "ltx_2.3_22b_distilled_q6p.ckpt",
        "prompt": "fixture", "inputs": [], "loras": [], "billingPolicy": "freeOnly",
        "configuration": ["width": 64, "height": 64, "steps": 8, "seed": 42, "numFrames": 9, "fps": 24],
        "outputDirectory": root.path,
        "profile": ["route": "grpc", "host": "127.0.0.1", "port": server.channel.localAddress!.port!, "useTLS": false]
      ]
      if omitAudio {
        XCTAssertThrowsError(try Generation.run(request, authorize: { _ in nil }, progress: { _ in }))
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.appendingPathComponent("manifest.json").path))
      } else {
        _ = try Generation.run(request, authorize: { _ in nil }, progress: { _ in })
        let manifest = try JSONSerialization.jsonObject(with: Data(contentsOf: root.appendingPathComponent("manifest.json"))) as! [String: Any]
        XCTAssertEqual(manifest["frameCount"] as? Int, 9)
        XCTAssertEqual(manifest["sampleCount"] as? Int, 18000)
        XCTAssertEqual(manifest["sampleRate"] as? Int, 48000)
        XCTAssertTrue(FileManager.default.fileExists(atPath: root.appendingPathComponent("audio.wav").path))
      }
      XCTAssertEqual(fixture.calls, 1)
    }
  }
  func testActualGRPCStreamsAnImageAndDoesNotPublishFailedRPC() throws {
    for failure in [false, true] {
      let fixture = GenerationFixture(failsAfterFrames: failure)
      let group = MultiThreadedEventLoopGroup(numberOfThreads: 1)
      defer { try? group.syncShutdownGracefully() }
      let server = try Server.insecure(group: group).withServiceProviders([fixture])
        .bind(host: "127.0.0.1", port: 0).wait()
      defer { try? server.close().wait() }
      let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
      defer { try? FileManager.default.removeItem(at: root) }
      let request: [String: Any] = [
        "requestID": "generation-fixture", "operation": "image", "modelID": "flux_2_klein_4b_q8p.ckpt",
        "prompt": "fixture", "negativePrompt": "", "inputs": [], "loras": [],
        "billingPolicy": "freeOnly", "configuration": ["width": 64, "height": 64, "steps": 4, "seed": 42],
        "outputDirectory": root.path,
        "profile": ["route": "grpc", "host": "127.0.0.1", "port": server.channel.localAddress!.port!, "useTLS": false]
      ]
      if failure {
        XCTAssertThrowsError(try Generation.run(request, authorize: { _ in nil }, progress: { _ in })) {
          XCTAssertEqual($0 as? TransportError, .submissionUncertain)
        }
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.appendingPathComponent("manifest.json").path))
      } else {
        let result = try Generation.run(request, authorize: { _ in nil }, progress: { _ in })
        XCTAssertEqual(result["manifestPath"] as? String, root.appendingPathComponent("manifest.json").path)
      }
      XCTAssertEqual(fixture.calls, 1)
    }
  }

  func testLocalConnectionFailureNeverReservesCloudQuota() throws {
    var authorizations = 0
    let request: [String: Any] = [
      "requestID": "blocked", "operation": "image", "modelID": "flux_2_klein_4b_q8p.ckpt",
      "prompt": "fixture", "configuration": ["width": 64, "height": 64, "steps": 4, "seed": 42],
      "outputDirectory": "/unused", "profile": ["route": "grpc", "host": "127.0.0.1", "port": 1, "useTLS": false]
    ]
    XCTAssertThrowsError(try Generation.run(request, authorize: { _ in authorizations += 1; return nil }, progress: { _ in }))
    XCTAssertEqual(authorizations, 0)
  }
}
