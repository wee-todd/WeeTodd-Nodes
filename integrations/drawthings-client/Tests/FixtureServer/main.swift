// Development-only protocol fixture. Never packaged with Studio or used for quality claims.
import Foundation
import DataModels
import GRPC
import GRPCImageServiceModels
import NIO
import NNC

final class FixtureService: ImageGenerationServiceProvider {
  var interceptors: ImageGenerationServiceServerInterceptorFactoryProtocol? { nil }
  func echo(request: EchoRequest, context: StatusOnlyCallContext) -> EventLoopFuture<EchoReply> {
    context.eventLoop.makeSucceededFuture(EchoReply.with {
      $0.files = ["flux_2_klein_4b_q8p.ckpt", "ltx_2.3_22b_distilled_q6p.ckpt"]
      $0.serverIdentifier = 12345
    })
  }
  func generateImage(request: ImageGenerationRequest,
                     context: StreamingResponseCallContext<ImageGenerationResponse>) -> EventLoopFuture<GRPCStatus> {
    let config = GenerationConfiguration.from(data: request.configuration)
    let width = Int(config.startWidth) * 64, height = Int(config.startHeight) * 64
    guard width <= 1024 && height <= 1024 else {
      return context.eventLoop.makeSucceededFuture(GRPCStatus(code: .resourceExhausted))
    }
    let video = config.model == "ltx_2.3_22b_distilled_q6p.ckpt"
    let frames = video ? Int(config.numFrames) : 1
    guard frames <= 121 && config.fpsId > 0 else {
      return context.eventLoop.makeSucceededFuture(GRPCStatus(code: .resourceExhausted))
    }
    var tensor = Tensor<Float>(.CPU, .NHWC(1, height, width, 3))
    for y in 0..<height { for x in 0..<width {
      tensor[0,y,x,0] = Float(x)*2/Float(width) - 1
      tensor[0,y,x,1] = Float(y)*2/Float(height) - 1
      tensor[0,y,x,2] = 0.1
    } }
    let samples = video ? frames * 48000 / Int(config.fpsId) : 1
    var tone = Tensor<Float>(.CPU, .NC(2, samples))
    for c in 0..<2 { for i in 0..<samples {
      tone[c,i] = sin(Float(i) * 440 * 2 * .pi / 48000) * 0.1
    } }
    return context.sendResponse(ImageGenerationResponse.with {
      $0.generatedImages = Array(repeating: tensor.data(using: [.zip, .fpzip]), count: frames)
      if video { $0.generatedAudio = [tone.data(using: [.zip, .fpzip])] }
    }).map { GRPCStatus.ok }
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
let group = MultiThreadedEventLoopGroup(numberOfThreads: 1)
let server = try Server.insecure(group: group).withServiceProviders([FixtureService()])
  .bind(host: "127.0.0.1", port: 0).wait()
print("FIXTURE_PORT=\(server.channel.localAddress!.port!)")
fflush(stdout)
try server.onClose.wait()
try group.syncShutdownGracefully()
