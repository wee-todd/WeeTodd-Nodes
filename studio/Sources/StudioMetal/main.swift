import AVFoundation
import CoreImage
import CoreVideo
import Foundation
import Metal
import MetalFX

struct Failure: LocalizedError {
  var message: String
  var errorDescription: String? { message }
}
func fail(_ value: String) throws -> Never { throw Failure(message: value) }
func texture(
  _ device: MTLDevice, _ width: Int, _ height: Int, _ format: MTLPixelFormat,
  privateMode: Bool = false, usage: MTLTextureUsage = [.shaderRead, .shaderWrite, .renderTarget]
) throws -> MTLTexture {
  let d = MTLTextureDescriptor.texture2DDescriptor(
    pixelFormat: format, width: width, height: height, mipmapped: false)
  d.usage = usage
  d.storageMode = privateMode ? .private : .shared
  guard let t = device.makeTexture(descriptor: d) else {
    try fail("Cannot allocate MetalFX textures.")
  }
  return t
}
func guides(
  _ folder: String, _ index: Int, _ device: MTLDevice, _ width: Int, _ height: Int, _ channels: Int
) throws -> MTLTexture {
  let url = URL(fileURLWithPath: folder).appendingPathComponent(String(format: "%06d.f32", index))
  let data = try Data(contentsOf: url)
  guard data.count == width * height * channels * 4 else {
    try fail(
      "Guide \(url.lastPathComponent) has the wrong dimensions. Use tightly packed little-endian float32 at the processed clip resolution."
    )
  }
  let values = data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
  guard values.allSatisfy(\.isFinite) else { try fail("Guide contains non-finite values.") }
  let t = try texture(
    device, width, height, channels == 1 ? .r32Float : .rg32Float, usage: .shaderRead)
  data.withUnsafeBytes {
    t.replace(
      region: MTLRegionMake2D(0, 0, width, height), mipmapLevel: 0, withBytes: $0.baseAddress!,
      bytesPerRow: width * channels * 4)
  }
  return t
}
func process() async throws {
  let args = CommandLine.arguments
  guard let device = MTLCreateSystemDefaultDevice() else { try fail("No Metal device available.") }
  if args.count == 2 && args[1] == "capabilities" {
    var d: [String: Any] = [
      "device": device.name, "spatial": MTLFXSpatialScalerDescriptor.supportsDevice(device),
    ]
    if #available(macOS 26.0, *) {
      d["interpolation"] = MTLFXFrameInterpolatorDescriptor.supportsDevice(device)
    } else {
      d["interpolation"] = false
    }
    print(String(data: try JSONSerialization.data(withJSONObject: d), encoding: .utf8)!)
    return
  }
  guard args.count == 6, ["upscale", "interpolate"].contains(args[1]) else {
    try fail(
      "Usage: StudioMetal upscale input output width height | interpolate input output depth-folder motion-folder"
    )
  }
  let mode = args[1]
  let input = URL(fileURLWithPath: args[2])
  let output = URL(fileURLWithPath: args[3])
  guard !FileManager.default.fileExists(atPath: output.path) else {
    try fail("Output already exists.")
  }
  let asset = AVURLAsset(url: input)
  guard let track = try await asset.loadTracks(withMediaType: .video).first else {
    try fail("Input has no video.")
  }
  let size = try await track.load(.naturalSize)
  let rate = try await track.load(.nominalFrameRate)
  let w = Int(size.width)
  let h = Int(size.height)
  let ow = mode == "upscale" ? Int(args[4]) ?? 0 : w
  let oh = mode == "upscale" ? Int(args[5]) ?? 0 : h
  guard ow >= 64, oh >= 64, ow <= 8192, oh <= 8192, rate > 0 else {
    try fail("Invalid output dimensions or input frame rate.")
  }
  let reader = try AVAssetReader(asset: asset)
  let video = AVAssetReaderTrackOutput(
    track: track,
    outputSettings: [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA])
  video.alwaysCopiesSampleData = false
  reader.add(video)
  let writer = try AVAssetWriter(outputURL: output, fileType: .mp4)
  let writerInput = AVAssetWriterInput(
    mediaType: .video,
    outputSettings: [
      AVVideoCodecKey: AVVideoCodecType.h264, AVVideoWidthKey: ow, AVVideoHeightKey: oh,
      AVVideoCompressionPropertiesKey: [AVVideoAverageBitRateKey: max(2_000_000, ow * oh * 6)],
    ])
  writerInput.expectsMediaDataInRealTime = false
  let adaptor = AVAssetWriterInputPixelBufferAdaptor(
    assetWriterInput: writerInput,
    sourcePixelBufferAttributes: [
      kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
      kCVPixelBufferWidthKey as String: ow, kCVPixelBufferHeightKey as String: oh,
      kCVPixelBufferMetalCompatibilityKey as String: true,
    ])
  writer.add(writerInput)
  let context = CIContext(mtlDevice: device, options: [.cacheIntermediates: false])
  let space = CGColorSpaceCreateDeviceRGB()
  guard let queue = device.makeCommandQueue() else { try fail("No Metal command queue.") }
  var spatial: (any MTLFXSpatialScaler)?
  if mode == "upscale" {
    guard MTLFXSpatialScalerDescriptor.supportsDevice(device) else {
      try fail("MetalFX spatial upscaling is unsupported by this GPU.")
    }
    let d = MTLFXSpatialScalerDescriptor()
    d.inputWidth = w
    d.inputHeight = h
    d.outputWidth = ow
    d.outputHeight = oh
    d.colorTextureFormat = .bgra8Unorm
    d.outputTextureFormat = .bgra8Unorm
    d.colorProcessingMode = .perceptual
    guard let scaler = d.makeSpatialScaler(device: device) else {
      try fail("MetalFX rejected the scaling configuration.")
    }
    spatial = scaler
  }
  var interpolate: ((any MTLTexture, any MTLTexture, Int) throws -> any MTLTexture)?
  if mode == "interpolate" {
    if #available(macOS 26.0, *) {
      guard MTLFXFrameInterpolatorDescriptor.supportsDevice(device) else {
        try fail("MetalFX frame interpolation is unsupported by this GPU.")
      }
      let d = MTLFXFrameInterpolatorDescriptor()
      d.inputWidth = w
      d.inputHeight = h
      d.outputWidth = w
      d.outputHeight = h
      d.colorTextureFormat = .bgra8Unorm
      d.outputTextureFormat = .bgra8Unorm
      d.depthTextureFormat = .r32Float
      d.motionTextureFormat = .rg32Float
      guard let fi = d.makeFrameInterpolator(device: device) else {
        try fail("MetalFX could not create the frame interpolator.")
      }
      let cameraURL = URL(fileURLWithPath: args[4]).appendingPathComponent("camera.json")
      guard
        let camera = try JSONSerialization.jsonObject(with: Data(contentsOf: cameraURL))
          as? [String: Any],
        let near = camera["nearPlane"] as? Double, let far = camera["farPlane"] as? Double,
        let fov = camera["fieldOfView"] as? Double, let reversed = camera["depthReversed"] as? Bool,
        near.isFinite, far.isFinite, fov.isFinite, near > 0, far > near, fov > 0, fov < 180
      else {
        try fail(
          "Depth folder needs camera.json with valid nearPlane, farPlane, fieldOfView (degrees), and depthReversed."
        )
      }
      fi.motionVectorScaleX = 1
      fi.motionVectorScaleY = 1
      fi.deltaTime = 1 / rate
      fi.nearPlane = Float(near)
      fi.farPlane = Float(far)
      fi.fieldOfView = Float(fov)
      fi.aspectRatio = Float(w) / Float(h)
      fi.isDepthReversed = reversed
      let out = try texture(device, w, h, .bgra8Unorm, privateMode: true)
      interpolate = { previous, current, index in
        fi.prevColorTexture = previous
        fi.colorTexture = current
        fi.depthTexture = try guides(args[4], index, device, w, h, 1)
        fi.motionTexture = try guides(args[5], index, device, w, h, 2)
        fi.outputTexture = out
        fi.shouldResetHistory = index == 1
        guard let command = queue.makeCommandBuffer() else {
          try fail("Metal command allocation failed.")
        }
        fi.encode(commandBuffer: command)
        command.commit()
        command.waitUntilCompleted()
        if let error = command.error { throw error }
        return out
      }
    } else {
      try fail("MetalFX frame interpolation requires macOS 26 or later.")
    }
  }
  guard writer.startWriting(), reader.startReading() else {
    try fail("Cannot start media reader/writer.")
  }
  writer.startSession(atSourceTime: .zero)
  let outputRate = Double(rate) * (mode == "interpolate" ? 2 : 1)
  func append(_ image: CIImage, _ index: Int) throws {
    while !writerInput.isReadyForMoreMediaData {
      if writer.status == .failed { throw writer.error! }
      Thread.sleep(forTimeInterval: 0.002)
    }
    var pb: CVPixelBuffer?
    guard let pool = adaptor.pixelBufferPool,
      CVPixelBufferPoolCreatePixelBuffer(nil, pool, &pb) == kCVReturnSuccess, let pb
    else { try fail("Cannot allocate output pixel buffer.") }
    context.render(
      image, to: pb, bounds: CGRect(x: 0, y: 0, width: ow, height: oh), colorSpace: space)
    guard
      adaptor.append(
        pb,
        withPresentationTime: CMTime(seconds: Double(index) / outputRate, preferredTimescale: 60000)
      )
    else { try fail(writer.error?.localizedDescription ?? "Cannot write output frame.") }
  }
  var index = 0
  var outputIndex = 0
  var previous: (any MTLTexture)?
  while let sample = video.copyNextSampleBuffer() {
    try autoreleasepool {
      guard let pb = CMSampleBufferGetImageBuffer(sample) else {
        try fail("Missing input pixel buffer.")
      }
      let current = try texture(device, w, h, .bgra8Unorm)
      context.render(
        CIImage(cvPixelBuffer: pb), to: current, commandBuffer: nil,
        bounds: CGRect(x: 0, y: 0, width: w, height: h), colorSpace: space)
      if let spatial {
        let out = try texture(device, ow, oh, .bgra8Unorm, privateMode: true)
        spatial.colorTexture = current
        spatial.outputTexture = out
        guard let cmd = queue.makeCommandBuffer() else {
          try fail("Cannot allocate Metal commands.")
        }
        spatial.encode(commandBuffer: cmd)
        cmd.commit()
        cmd.waitUntilCompleted()
        if let error = cmd.error { throw error }
        guard let image = CIImage(mtlTexture: out, options: [.colorSpace: space]) else {
          try fail("Cannot read MetalFX output.")
        }
        try append(image, outputIndex)
        outputIndex += 1
      } else {
        if let previous, let interpolate {
          let middle = try interpolate(previous, current, index)
          guard let image = CIImage(mtlTexture: middle, options: [.colorSpace: space]) else {
            try fail("Cannot read interpolated frame.")
          }
          try append(image, outputIndex)
          outputIndex += 1
        }
        try append(CIImage(cvPixelBuffer: pb), outputIndex)
        outputIndex += 1
        previous = current
      }
      index += 1
    }
  }
  if reader.status == .failed { throw reader.error! }
  writerInput.markAsFinished()
  await writer.finishWriting()
  guard writer.status == .completed else {
    try fail(writer.error?.localizedDescription ?? "Movie writing failed.")
  }
  print("Processed \(index) input frames → \(outputIndex) output frames with MetalFX \(mode)")
}
do { try await process() } catch {
  fputs("\(error.localizedDescription)\n", stderr)
  exit(1)
}
