import CoreGraphics
import CryptoKit
import DataModels
import Foundation
import ImageIO
import NNC

enum Conditioning {
  static func inputs(_ request: [String: Any]) throws -> [[String: Any]] {
    guard let inputs = request["inputs"] as? [[String: Any]] ?? (request["inputs"] == nil ? [] : nil),
      inputs.count <= 1 else { throw TransportError.unsupportedConditioning }
    for input in inputs {
      guard request["operation"] as? String == "video",
        Set(input.keys) == ["role", "path", "sha256", "frameIndex", "strength"],
        input["role"] as? String == "first", let path = input["path"] as? String, path.hasPrefix("/"),
        let digest = input["sha256"] as? String, digest.count == 64,
        digest.allSatisfy({ "0123456789abcdef".contains($0) }),
        try Configuration.number(input["frameIndex"], min: 0, max: 0, integer: true) == 0,
        try Configuration.number(input["strength"], min: 1, max: 1) == 1 else {
        throw TransportError.unsupportedConditioning
      }
    }
    return inputs
  }

  static func loras(_ request: [String: Any]) throws -> [LoRA] {
    guard let values = request["loras"] as? [[String: Any]] ?? (request["loras"] == nil ? [] : nil),
      values.count <= 16 else { throw TransportError.unsupportedConditioning }
    var seen = Set<String>()
    return try values.map { value in
      guard Set(value.keys) == ["modelID", "weight"], let id = value["modelID"] as? String,
        !id.isEmpty, seen.insert(id).inserted else { throw TransportError.unsupportedConditioning }
      return LoRA(file: id, weight: Float(try Configuration.number(value["weight"], min: 0, max: 2)), mode: .all)
    }
  }

  static func validateAvailability(_ request: [String: Any], catalog: [String: Any]) throws {
    let loras = try loras(request)
    let entries = catalog["loras"] as? [[String: Any]] ?? []
    let files = catalog["files"] as? [String] ?? []
    for lora in loras {
      guard let id = lora.file, files.contains(id), let model = request["modelID"] as? String,
        entries.contains(where: { $0["id"] as? String == id &&
          ($0["compatibleModelIDs"] as? [String] ?? []).contains(model) }) else {
        throw TransportError.unsupportedConditioning
      }
    }
  }

  static func image(_ request: [String: Any], width: Int, height: Int) throws -> Data? {
    guard let input = try inputs(request).first else { return nil }
    let url = URL(fileURLWithPath: input["path"] as! String)
    let attributes = try FileManager.default.attributesOfItem(atPath: url.path)
    guard let size = attributes[.size] as? NSNumber, size.intValue > 0, size.intValue <= 64 * 1024 * 1024 else {
      throw TransportError.invalidMedia
    }
    let data = try Data(contentsOf: url)
    let digest = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    guard digest == input["sha256"] as? String,
      let source = CGImageSourceCreateWithData(data as CFData, nil),
      let image = CGImageSourceCreateThumbnailAtIndex(source, 0, [
        kCGImageSourceCreateThumbnailFromImageAlways: true,
        kCGImageSourceCreateThumbnailWithTransform: true,
        kCGImageSourceThumbnailMaxPixelSize: 4096
      ] as CFDictionary) else { throw TransportError.invalidMedia }
    var pixels = [UInt8](repeating: 0, count: width * height * 4)
    try pixels.withUnsafeMutableBytes { storage in
      guard let context = CGContext(data: storage.baseAddress, width: width, height: height,
        bitsPerComponent: 8, bytesPerRow: width * 4, space: CGColorSpaceCreateDeviceRGB(),
        bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { throw TransportError.invalidMedia }
      context.setFillColor(CGColor(gray: 0, alpha: 1)); context.fill(CGRect(x: 0, y: 0, width: width, height: height))
      let scale = max(Double(width) / Double(image.width), Double(height) / Double(image.height))
      let w = Double(image.width) * scale, h = Double(image.height) * scale
      context.interpolationQuality = .high
      context.draw(image, in: CGRect(x: (Double(width)-w)/2, y: (Double(height)-h)/2, width: w, height: h))
    }
    var tensor = Tensor<Float>(.CPU, .NHWC(1, height, width, 3))
    for y in 0..<height { for x in 0..<width { for c in 0..<3 {
      tensor[0,y,x,c] = Float(pixels[(y * width + x) * 4 + c]) / 127.5 - 1
    } } }
    return tensor.data(using: [.zip, .fpzip])
  }
}
