import Foundation

/// Measured values only. RSS and the resettable MLX counter have different scopes.
public struct RenderStats: Codable, Equatable {
  public var elapsedSeconds: Double?
  public var samplingSeconds: Double?
  public var samplingScope: String?
  public var processPeakBytes: Double?
  public var mlxPeakBytes: Double?
  public var mlxPeakScope: String?

  public init?(result: [String: Any]) {
    func measured(_ value: Any?) -> Double? {
      guard let value = value as? NSNumber, String(cString: value.objCType) != "c" else { return nil }
      let number = value.doubleValue
      return number.isFinite && number >= 0 ? number : nil
    }
    let metadata = result["metadata"] as? [String: Any] ?? [:]
    let timings = metadata["stage_timings"] as? [String: Any] ?? [:]
    let phases = metadata["phase_memory"] as? [String: Any] ?? [:]
    elapsedSeconds = measured(result["seconds"]) ?? measured(metadata["total_seconds"])
    samplingSeconds = measured(result["sampling_seconds"]) ?? measured(timings["sampling_total_seconds"])
    if measured(result["sampling_seconds"]) != nil { samplingScope = "Transformer sampling" }
    else if samplingSeconds != nil { samplingScope = "Pre-decode pipeline" }
    processPeakBytes = measured(result["process_peak_rss_bytes"])
    if let peak = measured(phases["run_peak_bytes"]) {
      mlxPeakBytes = peak; mlxPeakScope = "Instrumented stages"
    } else if let peak = measured(metadata["mlx_peak_bytes"]) {
      mlxPeakBytes = peak; mlxPeakScope = "MLX generation"
    }
    if elapsedSeconds == nil && samplingSeconds == nil && processPeakBytes == nil && mlxPeakBytes == nil {
      return nil
    }
  }

  public static func duration(_ seconds: Double) -> String {
    guard seconds.isFinite, seconds >= 0, seconds < Double(Int.max / 2) else { return "—" }
    let whole = Int(seconds)
    if whole >= 3600 { return String(format: "%d:%02d:%02d", whole / 3600, whole / 60 % 60, whole % 60) }
    return String(format: "%d:%02d", whole / 60, whole % 60)
  }
}

public struct BridgeProgressEvent: Equatable, Sendable {
  public let message: String
  public let fraction: Double?
}

/// A pipe read is not a line or necessarily a complete UTF-8 character.
public struct BridgeProgressStream {
  private var line = Data()
  private var dropping = false
  public init() {}
  public mutating func append(_ data: Data) -> [BridgeProgressEvent] {
    var events: [BridgeProgressEvent] = []
    for byte in data {
      if byte == 10 {
        if !dropping,
          let object = try? JSONSerialization.jsonObject(with: line) as? [String: Any],
          object["event"] as? String == "progress",
          let message = object["message"] as? String, !message.isEmpty {
          let fraction = object["fraction"] as? Double
          if fraction == nil || (fraction!.isFinite && (0...1).contains(fraction!)) {
            events.append(BridgeProgressEvent(message: String(message.prefix(512)), fraction: fraction))
          }
        }
        line.removeAll(keepingCapacity: true); dropping = false
      } else if !dropping {
        if line.count < 65536 { line.append(byte) }
        else { line.removeAll(keepingCapacity: true); dropping = true }
      }
    }
    return events
  }
}
