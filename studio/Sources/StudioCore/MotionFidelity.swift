import Foundation

public struct MotionFidelitySettings: Codable, Equatable {
  public var enabled = false
  public var mode = "adaptive"
  public var strength = 0.5
  public var maxHold = 2
  public var sensitivity = 0.5
  public var seed = 42
  public var maxFrames = 345
  public var evaluations: Int?
  public init() {}
}
public struct MotionFidelityResult: Codable, Equatable {
  public var path: String
  public var sourcePath: String
  public var sourceIn: Double
  public var duration: Double
  public var outputIn: Double
  public var recipeID: String
  public var settings: MotionFidelitySettings
  public var sourceSHA256: String
  public var sha256: String
  public var report: String
  public var sourceSize: Int
  public var sourceModified: Double
  public var outputSize: Int
  public var outputModified: Double
  public var recipePath: String?
  public var recipeSHA256: String?
  public var recipeSize: Int?
  public var recipeModified: Double?
}
extension Clip {
  public var motionIsCurrent: Bool {
    guard let settings = motionFidelity, settings.enabled, let result = motionResult,
      result.settings == settings, result.sourcePath == sourcePath,
      result.sourceIn == sourceIn, result.duration == duration,
      result.recipeID == (motionRecipeID ?? "") else { return false }
    func matches(_ path: String, _ size: Int, _ modified: Double) -> Bool {
      guard let a = try? FileManager.default.attributesOfItem(atPath: path),
        let date = a[.modificationDate] as? Date, let bytes = a[.size] as? NSNumber
      else { return false }
      return bytes.intValue == size && abs(date.timeIntervalSince1970 - modified) < 0.001
    }
    if let path = result.recipePath, let size = result.recipeSize, let modified = result.recipeModified,
      !matches(path, size, modified) { return false }
    return matches(sourcePath, result.sourceSize, result.sourceModified)
      && matches(result.path, result.outputSize, result.outputModified)
  }
  public var playbackPath: String { motionIsCurrent ? motionResult!.path : sourcePath }
  public var playbackIn: Double { motionIsCurrent ? motionResult!.outputIn : sourceIn }
}
