import Foundation

public enum DrawThingsTaskFilter {
  public static func modelIDs(in capabilities: [String: Any], task: String) -> Set<String> {
    let roles: [String]
    switch task {
    case "t2v": roles = []
    case "i2v": roles = ["first"]
    case "fflf": roles = ["first", "last"]
    default: return []
    }
    return Set(capabilities.compactMap { id, value in
      guard let model = value as? [String: Any],
        let operations = model["operations"] as? [String: Any],
        let video = operations["video"] as? [String: Any],
        let combinations = video["inputRoleCombinations"] as? [[String]],
        combinations.contains(where: { $0.sorted() == roles }) else { return nil }
      return id
    })
  }
}
