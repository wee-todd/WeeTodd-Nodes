import Foundation
import Security

/// Credentials for explicit model downloads. Secrets stay out of Studio's persisted settings.
public enum ModelDownloadToken {
  private static let service = "org.weetodd.studio.model-download"
  private static let account = "huggingface"

  private static var query: [String: Any] {
    [
      kSecClass as String: kSecClassGenericPassword,
      kSecAttrService as String: service, kSecAttrAccount as String: account,
    ]
  }

  public static func normalized(_ token: String) throws -> String {
    let value = token.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !value.isEmpty,
      !value.unicodeScalars.contains(where: {
        CharacterSet.whitespacesAndNewlines.contains($0)
          || CharacterSet.controlCharacters.contains($0)
      })
    else { throw StudioError.invalid("Enter a single Hugging Face read token.") }
    return value
  }

  public static func environment(_ original: [String: String], savedToken: String?) throws
    -> [String: String]
  {
    guard let savedToken else { return original }
    var result = original
    result["HF_TOKEN"] = try normalized(savedToken)
    return result
  }

  public static func isConfigured() throws -> Bool {
    var request = query
    request[kSecMatchLimit as String] = kSecMatchLimitOne
    request[kSecReturnAttributes as String] = true
    let status = SecItemCopyMatching(request as CFDictionary, nil)
    if status == errSecItemNotFound { return false }
    try check(status)
    return true
  }

  public static func read() throws -> String? {
    var request = query
    request[kSecMatchLimit as String] = kSecMatchLimitOne
    request[kSecReturnData as String] = true
    var result: CFTypeRef?
    let status = SecItemCopyMatching(request as CFDictionary, &result)
    if status == errSecItemNotFound { return nil }
    try check(status)
    guard let data = result as? Data, let token = String(data: data, encoding: .utf8) else {
      throw StudioError.invalid(
        "The saved model token could not be read. Remove it and save a new token.")
    }
    return try normalized(token)
  }

  public static func save(_ token: String) throws {
    let bytes = Data(try normalized(token).utf8)
    let status = SecItemUpdate(
      query as CFDictionary, [kSecValueData as String: bytes] as CFDictionary)
    if status == errSecItemNotFound {
      var item = query
      item[kSecValueData as String] = bytes
      item[kSecAttrLabel as String] = "WeeTodd Studio model downloads"
      item[kSecAttrAccessible as String] = kSecAttrAccessibleAfterFirstUnlockThisDeviceOnly
      try check(SecItemAdd(item as CFDictionary, nil))
    } else {
      try check(status)
    }
  }

  public static func remove() throws {
    let status = SecItemDelete(query as CFDictionary)
    if status != errSecItemNotFound { try check(status) }
  }

  private static func check(_ status: OSStatus) throws {
    guard status == errSecSuccess else {
      throw StudioError.invalid(
        "macOS Keychain could not complete the model-token operation (\(status)). Unlock your Keychain or try again."
      )
    }
  }
}
