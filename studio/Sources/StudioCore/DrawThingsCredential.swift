import Foundation
import Security

public enum DrawThingsCredential {
  private static func query(_ reference: String) -> [String: Any] {
    [kSecClass as String: kSecClassGenericPassword,
     kSecAttrService as String: "org.weetodd.studio.drawthings",
     kSecAttrAccount as String: reference]
  }
  private static func check(_ status: OSStatus) throws {
    guard status == errSecSuccess else {
      throw StudioError.invalid("macOS Keychain could not access the Draw Things credential (\(status)).")
    }
  }
  public static func read(_ reference: String) throws -> String? {
    var request = query(reference)
    request[kSecMatchLimit as String] = kSecMatchLimitOne
    request[kSecReturnData as String] = true
    var result: CFTypeRef?
    let status = SecItemCopyMatching(request as CFDictionary, &result)
    if status == errSecItemNotFound { return nil }
    try check(status)
    guard let data = result as? Data, let value = String(data: data, encoding: .utf8) else {
      throw StudioError.invalid("The saved Draw Things credential cannot be decoded.")
    }
    return value
  }
  public static func save(_ value: String, reference: String) throws {
    guard !reference.isEmpty, !value.isEmpty,
      !value.unicodeScalars.contains(where: CharacterSet.controlCharacters.contains) else {
      throw StudioError.invalid("Enter a nonempty credential without control characters.")
    }
    let data = Data(value.utf8)
    let status = SecItemUpdate(query(reference) as CFDictionary, [kSecValueData as String: data] as CFDictionary)
    if status == errSecItemNotFound {
      var item = query(reference)
      item[kSecValueData as String] = data
      item[kSecAttrAccessible as String] = kSecAttrAccessibleAfterFirstUnlockThisDeviceOnly
      try check(SecItemAdd(item as CFDictionary, nil))
    } else { try check(status) }
  }
  public static func remove(_ reference: String) throws {
    let status = SecItemDelete(query(reference) as CFDictionary)
    if status != errSecItemNotFound { try check(status) }
  }
}
