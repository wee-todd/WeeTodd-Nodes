import Foundation

/// Keychain may wait for an OS permission dialog; keep that wait off the UI executor.
public enum BackgroundCredential {
  public static func read(using lookup: @escaping @Sendable () throws -> String?) async throws -> String? {
    try await withCheckedThrowingContinuation { continuation in
      DispatchQueue.global(qos: .userInitiated).async {
        continuation.resume(with: Result { try lookup() })
      }
    }
  }
}
