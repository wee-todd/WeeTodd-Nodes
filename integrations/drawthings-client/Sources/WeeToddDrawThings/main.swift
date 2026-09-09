import Darwin
import DrawThingsTransport
import Foundation

// Upstream libraries write diagnostic text to stdout. Reserve a clean protocol descriptor.
let protocolOutput = FileHandle(fileDescriptor: dup(STDOUT_FILENO), closeOnDealloc: true)
dup2(STDERR_FILENO, STDOUT_FILENO)
var requestID = "invalid-request"
func emit(_ value: [String: Any]) throws {
  var record = value
  record["requestID"] = requestID
  var data = try JSONSerialization.data(withJSONObject: record, options: [.sortedKeys])
  data.append(10)
  try protocolOutput.write(contentsOf: data)
}

do {
  var data = Data()
  while data.count <= 1024 * 1024 {
    let chunk = try FileHandle.standardInput.read(upToCount: min(65536, 1024 * 1024 + 1 - data.count)) ?? Data()
    if chunk.isEmpty { break }
    data.append(chunk)
  }
  guard data.count <= 1024 * 1024,
    let request = try JSONSerialization.jsonObject(with: data) as? [String: Any],
    let id = request["requestID"] as? String, !id.isEmpty,
    CommandLine.arguments.count == 2 else { throw TransportError.invalidRequest }
  requestID = id
  let value: [String: Any]
  switch CommandLine.arguments[1] {
  case "estimate": value = try ComputeEstimate.evaluate(request)
  case "capabilities": value = try Discovery.fetch(request)
  case "generate": value = try Submission.run(request) { progress in
    try? emit(["type": "progress", "value": progress])
  }
  default: throw TransportError.unsupportedOperation
  }
  try emit(["type": "result", "value": value])
} catch {
  let code = (error as? TransportError)?.rawValue ?? "transport_failed"
  try? emit(["type": "error", "code": code])
  exit(1)
}
