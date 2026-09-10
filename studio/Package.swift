// swift-tools-version: 6.0
import PackageDescription

let package = Package(
  name: "WeeToddStudio",
  platforms: [.macOS(.v14)],
  products: [
    .executable(name: "WeeToddStudio", targets: ["WeeToddStudio"]),
    .executable(name: "StudioMetal", targets: ["StudioMetal"]),
    .executable(name: "WeeToddCLI", targets: ["WeeToddCLI"]),
    .library(name: "StudioCore", targets: ["StudioCore"]),
  ],
  targets: [
    .target(name: "StudioCore"),
    .executableTarget(name: "WeeToddStudio", dependencies: ["StudioCore"]),
    .executableTarget(name: "StudioMetal"),
    .executableTarget(name: "WeeToddCLI"),
    .testTarget(name: "StudioCoreTests", dependencies: ["StudioCore"]),
    .testTarget(name: "StudioAppTests", dependencies: ["WeeToddStudio"]),
  ],
  swiftLanguageModes: [.v5]
)
