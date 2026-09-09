// swift-tools-version: 6.0
import PackageDescription

let package = Package(
  name: "WeeToddDrawThingsClient",
  platforms: [.macOS(.v14)],
  products: [.executable(name: "WeeToddDrawThings", targets: ["WeeToddDrawThings"])],
  dependencies: [
    .package(url: "https://github.com/drawthingsai/media-generation-kit.git",
             revision: "8868a9685d9c299816f43ef53efd455ffca437f0")
  ],
  targets: [
    .target(name: "DrawThingsTransport", dependencies: [
      .product(name: "MediaGenerationKit", package: "media-generation-kit")
    ]),
    .executableTarget(name: "WeeToddDrawThings", dependencies: ["DrawThingsTransport"]),
    .executableTarget(name: "WeeToddDrawThingsFixtureServer", dependencies: ["DrawThingsTransport"],
                      path: "Tests/FixtureServer"),
    .testTarget(name: "DrawThingsTransportTests", dependencies: ["DrawThingsTransport"])
  ],
  swiftLanguageModes: [.v5]
)
