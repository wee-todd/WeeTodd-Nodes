// swift-tools-version: 6.0
import PackageDescription

let package = Package(
  name: "WeeToddDrawThingsClient",
  platforms: [.macOS(.v14)],
  products: [.executable(name: "WeeToddDrawThings", targets: ["WeeToddDrawThings"])],
  dependencies: [
    .package(url: "https://github.com/drawthingsai/draw-things-community.git",
             revision: "08e798b5ad59c3db78b2be53f0ed60b071653302")
  ],
  targets: [
    .target(name: "DrawThingsTransport", dependencies: [
      .product(name: "_MediaGenerationKit", package: "draw-things-community")
    ]),
    .executableTarget(name: "WeeToddDrawThings", dependencies: ["DrawThingsTransport"]),
    .executableTarget(name: "WeeToddDrawThingsFixtureServer", dependencies: ["DrawThingsTransport"],
                      path: "Tests/FixtureServer"),
    .testTarget(name: "DrawThingsTransportTests", dependencies: ["DrawThingsTransport"])
  ],
  swiftLanguageModes: [.v5]
)
