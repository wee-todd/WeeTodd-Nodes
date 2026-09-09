import XCTest
@testable import StudioCore

final class DrawThingsImageTests: XCTestCase {
  func testDestinationUsesCapturedClipAndRejectsDifferentProject() throws {
    var project = StudioProject()
    let first = Clip(name: "First"), second = Clip(name: "Second")
    project.clips = [first, second]
    let destination = ImageAssetDestination(scope: .clip, projectID: project.id, owner: first.id)
    let asset = try destination.asset(name: "Generated", path: "/fixture/image.png", in: project)
    XCTAssertEqual(asset.owner, first.id)
    XCTAssertEqual(asset.scope, .clip)
    XCTAssertEqual(project.clips, [first, second])
    XCTAssertThrowsError(try destination.asset(name: "Generated", path: "/fixture/image.png", in: StudioProject()))
    project.clips.removeFirst()
    XCTAssertThrowsError(try destination.asset(name: "Generated", path: "/fixture/image.png", in: project))
  }
  func testGlobalDestinationSurvivesProjectChange() throws {
    let destination = ImageAssetDestination(scope: .global, projectID: StudioProject().id)
    let asset = try destination.asset(name: "Generated", path: "/fixture/image.png", in: StudioProject())
    XCTAssertEqual(asset.scope, .global)
    XCTAssertNil(asset.owner)
  }
  func testOldConnectionRequiresExplicitSelfHostedSetting() throws {
    let json = #"{"id":"local","name":"Local","route":"grpc","host":"localhost","port":7859,"useTLS":false}"#
    let profile = try JSONDecoder().decode(DrawThingsConnection.self, from: Data(json.utf8))
    XCTAssertFalse(profile.selfHostedConfirmed ?? false)
    XCTAssertNil(profile.credentialRef)
  }
}
