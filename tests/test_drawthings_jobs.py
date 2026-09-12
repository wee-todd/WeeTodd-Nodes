import copy
import importlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
studio_job = importlib.import_module("studio_job")
jobs_module = importlib.import_module("wee_todd_remote.jobs")
validate_job_dependencies = jobs_module.validate_job_dependencies


def remote_request(operation="image", request_id="request-1"):
    configuration = {"width": 512, "height": 512, "steps": 4, "seed": 1}
    if operation == "video":
        configuration.update(numFrames=9, fps=24)
    return {
        "schema": "weetodd-drawthings-request-v1", "requestID": request_id,
        "operation": operation, "profileID": "portable-profile",
        "modelID": "DISCOVER_MODEL_ID", "prompt": "fixture", "negativePrompt": "",
        "configuration": configuration, "inputs": [], "loras": [],
        "billingPolicy": "freeOnly",
    }


def connection():
    return {"id": "portable-profile", "name": "Portable", "route": "dtCloud",
            "host": "", "port": 443, "tls": True, "credentialRef": "env:DT_TOKEN"}


def test_image_workspace_export_preserves_ordered_linked_inputs(tmp_path):
    import hashlib

    canonical = remote_request()
    canonical["configuration"].update(strength=0.65, sampler=18, guidanceScale=1)
    canonical["loras"] = [{"modelID": "style", "weight": 0.4}]
    for index, role in enumerate(("canvas", "moodboard", "moodboard")):
        image = tmp_path / f"input-{index}.png"
        image.write_bytes(f"fixture {index}".encode())
        canonical["inputs"].append({
            "role": role, "path": str(image), "fit": "fit",
            "sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
            "strength": 1 if index == 0 else index * 0.3,
        })
    exported = tmp_path / "images.weetodd-job.json"
    studio_job.export_job({
        "project": {"name": "Images", "clips": [], "assets": [], "audio": [],
                    "titles": [], "settings": {}},
        "runtime": {}, "generateIDs": [], "drawThingsImageJobs": [{
            "id": "image", "kind": "image", "request": canonical,
            "connection": connection(), "dependsOn": [],
        }],
    }, exported)
    job = json.loads(exported.read_text())
    assert job["format"] == "weetodd-studio-job-v3"
    assert job["remoteJobs"][0]["request"] == canonical
    jobs_without_manifest = {k: v for k, v in job.items() if k != "manifestSHA256"}
    assert job["manifestSHA256"] == studio_job.digest(jobs_without_manifest)


def image_job(tmp_path, *, source=None):
    project = {"name": "Images", "clips": [], "assets": [], "audio": [],
               "titles": [], "settings": {}}
    job = {
        "format": "weetodd-studio-job-v3", "scope": "movie", "project": project,
        "globalAssets": ([{"id": "source", "path": str(source)}] if source else []),
        "runtime": {}, "recipes": {},
        "remoteJobs": [{"id": "image", "kind": "image", "request": remote_request(),
                        "connection": connection(), "dependsOn": []}],
        "execution": {"parallelGenerations": 1,
                      "rendererSHA256": studio_job.renderer_fingerprint()},
    }
    job["manifestSHA256"] = studio_job.digest(job)
    return job


class FakeAdapter:
    def __init__(self, calls, *, fail=False):
        self.calls = calls
        self.fail = fail

    def prepare(self, request):
        self.calls.append(("prepare", request))
        return {"eligibility": "allowed"}

    def generate(self, request, *, output_directory, cancelled):
        self.calls.append(("generate", request))
        if self.fail:
            raise RuntimeError("transport ended after submission")
        output_directory.mkdir(parents=True)
        image = output_directory / "image.png"
        image.write_bytes(b"tiny fixture image")
        yield {"type": "result", "value": {"media": {"imagePaths": [str(image)]}}}


def test_preflight_counts_remote_image_generations(tmp_path, monkeypatch):
    import studio_drawthings

    calls = []
    monkeypatch.setattr(studio_drawthings, "adapter_for", lambda _: FakeAdapter(calls))
    report = studio_job.preflight(image_job(tmp_path), tmp_path / "preflight")
    assert report["clips"] == 0
    assert report["generations"] == 1
    assert report["remoteGenerations"] == 1
    assert calls[0][0] == "prepare"


def test_dependency_cycle_fails_before_generation():
    with pytest.raises(ValueError, match="cycle"):
        validate_job_dependencies([
            {"id": "a", "dependsOn": ["b"]}, {"id": "b", "dependsOn": ["a"]}
        ])


def test_dependencies_are_stable_and_missing_ids_fail():
    jobs = [{"id": "video", "dependsOn": ["image"]}, {"id": "image", "dependsOn": []}]
    assert [job["id"] for job in validate_job_dependencies(jobs)] == ["image", "video"]
    with pytest.raises(ValueError, match="missing dependency"):
        validate_job_dependencies([{"id": "video", "dependsOn": ["absent"]}])


def test_v3_export_contains_only_profile_and_credential_references(tmp_path):
    project = {"name": "Images", "clips": [], "assets": [], "audio": [],
               "titles": [], "settings": {}}
    target = tmp_path / "image.json"
    studio_job.export_job({
        "project": project, "runtime": {}, "generateIDs": [],
        "drawThingsImageJobs": [{"id": "image", "kind": "image", "request": remote_request(),
                                  "connection": connection(), "dependsOn": []}],
    }, target)
    job = json.loads(target.read_text())
    assert job["format"] == "weetodd-studio-job-v3"
    serialized = target.read_text().lower()
    assert "dt_token" in serialized
    assert "actual-secret" not in serialized
    assert job["remoteJobs"][0]["connection"]["credentialRef"] == "env:DT_TOKEN"


def test_secret_bearing_remote_export_is_rejected(tmp_path):
    project = {"name": "Images", "clips": [], "assets": [], "audio": [],
               "titles": [], "settings": {}}
    bad = connection() | {"apiKey": "actual-secret"}
    with pytest.raises(ValueError, match="secret"):
        studio_job.export_job({"project": project, "runtime": {}, "generateIDs": [],
            "drawThingsImageJobs": [{"id": "image", "kind": "image", "request": remote_request(),
                                      "connection": bad}]}, tmp_path / "bad.json")


def test_secret_scan_covers_entire_v3_candidate(tmp_path):
    project = {"name": "Images", "clips": [], "assets": [{"token": "actual-secret"}],
               "audio": [], "titles": [], "settings": {}}
    with pytest.raises(ValueError, match="v3 job contains prohibited secret"):
        studio_job.export_job({"project": project, "runtime": {}, "generateIDs": [],
            "drawThingsImageJobs": [{"id": "image", "kind": "image",
                                      "request": remote_request(),
                                      "connection": connection()}]}, tmp_path / "bad-project.json")


def test_v1_v2_formats_remain_allowed():
    assert studio_job.SUPPORTED_JOB_FORMATS == {
        "weetodd-studio-job-v1", "weetodd-studio-job-v2", "weetodd-studio-job-v3"
    }


def test_dependency_binding_accepts_only_explicit_first_role(tmp_path):
    jobs = [
        {"id": "image", "kind": "image", "request": remote_request(),
         "connection": connection(), "dependsOn": []},
        {"id": "video", "kind": "clip", "clipID": "clip", "request": remote_request("video", "r2"),
         "connection": connection(), "dependsOn": ["image"],
         "inputBindings": {"first": "image"}},
    ]
    project = clip_job(tmp_path)["project"]
    assert [job["id"] for job in studio_job.validate_remote_jobs(jobs, project)] == [
        "image", "video",
    ]
    jobs[1]["inputBindings"] = {"control": "image"}
    with pytest.raises(ValueError, match="unsupported dependency conditioning role"):
        studio_job.validate_remote_jobs(jobs, project)


def test_execute_reuses_verified_remote_hash_without_second_adapter(tmp_path, monkeypatch):
    module = importlib.import_module("studio_drawthings")
    calls = []
    monkeypatch.setattr(module, "adapter_for", lambda request: FakeAdapter(calls))
    job = image_job(tmp_path)
    output = tmp_path / "output"
    first = studio_job.execute(job, output, resume=False)
    assert Path(first["remoteResults"]["image"]["path"]).is_file()
    assert [name for name, _ in calls] == ["prepare", "generate"]
    second = studio_job.execute(job, output, resume=True)
    assert second == first
    assert [name for name, _ in calls] == ["prepare", "generate"]


def test_failed_submission_is_not_retried(tmp_path, monkeypatch):
    module = importlib.import_module("studio_drawthings")
    calls = []
    monkeypatch.setattr(module, "adapter_for", lambda request: FakeAdapter(calls, fail=True))
    job = image_job(tmp_path)
    output = tmp_path / "output"
    with pytest.raises(RuntimeError, match="after submission"):
        studio_job.execute(job, output, resume=False)
    with pytest.raises(ValueError, match="will not be retried automatically"):
        studio_job.execute(job, output, resume=True)
    assert [name for name, _ in calls].count("generate") == 1


@pytest.mark.parametrize("damage", ["missing", "corrupt"])
def test_completed_remote_with_unverified_artifact_is_not_retried(
    tmp_path, monkeypatch, damage
):
    module = importlib.import_module("studio_drawthings")
    calls = []
    monkeypatch.setattr(module, "adapter_for", lambda request: FakeAdapter(calls))
    job = image_job(tmp_path)
    output = tmp_path / "output"
    result = studio_job.execute(job, output, resume=False)
    artifact = Path(result["remoteResults"]["image"]["path"])
    if damage == "missing":
        artifact.unlink()
    else:
        artifact.write_bytes(b"changed")
    with pytest.raises(ValueError, match="deliberate new job"):
        studio_job.execute(job, output, resume=True)
    assert [name for name, _ in calls].count("generate") == 1


def test_changed_remote_input_invalidates_resume_before_adapter(tmp_path, monkeypatch):
    module = importlib.import_module("studio_drawthings")
    calls = []
    monkeypatch.setattr(module, "adapter_for", lambda request: FakeAdapter(calls))
    source = tmp_path / "source.png"
    source.write_bytes(b"source")
    job = image_job(tmp_path, source=source)
    output = tmp_path / "output"
    studio_job.execute(job, output, resume=False)
    source.write_bytes(b"changed source")
    with pytest.raises(ValueError, match="source inputs changed"):
        studio_job.execute(job, output, resume=True)
    assert [name for name, _ in calls].count("generate") == 1


def test_first_dependency_is_resolved_with_verified_path_and_hash(tmp_path):
    image = tmp_path / "image.png"
    image.write_bytes(b"image")
    request = remote_request("video", "video-request")
    remote = {"request": request, "inputBindings": {"first": "image"}}
    resolved = studio_job.bind_remote_inputs(
        remote, {"image": {"path": str(image), "sha256": studio_job.file_hash(image)}}
    )
    assert resolved["inputs"] == [{
        "role": "first", "path": str(image.resolve()),
        "sha256": studio_job.file_hash(image), "frameIndex": 0, "strength": 1,
    }]


def clip_job(tmp_path):
    job = image_job(tmp_path)
    job["project"]["clips"] = [{
        "id": "clip", "name": "Remote clip", "engine": "drawThings",
        "sourcePath": "existing-source.mp4", "duration": 1,
        "drawThings": {"profileID": "portable-profile", "modelID": "DISCOVER_MODEL_ID"},
    }]
    job["runtime"]["ffmpegPath"] = "fixture-ffmpeg"
    job["remoteJobs"] = [{
        "id": "video", "kind": "clip", "clipID": "clip",
        "request": remote_request("video"), "connection": connection(), "dependsOn": [],
    }]
    return job


def test_headless_endpoint_clip_keeps_last_frame_when_finishing_and_resuming(tmp_path, monkeypatch):
    job = clip_job(tmp_path)
    canonical = job["remoteJobs"][0]["request"]
    canonical["configuration"].update(numFrames=124, fps=24)
    canonical["inputs"] = [{"role": "first"}, {"role": "last"}]
    job["project"]["clips"][0]["duration"] = 5
    monkeypatch.setattr(studio_job, "preflight", lambda *a, **k: None)
    monkeypatch.setattr(studio_job, "inputs_fingerprint", lambda *a: "fixture")
    monkeypatch.setattr(studio_job, "validate_remote_jobs", lambda jobs, project: jobs)
    monkeypatch.setattr(studio_job, "bind_remote_inputs", lambda remote, results: remote["request"])
    monkeypatch.setattr(importlib.import_module("studio_drawthings"), "adapter_for",
                        lambda request: FakeAdapter([]))
    renders = []

    def render(request, adapter, folder, *args):
        renders.append(request)
        video = folder / "clip.mp4"
        video.write_bytes(b"validated remote fixture")
        return {"video": str(video)}

    def export(request, output, **kwargs):
        assert request["project"]["clips"][0]["duration"] == pytest.approx(124 / 24)
        output.write_bytes(b"finished fixture")
        return {"video": str(output)}

    monkeypatch.setattr(importlib.import_module("wee_todd_remote.studio"),
                        "render_drawthings_clip", render)
    monkeypatch.setattr(studio_job.bridge, "export_movie", export)
    output = tmp_path / "output"
    studio_job.execute(job, output, resume=False)
    studio_job.execute(job, output, resume=True)
    assert len(renders) == 1
    resolved = json.loads((output / "job-state.json").read_text())["resolvedProject"]
    assert resolved["clips"][0]["duration"] == pytest.approx(124 / 24)


@pytest.mark.parametrize("action", ["export", "preflight", "execute"])
@pytest.mark.parametrize("malformation", [
    "missing_target", "duplicate_project_id", "wrong_engine", "duplicate_target",
    "missing_selection", "wrong_profile", "wrong_model", "clip_producer",
])
def test_malformed_remote_destinations_fail_before_adapter(
    tmp_path, monkeypatch, action, malformation
):
    job = clip_job(tmp_path)
    clip = job["project"]["clips"][0]
    remote = job["remoteJobs"][0]
    if malformation == "missing_target":
        remote["clipID"] = "absent"
    elif malformation == "duplicate_project_id":
        job["project"]["clips"].append(copy.deepcopy(clip))
    elif malformation == "wrong_engine":
        clip["engine"] = "ltx23"
    elif malformation == "duplicate_target":
        job["remoteJobs"].append({**copy.deepcopy(remote), "id": "second-video"})
    elif malformation == "missing_selection":
        del clip["drawThings"]
    elif malformation == "wrong_profile":
        clip["drawThings"]["profileID"] = "another-profile"
    elif malformation == "wrong_model":
        clip["drawThings"]["modelID"] = "another-model"
    elif malformation == "clip_producer":
        job["project"]["clips"].append({**copy.deepcopy(clip), "id": "producer-clip"})
        job["remoteJobs"].append({
            **copy.deepcopy(remote), "id": "producer", "clipID": "producer-clip",
        })
        remote.update(dependsOn=["producer"], inputBindings={"first": "producer"})

    calls = []
    module = importlib.import_module("studio_drawthings")
    monkeypatch.setattr(module, "adapter_for", lambda _: FakeAdapter(calls))
    monkeypatch.setattr(studio_job.bridge, "preflight_finishing", lambda *_: None)
    monkeypatch.setattr(studio_job.bridge, "inspect_media", lambda *_: {"kind": "image"})

    def forbidden_generation(*args, **kwargs):
        calls.append(("generate-video", None))
        raise AssertionError("malformed job reached generation")

    monkeypatch.setattr(
        importlib.import_module("wee_todd_remote.studio"),
        "render_drawthings_clip", forbidden_generation,
    )
    with pytest.raises(ValueError, match="Draw Things|first input binding"):
        if action == "export":
            studio_job.export_job({
                "project": job["project"], "runtime": {}, "generateIDs": [],
                "drawThingsImageJobs": job["remoteJobs"],
            }, tmp_path / "invalid.json")
        elif action == "preflight":
            studio_job.preflight(job, tmp_path / "preflight")
        else:
            studio_job.execute(job, tmp_path / "execute", resume=False)
    assert calls == []
    assert not (tmp_path / "invalid.json").exists()
