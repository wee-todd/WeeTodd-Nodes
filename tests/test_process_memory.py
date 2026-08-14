import subprocess

from scripts.comfy_process_memory import parse_footprint
from wee_todd_nodes.process_memory import complete_process_memory, parse_macos_footprint


def test_parse_complete_process_footprint_script_contract():
    report = parse_footprint("phys_footprint: 1234\nphys_footprint_peak: 5678\n")
    assert report == {"process_peak_bytes": 5678, "process_current_bytes": 1234}


def test_parse_macos_footprint_reports_peak_and_current():
    report = parse_macos_footprint(
        "phys_footprint: 123456\nother: 1\nphys_footprint_peak: 987654\n"
    )
    assert report == {
        "complete_process_peak_bytes": 987654,
        "complete_process_current_bytes": 123456,
    }


def test_complete_process_memory_is_nonfatal_when_probe_fails(monkeypatch):
    monkeypatch.setattr("os.path.isfile", lambda _path: True)

    def fail(*_args, **_kwargs):
        raise subprocess.TimeoutExpired("footprint", 5)

    monkeypatch.setattr("subprocess.run", fail)
    report = complete_process_memory()
    assert report["complete_process_memory_scope"] == "unavailable"
    assert "complete_process_memory_error" in report
