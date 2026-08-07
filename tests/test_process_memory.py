from scripts.comfy_process_memory import parse_footprint


def test_parse_complete_process_footprint():
    report = parse_footprint("phys_footprint: 1234\nphys_footprint_peak: 5678\n")
    assert report == {"process_peak_bytes": 5678, "process_current_bytes": 1234}
