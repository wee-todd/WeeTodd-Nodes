"""Execute an unchanged saved ComfyUI API graph and record server-side elapsed time.

Use a dedicated ComfyUI server started with --cache-none for repeated warm runs.
The engine's explicit unload controls remain authoritative; this script does not
edit the prompt, change seeds, start/stop servers, or clear another user's queue.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path


def request(url, endpoint, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        url.rstrip("/") + endpoint, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=30) as response:
        return json.load(response)


def elapsed_seconds(history):
    messages = history["status"]["messages"]
    starts = [v["timestamp"] for kind, v in messages if kind == "execution_start"]
    stops = [v["timestamp"] for kind, v in messages if kind == "execution_success"]
    if len(starts) != 1 or len(stops) != 1:
        raise ValueError("Successful history has no unambiguous server-side timing.")
    return (stops[0] - starts[0]) / 1000.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workflow", type=Path, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:8199")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=3600)
    args = parser.parse_args()
    if args.runs < 1 or args.timeout_seconds <= 0:
        parser.error("runs and timeout must be positive")
    if args.output.exists():
        parser.error("output already exists; choose a unique report filename")
    raw = args.workflow.read_bytes()
    graph = json.loads(raw)
    queue = request(args.url, "/queue")
    if queue["queue_running"] or queue["queue_pending"]:
        raise RuntimeError("The benchmark server is busy; no prompt was submitted.")
    report = {
        "workflow": str(args.workflow.resolve()),
        "workflow_sha256": hashlib.sha256(raw).hexdigest(),
        "runs": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for index in range(args.runs):
        submitted = request(args.url, "/prompt", {"prompt": graph})
        prompt_id = submitted["prompt_id"]
        print(f"Run {index + 1}/{args.runs}: {prompt_id}", flush=True)
        deadline = time.monotonic() + args.timeout_seconds
        while True:
            histories = request(args.url, f"/history/{prompt_id}")
            if prompt_id in histories:
                history = histories[prompt_id]
                entry = {"prompt_id": prompt_id, "history": history}
                if history["status"]["status_str"] == "success":
                    entry["server_seconds"] = elapsed_seconds(history)
                report["runs"].append(entry)
                args.output.write_text(json.dumps(report, indent=2) + "\n")
                if "server_seconds" not in entry:
                    raise RuntimeError(f"Render failed; inspect {args.output}")
                cached = [
                    v.get("nodes", [])
                    for kind, v in history["status"]["messages"]
                    if kind == "execution_cached"
                ]
                if any(cached):
                    raise RuntimeError(
                        "Cached graph nodes invalidate this benchmark; use --cache-none."
                    )
                print(f"Completed run {index + 1}: {entry['server_seconds']:.3f}s", flush=True)
                break
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Prompt {prompt_id} is still running; it was not interrupted.")
            time.sleep(20)
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
