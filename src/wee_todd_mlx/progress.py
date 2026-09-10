"""Lightweight newline-delimited progress shared by native headless and Studio runs."""

import json


def render_progress(stage, message, *, completed=None, total=None):
    event = {"event": "progress", "stage": stage, "message": message, "fraction": 0}
    if completed is not None and total is not None and total > 0:
        if completed >= total:
            event.update(stage="finishing", message="Decoding and publishing")
        else:
            event.update(message=f"{message} {completed}/{total}", fraction=completed / total)
    print(json.dumps(event), flush=True)
