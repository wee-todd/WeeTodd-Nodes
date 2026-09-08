"""Append default H3 controls to saved UI graphs without shifting older widget values."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    changed = []
    for path in sorted((args.project / "workflows").rglob("*.json")):
        graph = json.loads(path.read_text())
        dirty = False
        for node in graph.get("nodes", []):
            kind = node.get("type")
            if kind not in {"WeeToddH3GenerationConfig", "WeeToddH3TextEncode"}:
                continue
            field, value, old_count = (
                ("inference_optimization", "off", 14)
                if kind == "WeeToddH3GenerationConfig"
                else ("persistent_cache", True, 2)
            )
            if any(
                i["name"] == field and i.get("link") is not None for i in node.get("inputs", [])
            ):
                continue
            values = node["widgets_values"]
            if len(values) == old_count:
                values.append(value)
                dirty = True
            elif len(values) != old_count + 1:
                raise ValueError(f"Unexpected widget layout: {path}, node {node['id']}")
        if dirty:
            changed.append(str(path))
            if args.write:
                path.write_text(json.dumps(graph, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"changed": changed, "written": args.write}, indent=2))


if __name__ == "__main__":
    main()
