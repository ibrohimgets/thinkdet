#!/usr/bin/env python
import argparse
import json
from collections import defaultdict
from pathlib import Path

from functional_layer_signal import summarize_rows, write_summary_markdown


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", help="Shard JSONs or directories containing *.json")
    parser.add_argument("--output", type=str, required=True)
    return parser.parse_args()


def iter_jsons(inputs):
    for raw in inputs:
        path = Path(raw)
        if path.is_dir():
            for child in sorted(path.glob("*.json")):
                if child.name not in {"summary.json"}:
                    yield child
        else:
            yield path


def main():
    args = parse_args()
    paths = list(iter_jsons(args.inputs))
    if not paths:
        raise SystemExit("No shard JSON files found")

    combined_rows = defaultdict(list)
    combined_affordance = defaultdict(lambda: defaultdict(list))
    base = None

    for path in paths:
        with path.open("r") as f:
            payload = json.load(f)
        if payload.get("status") != "ok":
            raise ValueError(f"Bad shard status in {path}")
        if base is None:
            base = dict(payload)
        for layer, item in payload["per_layer"].items():
            combined_rows[layer].extend(item["per_sample"])
            for row in item["per_sample"]:
                combined_affordance[layer][row["affordance_id"]].append(row)

    topk_tokens = base["topk_tokens"]
    per_layer = {}
    for layer, rows in sorted(combined_rows.items(), key=lambda kv: int(kv[0])):
        per_layer[str(layer)] = {
            "summary": summarize_rows(rows, topk_tokens),
            "per_affordance": {
                aff: summarize_rows(aff_rows, topk_tokens)
                for aff, aff_rows in sorted(combined_affordance[layer].items())
            },
            "per_sample": rows,
        }

    payload = {
        **base,
        "status": "ok",
        "num_shards": len(paths),
        "shard_index": None,
        "n_samples": len(next(iter(combined_rows.values()))) if combined_rows else 0,
        "per_layer": per_layer,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    md_path, winner = write_summary_markdown(payload, str(out_path))

    print("Saved:", out_path)
    print("Saved:", md_path)
    print("winner_layer:", winner)


if __name__ == "__main__":
    main()
