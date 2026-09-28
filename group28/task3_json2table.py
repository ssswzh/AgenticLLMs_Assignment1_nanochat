"""Collect Task 3 evaluation JSON files into a TSV table."""

import argparse
import json
from pathlib import Path

import pandas as pd


METRICS = {"ARC-Easy": 2376, "ARC-Challenge": 1172, "GSM8K": 1319}
STAGE_ORDER = {"Base": 0, "Mid-training": 1, "SFT": 2}


def parse_args():
    parser = argparse.ArgumentParser(description="Convert evaluation json files to TSV.")
    parser.add_argument("--input", type=str, default=None, help="Evaluation json files (multiple files)", nargs="+")
    parser.add_argument("--output", type=Path, default=None, help="Output TSV path")
    return parser.parse_args()


def infer_stage(path, model_tag):
    text = f"{path.stem} {model_tag}".lower()
    if "midtrain" in text:
        return "Mid-training"
    if "base" in text:
        return "Base"
    if "sft" in text:
        return "SFT"
    return text.replace(" ", "-")


def main():
    args = parse_args()
    json_paths = sorted(args.input)
    if not json_paths:
        raise FileNotFoundError(f"No json files found in {args.input}")

    rows = []
    for path in json_paths:
        path = Path(path)
        with path.open(encoding="utf-8") as file:
            result = json.load(file)

        required = ["model_tag", "step", *METRICS]
        missing = [field for field in required if field not in result]
        if missing:
            raise ValueError(f"{path}: missing fields: {', '.join(missing)}")

        rows.append(
            {
                "Stage": infer_stage(path, result["model_tag"]),
                "Model tag": result["model_tag"],
                "Step": result["step"],
                **{f"{metric} (N={total})": f"{round(total*result[metric])} ({round(100 * result[metric],2)}%)" for metric, total in METRICS.items()},
            }
        )

    table = pd.DataFrame(rows)
    table["_stage_order"] = table["Stage"].map(STAGE_ORDER).fillna(len(STAGE_ORDER))
    table = table.sort_values(["_stage_order", "Step"]).drop(columns="_stage_order")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, sep="\t", index=False, float_format="%.2f")
    print(table.to_string(index=False, float_format=lambda value: f"{value:.2f}"))
    print(f"Saved table to {args.output}")


if __name__ == "__main__":
    main()
