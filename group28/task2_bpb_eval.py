"""Plot training/validation BPB and detect where each curve flattens."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


CONSECUTIVE_EVALUATIONS = 3


def find_flatten_point(df, metric, threshold, consecutive=CONSECUTIVE_EVALUATIONS):
    """Return the first step confirming consecutive small relative BPB changes."""
    previous = df[metric].shift(1)
    relative_change = df[metric].sub(previous).abs().div(previous.abs())
    is_flat = relative_change.lt(threshold)
    confirmed = is_flat.rolling(consecutive, min_periods=consecutive).sum().eq(consecutive)

    matches = df.loc[confirmed, "step"]
    return None if matches.empty else int(matches.iloc[0])


def main():
    if len(sys.argv) != 3:
        print(f"Usage: python {sys.argv[0]} <metrics.jsonl> <relative_threshold>")
        print("Example: python group28/task2_bpb_eval.py metrics.jsonl 0.01")
        sys.exit(1)

    metrics_path = Path(sys.argv[1])
    threshold = float(sys.argv[2])
    if not 0 < threshold < 1:
        raise ValueError("relative_threshold must be between 0 and 1 (for example, 0.01 for 1%)")

    df = pd.read_json(metrics_path, lines=True)
    required_columns = {"step", "train_bpb", "val_bpb"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    df = (
        df.dropna(subset=list(required_columns))
        .sort_values("step")
        .drop_duplicates(subset="step", keep="last")
        .reset_index(drop=True)
    )
    if df.empty:
        raise ValueError("No complete BPB evaluation rows were found")
    df["step"] = df["step"].astype(int)

    train_flatten_point = find_flatten_point(df, "train_bpb", threshold)
    val_flatten_point = find_flatten_point(df, "val_bpb", threshold)

    threshold_percent = 100 * threshold
    print(
        f"Training flatten point ({CONSECUTIVE_EVALUATIONS} consecutive relative changes "
        f"below {threshold_percent:g}%): {train_flatten_point or 'not detected'}"
    )
    print(
        f"Validation flatten point ({CONSECUTIVE_EVALUATIONS} consecutive relative changes "
        f"below {threshold_percent:g}%): {val_flatten_point or 'not detected'}"
    )

    plt.figure(figsize=(8, 6))
    plt.plot(df["step"], df["train_bpb"], marker="o", color="tab:blue", label="Training BPB")
    plt.plot(df["step"], df["val_bpb"], marker="s", color="tab:orange", label="Validation BPB")

    if train_flatten_point is not None:
        plt.axvline(
            x=train_flatten_point,
            color="tab:blue",
            linestyle="--",
            label="Training flatten point",
        )
    if val_flatten_point is not None:
        plt.axvline(
            x=val_flatten_point,
            color="tab:orange",
            linestyle="--",
            label="Validation flatten point",
        )

    plt.xlabel("Training step")
    plt.ylabel("Bits per byte (BPB)")
    plt.title("Training and validation BPB")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    output_path = metrics_path.with_suffix(".png")
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Plot saved to: {output_path}")


if __name__ == "__main__":
    main()
