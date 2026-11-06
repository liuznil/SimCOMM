"""Summarize zero-shot and domain-classifier results from multiple runs.
  python summarize_runs.py A=results_a.csv B=results_b.csv [--out results/sensitivity.csv]
"""
import argparse
import os

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", metavar="NAME=CSV")
    ap.add_argument("--out", default=os.path.join("results", "sensitivity.csv"))
    args = ap.parse_args()

    frames = []
    required = {"H", "target", "method", "auroc", "ci_lo", "ci_hi"}
    for spec in args.runs:
        if "=" not in spec:
            ap.error(f"expected NAME=CSV, got {spec!r}")
        name, path = spec.split("=", 1)
        if not name or not path:
            ap.error(f"expected NAME=CSV, got {spec!r}")
        df = pd.read_csv(path)
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")
        is_domain = df["target"].eq("domain-classifier")
        is_zero_shot = df["method"].isin(["sim_drop", "sim_dropgeo", "sim_los2nlos"])
        selected = df[is_domain | is_zero_shot].copy()
        if selected.empty:
            raise ValueError(f"{path} contains no zero-shot or domain-classifier rows")
        selected.insert(0, "config", name)
        frames.append(selected[["config", "H", "target", "method", "auroc", "ci_lo", "ci_hi"]])

    result = pd.concat(frames, ignore_index=True).sort_values(
        ["config", "H", "target", "method"], kind="stable"
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    result.to_csv(args.out, index=False)
    print(result.to_string(index=False))
    print(f"\nsaved {len(result)} rows -> {args.out}")


if __name__ == "__main__":
    main()
