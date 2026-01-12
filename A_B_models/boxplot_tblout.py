#!/usr/bin/env python3
import re
import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def parse_tblout(tbl_path: Path) -> pd.DataFrame:
    rows = []
    for line in tbl_path.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        parts = line.split()
        # HMMER --tblout format: targetname accession queryname accession E-value score bias ...
        if len(parts) < 6:
            continue

        target = parts[0]
        try:
            evalue = float(parts[4])
            bitscore = float(parts[5])
        except ValueError:
            continue

        m = re.search(r"\|label=([^|]+)", target)
        label = m.group(1) if m else None

        rows.append({"target": target, "label": label, "evalue": evalue, "bitscore": bitscore})

    return pd.DataFrame(rows)


def binarize_label(label: str):
    """Return 0 for benign-like, 1 for pathogenic-like, else None."""
    if label is None:
        return None
    # Your labels look like: Benign, Likely_benign, Pathogenic, Pathogenic_or_Likely_pathogenic
    if "Benign" in label:
        return 0
    if "Pathogenic" in label:
        return 1
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tbl", required=True, help="HMMER hmmsearch --tblout file (e.g., scores_pfam.tbl)")
    ap.add_argument("--out", default="boxplot_scores.png", help="Output PNG filename")
    ap.add_argument("--score", choices=["bitscore", "neg_bitscore"], default="neg_bitscore",
                    help="Use bitscore or -bitscore. Often -bitscore makes 'higher=worse fit' (more pathogenic).")
    ap.add_argument("--title", default=None, help="Plot title")
    args = ap.parse_args()

    df = parse_tblout(Path(args.tbl))
    if df.empty:
        raise SystemExit("No parsable rows found in tblout (check file format).")

    df["y"] = df["label"].apply(binarize_label)
    df = df[df["y"].notna()].copy()
    df["y"] = df["y"].astype(int)

    if df.empty:
        raise SystemExit("No labeled rows found. Do your target names include '|label=...'?")

    # Choose score direction
    if args.score == "bitscore":
        df["plot_score"] = df["bitscore"]
        ylab = "HMMER bit score (higher = better fit)"
    else:
        df["plot_score"] = -df["bitscore"]
        ylab = "- bit score (higher = worse fit)"

    benign = df[df["y"] == 0]["plot_score"].to_numpy()
    patho = df[df["y"] == 1]["plot_score"].to_numpy()

    data = [benign, patho]
    labels = ["Benign-like", "Pathogenic-like"]

    plt.figure()
    plt.boxplot(data, labels=labels, showfliers=True)

    # Overlay jittered points (to show ties / discreteness)
    rng = np.random.default_rng(0)
    for i, arr in enumerate(data, start=1):
        x = rng.normal(loc=i, scale=0.04, size=len(arr))
        plt.scatter(x, arr, s=12)

    if args.title:
        plt.title(args.title)
    else:
        plt.title(f"Score distributions from {Path(args.tbl).name}")

    plt.ylabel(ylab)
    plt.grid(True, axis="y", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    print(f"Saved: {args.out}")
    print(f"N benign-like={len(benign)}, N pathogenic-like={len(patho)}")


if __name__ == "__main__":
    main()
