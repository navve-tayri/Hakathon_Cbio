#!/usr/bin/env python3
import re
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


LABEL_RE = re.compile(r"\|label=([^|]+)")

def extract_effective_label(target: str) -> str | None:
    """
    In your Model-C tbl, target may contain multiple |label=... parts, e.g.:
    TP53_D354Y|label=Likely_benign|label=Benign
    We'll take the LAST label as the final one (Benign/Pathogenic).
    """
    labs = LABEL_RE.findall(target)
    if not labs:
        return None
    return labs[-1].strip()

def binarize_label(label: str | None) -> int | None:
    if label is None:
        return None
    lab = label.lower()
    if "benign" in lab:
        return 0
    if "pathogenic" in lab:
        return 1
    return None

def parse_modelc_tbl(tbl_path: Path):
    """
    Parses the Model-C pseudo-HMMER tbl format you generate.
    We only require:
      - Column 0: target name (contains |label=...)
      - Column 5: score (your raw_score)
    We ignore e-value (which is '-') entirely.
    """
    benign_scores = []
    patho_scores = []

    for line in tbl_path.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        parts = line.split()
        if len(parts) < 6:
            continue

        target = parts[0]
        try:
            score = float(parts[5])
        except ValueError:
            continue

        label = extract_effective_label(target)
        y = binarize_label(label)
        if y is None:
            continue

        # Keep the SAME convention as your existing script default:
        # plot_score = -bitscore; here score plays the role of bitscore.
        plot_score = -score

        if y == 0:
            benign_scores.append(plot_score)
        else:
            patho_scores.append(plot_score)

    return np.array(benign_scores, dtype=float), np.array(patho_scores, dtype=float)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tbl", required=True, help="Model-C tbl file (e.g., scores_C.tbl)")
    ap.add_argument("--out", default="boxplot_scores_C.png", help="Output PNG filename")
    ap.add_argument("--title", default=None, help="Plot title")
    args = ap.parse_args()

    benign, patho = parse_modelc_tbl(Path(args.tbl))

    if len(benign) == 0 and len(patho) == 0:
        raise SystemExit("No labeled rows found. Do target names include '|label=Benign/Pathogenic'?")
    if len(benign) == 0 or len(patho) == 0:
        raise SystemExit(f"Need both classes for boxplot. Got benign={len(benign)}, pathogenic={len(patho)}")

    data = [benign, patho]
    labels = ["Benign-like", "Pathogenic-like"]

    plt.figure()
    plt.boxplot(data, labels=labels, showfliers=False)

    # overlay jitter points for visibility
    rng = np.random.default_rng(0)
    for i, arr in enumerate(data, start=1):
        x = rng.normal(loc=i, scale=0.04, size=len(arr))
        plt.scatter(x, arr, s=12)

    plt.ylabel("-log-likelihood (higher = worse fit)")
    if args.title:
        plt.title(args.title)
    else:
        plt.title(f"Score distributions (Model C) from {Path(args.tbl).name}")

    plt.grid(True, axis="y", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)

    print(f"Saved: {args.out}")
    print(f"N benign-like={len(benign)}, N pathogenic-like={len(patho)}")

if __name__ == "__main__":
    main()
