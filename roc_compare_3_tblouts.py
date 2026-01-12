#!/usr/bin/env python3
"""
Plot multiple ROC curves + AUC (rank-based) from multiple HMMER hmmsearch --tblout files.

- Labels are extracted from target name: ...|label=...
- Binarization (case-insensitive):
    pathogenic-like -> 1  (contains 'pathogenic')
    benign-like     -> 0  (contains 'benign')
- Score direction (default): -bitscore  (higher = worse fit), matching roc_from_tbl.py :contentReference[oaicite:1]{index=1}
- AUC: rank-based Mann–Whitney (same as roc_from_tbl.py) :contentReference[oaicite:2]{index=2}

Usage examples:
  python roc_compare_3_tblouts.py \
      --tbl scores_hmmer.tbl --name HMMER \
      --tbl scores_pfam.tbl  --name PFAM \
      --tbl scores_C.tbl     --name C_MODEL \
      --out roc_3models.png \
      --title "ROC: TP53 profile-HMM scores"
"""

from __future__ import annotations
import argparse
import re
from pathlib import Path
from typing import Optional, List, Tuple, Dict

import matplotlib.pyplot as plt

LABEL_RE = re.compile(r"\|label=([^|]+)")

def binarize_label(label: Optional[str]) -> Optional[int]:
    if label is None:
        return None
    lab = label.lower()
    if "pathogenic" in lab:
        return 1
    if "benign" in lab:
        return 0
    return None

def parse_tblout(tbl_path: Path) -> List[Tuple[float, int]]:
    """
    Returns list of (score, y) where y in {0,1}.
    Expects HMMER --tblout with bitscore in column 6 (0-based index 5).
    Uses score=-bitscore (higher=worse fit).
    """
    pairs: List[Tuple[float, int]] = []
    for line in tbl_path.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 6:
            continue

        target = parts[0]
        try:
            bitscore = float(parts[5])
        except ValueError:
            continue

        m = LABEL_RE.search(target)
        label = m.group(1) if m else None
        y = binarize_label(label)
        if y is None:
            continue

        score = -bitscore
        pairs.append((score, y))

    return pairs

def roc_auc_rank(y_true: List[int], scores: List[float]) -> float:
    """
    Rank-based ROC-AUC (Mann–Whitney U), with average ranks for ties.
    """
    pairs = list(zip(scores, y_true))
    pairs.sort(key=lambda x: x[0])  # ascending

    ranks = [0.0] * len(pairs)
    i = 0
    r = 1
    while i < len(pairs):
        j = i
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = (r + (r + (j - i) - 1)) / 2.0
        for k in range(i, j):
            ranks[k] = avg_rank
        r += (j - i)
        i = j

    n_pos = sum(1 for _, y in pairs if y == 1)
    n_neg = sum(1 for _, y in pairs if y == 0)
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    sum_ranks_pos = sum(rank for rank, (_, y) in zip(ranks, pairs) if y == 1)
    return (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)

def roc_curve_points(y_true: List[int], scores: List[float]) -> Tuple[List[float], List[float]]:
    """
    Compute ROC points by sweeping threshold from +inf downwards.
    Predict positive when score >= threshold.
    """
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    y_sorted = [y_true[i] for i in order]
    s_sorted = [scores[i] for i in order]

    P = sum(y_true)
    N = len(y_true) - P
    if P == 0 or N == 0:
        return [0.0, 1.0], [0.0, 1.0]

    tp = 0
    fp = 0
    fpr = [0.0]
    tpr = [0.0]

    i = 0
    while i < len(s_sorted):
        j = i
        while j < len(s_sorted) and s_sorted[j] == s_sorted[i]:
            j += 1

        for k in range(i, j):
            if y_sorted[k] == 1:
                tp += 1
            else:
                fp += 1

        fpr.append(fp / N)
        tpr.append(tp / P)
        i = j

    if fpr[-1] != 1.0 or tpr[-1] != 1.0:
        fpr.append(1.0)
        tpr.append(1.0)

    return fpr, tpr

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tbl", action="append", required=True,
                    help="Path to HMMER hmmsearch --tblout file. Can be provided 3 times.")
    ap.add_argument("--name", action="append", default=None,
                    help="Display name for the corresponding --tbl (same order). Optional.")
    ap.add_argument("--out", default="roc_3models.png", help="Output PNG filename")
    ap.add_argument("--title", default="ROC: TP53 profile-HMM scores", help="Plot title")
    args = ap.parse_args()

    tbls = [Path(p) for p in args.tbl]
    names = args.name if args.name is not None else []
    if names and len(names) != len(tbls):
        raise SystemExit("If you provide --name, you must provide it the same number of times as --tbl.")

    # Default names from filenames if not provided
    if not names:
        names = [t.stem for t in tbls]

    curves: List[Dict] = []
    for tbl_path, display_name in zip(tbls, names):
        pairs = parse_tblout(tbl_path)
        if not pairs:
            raise SystemExit(f"No labeled rows found in {tbl_path}. Do targets include '|label=...'?")

        scores = [s for s, _ in pairs]
        y_true = [y for _, y in pairs]

        auc = roc_auc_rank(y_true, scores)
        fpr, tpr = roc_curve_points(y_true, scores)

        n = len(y_true)
        n_pos = sum(y_true)
        n_neg = n - n_pos

        curves.append({
            "name": display_name,
            "fpr": fpr,
            "tpr": tpr,
            "auc": auc,
            "n": n,
            "n_pos": n_pos,
            "n_neg": n_neg,
            "path": str(tbl_path)
        })

        print(f"File: {tbl_path}")
        print(f"  N used = {n} (pos/pathogenic={n_pos}, neg/benign={n_neg})")
        print(f"  AUC (rank-based) = {auc:.6f}")

    # Plot all curves together
    plt.figure(figsize=(8, 6))

    for c in curves:
        plt.plot(c["fpr"], c["tpr"], linewidth=2,
                 label=f"{c['name']} (AUC={c['auc']:.3f})")

    # Diagonal baseline
    plt.plot([0, 1], [0, 1], linewidth=2, linestyle="--")

    plt.xlim(0, 1)
    plt.ylim(0, 1.02)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(args.title)
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    print(f"\nSaved: {args.out}")

if __name__ == "__main__":
    main()
