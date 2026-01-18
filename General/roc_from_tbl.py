#!/usr/bin/env python3
"""
Plot ROC curve + compute AUC from an HMMER hmmsearch --tblout file.

- Labels are taken from the target name: ...|label=...
- Label binarization (case-insensitive):
    pathogenic-like -> 1  (contains 'pathogenic')
    benign-like     -> 0  (contains 'benign')
- Score used by default: -bitscore (higher = worse fit), matching your other plots.
- AUC calculation: rank-based Mann–Whitney formulation (same as metrics_from_tbl.py).

Example:
  python roc_from_tbl.py --tbl scores_hmmer.tbl --out roc.png --title "ROC: TP53 profile-HMM"
"""

from __future__ import annotations
import argparse
import re
from pathlib import Path
from typing import Optional, List, Tuple

import matplotlib.pyplot as plt


LABEL_RE = re.compile(r"\|label=([^|]+)")

def parse_tblout(tbl_path: Path) -> List[Tuple[float, int]]:
    """
    Returns a list of (score, y) pairs where y in {0,1}.
    Expects HMMER --tblout with bitscore in column 6 (0-based index 5).
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

        # Default score direction: -bitscore (higher=worse fit)
        score = -bitscore
        pairs.append((score, y))

    return pairs


def binarize_label(label: Optional[str]) -> Optional[int]:
    """Return 0 for benign-like, 1 for pathogenic-like, else None."""
    if label is None:
        return None
    lab = label.lower()
    if "pathogenic" in lab:
        return 1
    if "benign" in lab:
        return 0
    return None


def roc_auc_rank(y_true: List[int], scores: List[float]) -> float:
    """
    Rank-based ROC-AUC (Mann–Whitney U).
    Same method used in metrics_from_tbl.py.
    """
    pairs = list(zip(scores, y_true))
    pairs.sort(key=lambda x: x[0])  # ascending score

    # average ranks for ties
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
    Compute ROC curve points by sweeping threshold from +inf down to -inf.
    We treat "score >= threshold" as predicting positive.
    """
    # Sort by score descending
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
        # process all ties at this score together (standard ROC handling)
        j = i
        while j < len(s_sorted) and s_sorted[j] == s_sorted[i]:
            j += 1

        # if we drop threshold to include these, we add their contributions
        for k in range(i, j):
            if y_sorted[k] == 1:
                tp += 1
            else:
                fp += 1

        fpr.append(fp / N)
        tpr.append(tp / P)

        i = j

    # Ensure it ends at (1,1)
    if fpr[-1] != 1.0 or tpr[-1] != 1.0:
        fpr.append(1.0)
        tpr.append(1.0)

    return fpr, tpr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tbl", required=True, help="HMMER hmmsearch --tblout file")
    ap.add_argument("--out", default="roc.png", help="Output PNG filename")
    ap.add_argument("--title", default="ROC: HMMER scores", help="Plot title prefix")
    args = ap.parse_args()

    pairs = parse_tblout(Path(args.tbl))
    if not pairs:
        raise SystemExit("No labeled rows found. Do target names include '|label=...'?")

    scores = [s for s, _ in pairs]
    y_true = [y for _, y in pairs]

    auc = roc_auc_rank(y_true, scores)
    fpr, tpr = roc_curve_points(y_true, scores)

    plt.figure()
    plt.plot(fpr, tpr, linewidth=2)
    # orange diagonal x=y
    plt.plot([0, 1], [0, 1], color="orange", linewidth=2, linestyle="--")

    plt.xlim(0, 1)
    plt.ylim(0, 1.02)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{args.title} (AUC={auc:.3f})")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)

    # Print summary like a sanity check
    n = len(y_true)
    n_pos = sum(y_true)
    n_neg = n - n_pos
    print(f"File: {args.tbl}")
    print(f"N used = {n} (pos/pathogenic={n_pos}, neg/benign={n_neg})")
    print(f"AUC (rank-based) = {auc:.6f}")
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
