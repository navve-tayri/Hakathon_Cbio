#!/usr/bin/env python3
import argparse
import re
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc

def read_tblout(path: str) -> pd.DataFrame:
    """
    Parse HMMER --tblout output.
    We keep target name and full-sequence bit score.
    """
    rows = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            # tblout is whitespace-separated, with fixed columns.
            # target name is col0, bitscore is col5, evalue is col4 (for full sequence)
            parts = line.split()
            if len(parts) < 6:
                continue
            target = parts[0]
            evalue = float(parts[4])
            bitscore = float(parts[5])
            rows.append((target, evalue, bitscore))
    return pd.DataFrame(rows, columns=["target", "evalue", "bitscore"])

def read_manifest(path: str) -> pd.DataFrame:
    """
    Reads mutants_manifest.tsv produced earlier.
    Expected columns: status, variant_name, protein_change_raw, parsed_mut, label, reason
    """
    df = pd.read_csv(path, sep="\t")
    # Keep only created variants (these are the ones present in mutants.fasta)
    df = df[df["status"] == "created"].copy()

    # Normalize label → binary
    # Positive class = pathogenic-like
    def to_y(lab: str):
        if pd.isna(lab):
            return None
        lab = str(lab)
        if "Pathogenic" in lab:
            return 1
        if "Benign" in lab:
            return 0
        return None

    df["y"] = df["label"].apply(to_y)
    df = df[df["y"].notna()].copy()

    # Build the FASTA header id we used: TP53_{ref}{pos}{alt}|label=...
    # The script wrote: >TP53_{ref}{pos}{alt}|label=LABEL
    # So "target" in tblout should start with "TP53_R175H|label=..."
    df["target_prefix"] = df["parsed_mut"].apply(lambda s: f"TP53_{s}")
    return df[["parsed_mut", "label", "y", "target_prefix"]]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tbl", required=True, help="HMMER --tblout file (scores.tbl)")
    ap.add_argument("--manifest", required=True, help="mutants_manifest.tsv from ClinVar->FASTA step")
    ap.add_argument("--out", default="roc.png", help="Output ROC PNG filename")
    ap.add_argument("--title", default="ROC: TP53 profile-HMM scores", help="Plot title")
    args = ap.parse_args()

    scores = read_tblout(args.tbl)
    man = read_manifest(args.manifest)

    # Join scores to labels.
    # tblout target might include extra suffix; match by prefix.
    # We'll map by the first token (target) starting with "TP53_R175H" etc.
    # Create a dict from prefix->(y,label)
    prefix_to_y = dict(zip(man["target_prefix"], man["y"]))

    def match_y(target: str):
        # exact prefix match
        m = re.match(r"^(TP53_[ACDEFGHIKLMNPQRSTVWY]\d+[ACDEFGHIKLMNPQRSTVWY])", target)
        if not m:
            return None
        prefix = m.group(1)
        return prefix_to_y.get(prefix)

    scores["y"] = scores["target"].apply(match_y)
    scores = scores[scores["y"].notna()].copy()
    scores["y"] = scores["y"].astype(int)

    # IMPORTANT: choose score direction.
    # For many “pathogenicity” intuitions you expect pathogenic variants to have LOWER bitscore (worse fit).
    # ROC expects higher score = more positive, so we can flip:
    scores["score_for_roc"] = -scores["bitscore"]

    if scores["y"].nunique() < 2:
        raise SystemExit("Need both benign-like and pathogenic-like labels after joining. Check headers/manifest.")

    fpr, tpr, _ = roc_curve(scores["y"], scores["score_for_roc"])
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{args.title} (AUC={roc_auc:.3f})")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)

    print(f"Wrote ROC plot to: {args.out}")
    print(f"AUC = {roc_auc:.3f}")
    print(f"Used N={len(scores)} variants after join.")

if __name__ == "__main__":
    main()
