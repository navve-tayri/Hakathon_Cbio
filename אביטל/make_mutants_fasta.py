#!/usr/bin/env python3
"""
Create a FASTA file of TP53 mutant protein sequences from a ClinVar tabular file.

Inputs:
  --clinvar clinvar_tp53.txt          (tab-delimited ClinVar export; includes 'Protein change' and 'Germline classification')
  --ref    tp53_orthologs_clean.fasta (FASTA that contains a human TP53 protein sequence)
Outputs:
  --out mutants.fasta                 (WT + one record per accepted missense variant)
  --manifest mutants_manifest.tsv     (created/skipped log)

Notes:
- ClinVar's "Protein change" column in your file appears to contain comma-separated short forms
  like "D10N, D49N" etc. We parse those (single-letter AA, position, single-letter AA).
- We verify that the reference AA at that position matches; otherwise we skip and log.
"""

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

AA3_TO_1 = {
    "Ala":"A","Arg":"R","Asn":"N","Asp":"D","Cys":"C","Gln":"Q","Glu":"E","Gly":"G",
    "His":"H","Ile":"I","Leu":"L","Lys":"K","Met":"M","Phe":"F","Pro":"P","Ser":"S",
    "Thr":"T","Trp":"W","Tyr":"Y","Val":"V","Ter":"*", "Stop":"*"
}

# Matches short form like R175H or D49N
SHORT_AA_RE = re.compile(r"\b([ACDEFGHIKLMNPQRSTVWY])(\d+)([ACDEFGHIKLMNPQRSTVWY])\b")

# Matches HGVS-like p.Arg175His
HGVS_P_RE = re.compile(r"p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2}|\*)")

def read_fasta(path: Path) -> List[Tuple[str, str]]:
    records = []
    header = None
    seq_chunks: List[str] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(seq_chunks)))
                header = line[1:].strip()
                seq_chunks = []
            else:
                seq_chunks.append(line.strip())
        if header is not None:
            records.append((header, "".join(seq_chunks)))
    return records

def pick_human_tp53(records: List[Tuple[str, str]]) -> Tuple[str, str]:
    """
    Choose a human TP53 sequence from the FASTA.
    Heuristics: header contains OX=9606 OR 'Homo sapiens' OR 'HUMAN'.
    """
    for h, s in records:
        if "OX=9606" in h or "Homo sapiens" in h or "HUMAN" in h.upper():
            return h, s
    # fallback: first record
    return records[0]

def parse_protein_change_field(field: str) -> Optional[Tuple[str, int, str]]:
    """
    Try to extract one mutation (refAA, position, altAA) from ClinVar "Protein change" field.

    Your file often has comma-separated short codes like "D10N, D49N" etc.
    We pick the FIRST match we find.
    """
    if not field:
        return None

    # 1) Try short form first
    m = SHORT_AA_RE.search(field)
    if m:
        ref, pos, alt = m.group(1), int(m.group(2)), m.group(3)
        return ref, pos, alt

    # 2) Try HGVS p.Arg175His format
    m2 = HGVS_P_RE.search(field)
    if m2:
        ref3, pos, alt3 = m2.group(1), int(m2.group(2)), m2.group(3)
        ref = AA3_TO_1.get(ref3)
        alt = AA3_TO_1.get(alt3) if alt3 != "*" else "*"
        if ref and alt and alt != "*":  # skip stop codons for missense pipeline
            return ref, pos, alt

    return None

def normalize_label(label: str) -> Optional[str]:
    """
    Keep only the labels useful for your benign vs pathogenic task:
      Benign, Likely benign, Pathogenic, Likely pathogenic, Pathogenic/Likely pathogenic
    """
    if not label:
        return None
    lab = label.strip()
    allowed = {
        "Benign",
        "Likely benign",
        "Pathogenic",
        "Likely pathogenic",
        "Pathogenic/Likely pathogenic",
        "Likely pathogenic/Pathogenic",
    }
    if lab in allowed:
        # normalize combined labels
        if "Pathogenic" in lab and "Likely" in lab:
            return "Pathogenic_or_Likely_pathogenic"
        return lab.replace(" ", "_")
    return None

def is_missense(consequence: str) -> bool:
    return bool(consequence) and ("missense variant" in consequence)

def mutate(seq: str, pos1: int, ref: str, alt: str) -> Optional[str]:
    """
    Apply a 1-based position mutation to sequence.
    Returns mutated sequence, or None if ref mismatch or pos out of range.
    """
    if pos1 < 1 or pos1 > len(seq):
        return None
    idx = pos1 - 1
    if seq[idx].upper() != ref.upper():
        return None
    return seq[:idx] + alt.upper() + seq[idx+1:]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clinvar", required=True, help="Path to clinvar_tp53.txt (tab-delimited)")
    ap.add_argument("--ref", required=True, help="Path to tp53_orthologs_clean.fasta (contains human TP53)")
    ap.add_argument("--out", default="mutants.fasta", help="Output FASTA path")
    ap.add_argument("--manifest", default="mutants_manifest.tsv", help="Output TSV log path")
    ap.add_argument("--max", type=int, default=0, help="Optional cap on number of mutants (0 = no cap)")
    args = ap.parse_args()

    clinvar_path = Path(args.clinvar)
    ref_path = Path(args.ref)

    # Load reference FASTA and pick human
    fasta_records = read_fasta(ref_path)
    if not fasta_records:
        raise SystemExit(f"No FASTA records found in {ref_path}")
    human_header, human_seq = pick_human_tp53(fasta_records)

    # Read ClinVar table
    created = 0
    seen_keys = set()

    # Write outputs
    with open(args.out, "w", encoding="utf-8") as fout, open(args.manifest, "w", encoding="utf-8", newline="") as mlog:
        mw = csv.writer(mlog, delimiter="\t")
        mw.writerow(["status","variant_name","protein_change_raw","parsed_mut","label","reason"])

        # Write WT first
        fout.write(f">TP53_WT|ref_from={human_header}\n")
        # wrap at 60 chars
        for i in range(0, len(human_seq), 60):
            fout.write(human_seq[i:i+60] + "\n")

        with clinvar_path.open("r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f, delimiter="\t")
            # expected headers include: Protein change, Molecular consequence, Germline classification, Name
            for row in reader:
                consequence = row.get("Molecular consequence", "")
                if not is_missense(consequence):
                    continue

                label = normalize_label(row.get("Germline classification", ""))
                if label is None:
                    continue

                prot_change_field = row.get("Protein change", "")
                parsed = parse_protein_change_field(prot_change_field)
                if parsed is None:
                    mw.writerow(["skipped", row.get("Name",""), prot_change_field, "", label, "Could not parse protein change"])
                    continue

                refAA, pos, altAA = parsed
                key = (refAA, pos, altAA, label)
                if key in seen_keys:
                    continue
                seen_keys.add(key)

                mut_seq = mutate(human_seq, pos, refAA, altAA)
                if mut_seq is None:
                    mw.writerow(["skipped", row.get("Name",""), prot_change_field, f"{refAA}{pos}{altAA}", label,
                                 "Reference AA mismatch or position out of range (check isoform/numbering)"])
                    continue

                # write mutant
                header = f"TP53_{refAA}{pos}{altAA}|label={label}"
                fout.write(f">{header}\n")
                for i in range(0, len(mut_seq), 60):
                    fout.write(mut_seq[i:i+60] + "\n")

                mw.writerow(["created", row.get("Name",""), prot_change_field, f"{refAA}{pos}{altAA}", label, ""])
                created += 1

                if args.max and created >= args.max:
                    break

    print(f"Done. Wrote WT + {created} mutants to {args.out}")
    print(f"Manifest written to {args.manifest}")
    print("If many variants were skipped due to 'Reference AA mismatch', your reference sequence isoform may not match ClinVar numbering.")

if __name__ == "__main__":
    main()
