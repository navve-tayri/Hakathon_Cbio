import pandas as pd
import re
from pathlib import Path

# =========================
# WT reference (full-length UniProt P04637)
# =========================
WT_SEQUENCE_FULL = """
MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGP
DEAPRMPEAAPPVAPAPAAPTPAAPAPAPSWPLSSSVPSQKTYQGSYGFRLGFLHSGTAK
SVTCTYSPALNKMFCQLAKTCPVQLWVDSTPPPGTRVRAMAIYKQSQHMTEVVRRCPHHE
RCSDSDGLAPPQHLIRVEGNLRVEYLDDRNTFRHSVVVPYEPPEVGSDCTTIHYNYMCNS
SCMGGMNRRPILTIITLEDSSGNLLGRNSFEVRVCACPGRDRRTEEENLRKKGEPHHELP
PGSTKRALPNNTSSSPQPKKKPLDGEYFTLQIRGRERFEMFRELNEALELKDAQAGKEPG
GSRAHSSHLKSKKGQSTSRHKKLMFKTEGPDSD
""".replace("\n", "").replace(" ", "").strip()

# =========================
# Output formatting settings
# =========================
TRIM_ANCHOR = "MDDLMLSPDDIEQWFTEDPGP"
ANCHOR_IDX = WT_SEQUENCE_FULL.find(TRIM_ANCHOR)
if ANCHOR_IDX == -1:
    raise ValueError("TRIM_ANCHOR not found in WT sequence")

WT_SEQUENCE_OUT = WT_SEQUENCE_FULL[ANCHOR_IDX:]  # trimmed for output display only

WT_HEADER = ">TP53_WT|ref_from=tr|H2EHT1|H2EHT1_HUMAN Cellular tumor antigen p53 OS=Homo sapiens OX=9606 GN=TP53 PE=2 SV=1"

# =========================
# Helpers
# =========================
PROT_VAR_RE = re.compile(r"^p\.([A-Z])(\d+)([A-Z])$")

def wrap_fasta(seq: str, width: int = 60) -> str:
    return "\n".join(seq[i:i+width] for i in range(0, len(seq), width))

def apply_missense_on_full_wt(wt_full: str, prot_variant: str) -> str:
    """
    prot_variant like 'p.A138V'
    Applies mutation using full-length WT indexing (1-based).
    Returns full-length mutant sequence.
    """
    m = PROT_VAR_RE.match(prot_variant.strip())
    if not m:
        return None

    refAA, pos_s, altAA = m.group(1), m.group(2), m.group(3)
    pos = int(pos_s)
    idx = pos - 1

    if idx < 0 or idx >= len(wt_full):
        return None

    # strict sanity check
    if wt_full[idx] != refAA:
        # If this triggers, you likely have isoform numbering issues or a different reference.
        raise ValueError(
            f"WT mismatch for {prot_variant}: expected {refAA} at position {pos}, "
            f"but WT has {wt_full[idx]}."
        )

    return wt_full[:idx] + altAA + wt_full[idx+1:]

def to_token(prot_variant: str) -> str:
    # 'p.A138V' -> 'A138V'
    return prot_variant.strip().replace("p.", "")

# =========================
# Main: build FASTA from Excel
# =========================
EXCEL_PATH = Path("../Data/supplementary_table_s1_bbab524.xlsx")
OUTPUT_FASTA = Path("../Data/tp53_s1_posneg_labeled.fasta")

# Adjust these if the sheet names differ in your file
POS_SHEET = "S1A. Positive Set"
NEG_SHEET = "S1B. Negative Set"

df_pos = pd.read_excel(EXCEL_PATH, sheet_name=POS_SHEET)
df_neg = pd.read_excel(EXCEL_PATH, sheet_name=NEG_SHEET)

# Keep only rows with Protein_Variant
df_pos = df_pos[df_pos["Protein_Variant"].notna()].copy()
df_neg = df_neg[df_neg["Protein_Variant"].notna()].copy()

# Deduplicate
pos_vars = sorted(set(df_pos["Protein_Variant"].astype(str).str.strip().tolist()))
neg_vars = sorted(set(df_neg["Protein_Variant"].astype(str).str.strip().tolist()))

written = 0

with OUTPUT_FASTA.open("w", encoding="utf-8") as f_out:
    # 1) WT first
    f_out.write(WT_HEADER + "\n")
    f_out.write(wrap_fasta(WT_SEQUENCE_OUT) + "\n")

    # 2) Positive set -> Pathogenic
    for pv in pos_vars:
        mutant_full = apply_missense_on_full_wt(WT_SEQUENCE_FULL, pv)
        mutant_out = mutant_full[ANCHOR_IDX:]  # trim for output display only

        token = to_token(pv)
        f_out.write(f">TP53_{token}|label=Pathogenic\n")
        f_out.write(wrap_fasta(mutant_out) + "\n")
        written += 1

    # 3) Negative set -> Benign
    for pv in neg_vars:
        mutant_full = apply_missense_on_full_wt(WT_SEQUENCE_FULL, pv)
        mutant_out = mutant_full[ANCHOR_IDX:]

        token = to_token(pv)
        f_out.write(f">TP53_{token}|label=Benign\n")
        f_out.write(wrap_fasta(mutant_out) + "\n")
        written += 1

print("Done.")
print("Output:", OUTPUT_FASTA.resolve())
print("Total variant sequences written (pos+neg):", written)
print("Positive variants:", len(pos_vars), "Negative variants:", len(neg_vars))
