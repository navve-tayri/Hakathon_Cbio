import csv
import re
from pathlib import Path

# =========================
# WT reference (as provided)
# =========================
WT_SEQUENCE = """
MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGP
DEAPRMPEAAPPVAPAPAAPTPAAPAPAPSWPLSSSVPSQKTYQGSYGFRLGFLHSGTAK
SVTCTYSPALNKMFCQLAKTCPVQLWVDSTPPPGTRVRAMAIYKQSQHMTEVVRRCPHHE
RCSDSDGLAPPQHLIRVEGNLRVEYLDDRNTFRHSVVVPYEPPEVGSDCTTIHYNYMCNS
SCMGGMNRRPILTIITLEDSSGNLLGRNSFEVRVCACPGRDRRTEEENLRKKGEPHHELP
PGSTKRALPNNTSSSPQPKKKPLDGEYFTLQIRGRERFEMFRELNEALELKDAQAGKEPG
GSRAHSSHLKSKKGQSTSRHKKLMFKTEGPDSD
""".replace("\n", "").strip()

# אם את רוצה בדיוק כמו בדוגמה (להתחיל מ-MDDLML...), השאירי True
TRIM_TO_MDDL = True
TRIM_ANCHOR = "MDDLMLSPDDIEQWFTEDPGP"

# כותרת WT בדיוק כמו שהדגמת
WT_HEADER = ">TP53_WT|ref_from=tr|H2EHT1|H2EHT1_HUMAN Cellular tumor antigen p53 OS=Homo sapiens OX=9606 GN=TP53 PE=2 SV=1"

# =========================
# IO
# =========================
INPUT_FILE = Path("../Data/clinvar_tp53.txt")   # הקובץ שצירפת
OUTPUT_FASTA = Path("../Data/tp53_clinvar_labeled.fasta")

# =========================
# Helpers
# =========================
MISSENSE_TOKEN_RE = re.compile(r"\b([A-Z])(\d+)([A-Z])\b")

def normalize_ref_sequence(wt: str) -> str:
    if not TRIM_TO_MDDL:
        return wt
    idx = wt.find(TRIM_ANCHOR)
    if idx == -1:
        raise ValueError(f"TRIM_ANCHOR not found in WT_SEQUENCE: {TRIM_ANCHOR}")
    return wt[idx:]  # מתחיל מ-MDDL...

def wrap_fasta(seq: str, width: int = 60) -> str:
    return "\n".join(seq[i:i+width] for i in range(0, len(seq), width))

def label_from_classification(classification: str) -> str:
    # לדוגמה: "Likely benign" -> "Likely_benign"
    return classification.strip().replace(" ", "_")

def pick_best_missense_token(protein_change_field: str, ref_seq: str):
    """
    protein_change_field יכול להיות: "D234Y, D261Y, D393Y, D354Y"
    נחזיר את הטוקן הראשון שמתאים גם לרפרנס (אות WT בעמדה = refAA).
    אם אין התאמה, נחזיר את הראשון שמצאנו.
    """
    tokens = [t.strip() for t in protein_change_field.split(",")]
    parsed = []
    for t in tokens:
        m = MISSENSE_TOKEN_RE.search(t)
        if m:
            refAA, pos, altAA = m.group(1), int(m.group(2)), m.group(3)
            parsed.append((t, refAA, pos, altAA))

    if not parsed:
        return None

    # נסי למצוא טוקן שמתאים לרפרנס (בדיקת sanity)
    for t, refAA, pos, altAA in parsed:
        idx = pos - 1
        if 0 <= idx < len(ref_seq) and ref_seq[idx] == refAA:
            return t, refAA, pos, altAA

    # אחרת: קחי את הראשון
    t, refAA, pos, altAA = parsed[0]
    return t, refAA, pos, altAA

def apply_single_aa_substitution(ref_seq: str, pos: int, altAA: str) -> str:
    idx = pos - 1
    if not (0 <= idx < len(ref_seq)):
        return None
    return ref_seq[:idx] + altAA + ref_seq[idx+1:]

# =========================
# Main
# =========================
ref_seq = normalize_ref_sequence(WT_SEQUENCE)

written = 0
skipped = 0

with INPUT_FILE.open(newline="", encoding="utf-8") as f_in, OUTPUT_FASTA.open("w", encoding="utf-8") as f_out:
    reader = csv.DictReader(f_in, delimiter="\t")

    # 1) כתיבת WT ראשון
    f_out.write(WT_HEADER + "\n")
    f_out.write(wrap_fasta(ref_seq) + "\n")

    # 2) כתיבת וריאנטים
    for row in reader:
        classification = (row.get("Germline classification") or "").strip()
        protein_change_field = (row.get("Protein change") or "").strip()

        if not classification or not protein_change_field:
            skipped += 1
            continue

        pick = pick_best_missense_token(protein_change_field, ref_seq)
        if pick is None:
            skipped += 1
            continue

        token, refAA, pos, altAA = pick

        # מייצר רצף מוטנטי
        mutant = apply_single_aa_substitution(ref_seq, pos, altAA)
        if mutant is None:
            skipped += 1
            continue

        # כותרת בפורמט שביקשת: TP53_F346L|label=Likely_benign
        label = label_from_classification(classification)
        header = f">TP53_{token}|label={label}"

        f_out.write(header + "\n")
        f_out.write(wrap_fasta(mutant) + "\n")

        written += 1

print("Done.")
print("Output:", OUTPUT_FASTA.resolve())
print("Variants written:", written)
print("Rows skipped:", skipped)
