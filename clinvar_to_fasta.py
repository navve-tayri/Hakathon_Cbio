import csv
import re

# =========================
# Reference WT sequence
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

# =========================
# Helpers
# =========================

AA3_TO_AA1 = {
    "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C",
    "Glu": "E", "Gln": "Q", "Gly": "G", "His": "H", "Ile": "I",
    "Leu": "L", "Lys": "K", "Met": "M", "Phe": "F", "Pro": "P",
    "Ser": "S", "Thr": "T", "Trp": "W", "Tyr": "Y", "Val": "V",
    "Ter": "*"
}

def parse_protein_change(name_field):
    """
    Extract something like: p.Arg337His
    Returns: (refAA, position, altAA) in 1-letter code
    """
    m = re.search(r"p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})", name_field)
    if not m:
        return None

    ref3, pos, alt3 = m.groups()
    if ref3 not in AA3_TO_AA1 or alt3 not in AA3_TO_AA1:
        return None

    ref = AA3_TO_AA1[ref3]
    alt = AA3_TO_AA1[alt3]
    pos = int(pos)

    return ref, pos, alt

def apply_mutation(wt_seq, ref, pos, alt):
    """
    pos is 1-based
    """
    idx = pos - 1
    if idx < 0 or idx >= len(wt_seq):
        return None

    if wt_seq[idx] != ref:
        # sanity check failed
        # still allow, but warn
        print(f"Warning: WT mismatch at {pos}: expected {ref}, found {wt_seq[idx]}")

    return wt_seq[:idx] + alt + wt_seq[idx+1:]

def classify_label(classification):
    c = classification.lower()

    if "pathogenic" in c:
        return "PATHOGENIC"
    if "benign" in c:
        return "BENIGN"
    return None  # unknown / skip

# =========================
# Main conversion
# =========================

INPUT_FILE = "clinvar_tp53.txt"   # תשני לשם הקובץ שלך
OUTPUT_FASTA = "tp53_variants.fasta"

count_written = 0
count_skipped = 0

with open(INPUT_FILE, newline="", encoding="utf-8") as f, open(OUTPUT_FASTA, "w") as out:
    reader = csv.DictReader(f, delimiter="\t")

    for row in reader:
        name = row["Name"]
        classification = row["Germline classification"]

        label = classify_label(classification)
        if label is None:
            count_skipped += 1
            continue

        parsed = parse_protein_change(name)
        if parsed is None:
            count_skipped += 1
            continue

        ref, pos, alt = parsed

        mutant_seq = apply_mutation(WT_SEQUENCE, ref, pos, alt)
        if mutant_seq is None:
            count_skipped += 1
            continue

        header = f">TP53_p.{ref}{pos}{alt}|{label}"
        out.write(header + "\n")

        # write sequence in 60 char lines
        for i in range(0, len(mutant_seq), 60):
            out.write(mutant_seq[i:i+60] + "\n")

        count_written += 1

print("Done.")
print("Written sequences:", count_written)
print("Skipped rows:", count_skipped)
print("Output file:", OUTPUT_FASTA)
