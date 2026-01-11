import numpy as np
import pandas as pd
from Bio import AlignIO
import matplotlib.pyplot as plt
import seaborn as sns

# --- הגדרות ---
MSA_FILE = "tp53_msa.fasta"
AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")  # 20 חומצות האמינו
AA_TO_INDEX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}


def load_msa(filename):
    """טוען את הקובץ וממיר למטריצה של תווים"""
    alignment = AlignIO.read(filename, "fasta")
    print(f"Loaded MSA with {len(alignment)} sequences and {alignment.get_alignment_length()} columns.")
    return alignment


def identify_match_columns(alignment, threshold=0.5):
    """
    מזהה אילו עמודות הן Match ואילו Insert.
    ההיגיון: אם בעמודה יש פחות מ-50% רווחים, היא Match.
    """
    n_seqs = len(alignment)
    n_cols = alignment.get_alignment_length()
    match_cols = []

    for i in range(n_cols):
        column = alignment[:, i]
        gap_count = column.count("-")
        if gap_count / n_seqs < threshold:
            match_cols.append(i)

    print(f"Identified {len(match_cols)} Match states out of {n_cols} alignment columns.")
    return match_cols


def calculate_emission_matrix(alignment, match_cols):
    """
    מחשב את ההסתברות (בלוג) לכל חומצה בכל עמודת Match.
    כולל Laplace Smoothing (Pseudocounts).
    """
    n_matches = len(match_cols)
    n_aa = len(AMINO_ACIDS)

    # מטריצה בגודל: [מספר עמודות הליבה] X [מספר חומצות האמינו]
    emission_counts = np.zeros((n_matches, n_aa))

    # 1. ספירה (Counting)
    for m_idx, col_idx in enumerate(match_cols):
        column = alignment[:, col_idx]
        for char in column:
            char = char.upper()
            if char in AA_TO_INDEX:
                emission_counts[m_idx, AA_TO_INDEX[char]] += 1

    # 2. הוספת Pseudocounts (כדי למנוע אפסים)
    emission_counts += 1

    # 3. נרמול להסתברויות (Probability)
    row_sums = emission_counts.sum(axis=1, keepdims=True)
    emission_probs = emission_counts / row_sums

    # 4. המרה ל-Log Space (כדי למנוע Underflow בהמשך)
    emission_log_probs = np.log(emission_probs)

    return emission_log_probs


# --- הרצה ---

# 1. טעינה
aln = load_msa(MSA_FILE)

# 2. זיהוי עמודות הליבה (המבנה של החלבון)
match_indices = identify_match_columns(aln)

# 3. חישוב מטריצת הפליטה
emission_matrix = calculate_emission_matrix(aln, match_indices)

# 4. ויזואליזציה (כדי שנראה שהכל הגיוני)
plt.figure(figsize=(15, 6))
sns.heatmap(emission_matrix.T, cmap="viridis", yticklabels=AMINO_ACIDS)
plt.title("HMM Emission Probabilities (Log Space)\nYellow = High Probability (Conserved), Blue = Low Probability")
plt.xlabel("HMM Match State Index")
plt.ylabel("Amino Acid")

output_img = "hmm_heatmap.png"
plt.savefig(output_img, dpi=300, bbox_inches='tight')
plt.close() # סוגר את הזיכרון הגרפי

print(f"✅ Success! Plot saved to '{output_img}'")
print("✅ Emission Matrix Built Successfully!")
print(f"Matrix Shape: {emission_matrix.shape}")