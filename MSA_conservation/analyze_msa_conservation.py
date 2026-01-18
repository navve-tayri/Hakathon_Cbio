import numpy as np
import pandas as pd
from Bio import AlignIO
import matplotlib.pyplot as plt
import seaborn as sns

# --- Constants ---
MSA_FILE = "../Data/tp53_msa.fasta"
AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
AA_TO_INDEX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}


# --- Functions ---
def load_msa(filename):
    alignment = AlignIO.read(filename, "fasta")
    print(f"Loaded MSA with {len(alignment)} sequences and {alignment.get_alignment_length()} columns.")
    return alignment


def get_valid_columns(alignment, gap_threshold=0.5):
    n_seqs = len(alignment)
    n_cols = alignment.get_alignment_length()
    valid_indices = []
    for i in range(n_cols):
        column = alignment[:, i]
        if column.count("-") / n_seqs < gap_threshold:
            valid_indices.append(i)
    print(f"Using {len(valid_indices)} valid columns out of {n_cols}.")
    return valid_indices


def get_column_probabilities(alignment, valid_indices):
    """Computes frequency (probability) of each AA in each column with pseudocounts."""
    n_cols = len(valid_indices)
    n_aa = len(AMINO_ACIDS)
    counts = np.zeros((n_cols, n_aa))

    for matrix_idx, align_col_idx in enumerate(valid_indices):
        column = alignment[:, align_col_idx]
        for char in column:
            char = char.upper()
            if char in AA_TO_INDEX:
                counts[matrix_idx, AA_TO_INDEX[char]] += 1

    # Add Pseudocounts and normalize
    counts += 1
    row_sums = counts.sum(axis=1, keepdims=True)
    probs = counts / row_sums
    return probs


def calculate_pssm(probs):
    """Calculates PSSM (Log-Likelihood) from probabilities."""
    return np.log(probs)


def calculate_shannon_entropy(probs):
    """Calculates Shannon Entropy (in bits) for each column."""
    # p * log2(p)
    p_log_p = probs * np.log2(probs)
    # Sum and negate
    entropy = -np.sum(p_log_p, axis=1)
    return entropy


# --- Main Execution ---

# 1. Calculations
aln = load_msa(MSA_FILE)
valid_cols = get_valid_columns(aln)
probs_matrix = get_column_probabilities(aln, valid_cols)

pssm_matrix = calculate_pssm(probs_matrix)
entropy_scores = calculate_shannon_entropy(probs_matrix)

# --- Visualization 1: Shannon Entropy Plot ---
print("Generating Entropy Plot...")
plt.figure(figsize=(15, 4))  # Shorter height for line plot

plt.plot(entropy_scores, color='black', linewidth=1.5)
plt.fill_between(range(len(entropy_scores)), entropy_scores, color='gray', alpha=0.3)

plt.title("Sequence Conservation (Shannon Entropy)")
plt.ylabel("Entropy (Bits)")
plt.xlabel("Alignment Column Index (Valid Columns Only)")
plt.grid(True, linestyle='--', alpha=0.6, which='both')

# Add explanatory text box
plt.text(0.01, 0.85, "Lower Value = Higher Conservation\nZero = Perfectly Conserved",
         transform=plt.gca().transAxes, fontsize=10,
         bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

entropy_img = "conservation_entropy.png"
plt.savefig(entropy_img, dpi=300, bbox_inches='tight')
plt.close()
print(f"✅ Saved: {entropy_img}")

# --- Visualization 2: PSSM Heatmap ---
print("Generating PSSM Heatmap...")
plt.figure(figsize=(15, 7))  # Taller height for heatmap

sns.heatmap(pssm_matrix.T, cmap="viridis", yticklabels=AMINO_ACIDS, cbar_kws={'label': 'Log Probability'})

plt.title("PSSM (Position-Specific Scoring Matrix)")
plt.ylabel("Amino Acid")
plt.xlabel("Alignment Column Index (Valid Columns Only)")

pssm_img = "conservation_pssm.png"
plt.savefig(pssm_img, dpi=300, bbox_inches='tight')
plt.close()
print(f"✅ Saved: {pssm_img}")

print("\nDone!")