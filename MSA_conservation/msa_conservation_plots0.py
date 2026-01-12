import numpy as np
import pandas as pd
from Bio import AlignIO
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

# --- Constants ---
MSA_FILE = "tp53_msa.fasta"
AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
AA_TO_INDEX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}


# --- Functions ---
def load_msa(filename):
    try:
        alignment = AlignIO.read(filename, "fasta")
        print(f"Loaded MSA with {len(alignment)} sequences and {alignment.get_alignment_length()} columns.")
        return alignment
    except FileNotFoundError:
        print(f"Error: {filename} not found. Please ensure the file is in the directory.")
        return None


def get_valid_columns(alignment, gap_threshold=0.5):
    n_seqs = len(alignment)
    n_cols = alignment.get_alignment_length()
    valid_indices = []
    for i in range(n_cols):
        column = alignment[:, i]
        if column.count("-") / n_seqs < gap_threshold:
            valid_indices.append(i)
    return valid_indices


def get_column_probabilities(alignment, valid_indices):
    n_cols = len(valid_indices)
    n_aa = len(AMINO_ACIDS)
    counts = np.zeros((n_cols, n_aa))

    for matrix_idx, align_col_idx in enumerate(valid_indices):
        column = alignment[:, align_col_idx]
        for char in column:
            char = char.upper()
            if char in AA_TO_INDEX:
                counts[matrix_idx, AA_TO_INDEX[char]] += 1

    counts += 1  # Pseudocounts
    row_sums = counts.sum(axis=1, keepdims=True)
    probs = counts / row_sums
    return probs


def calculate_shannon_entropy(probs):
    p_log_p = probs * np.log2(probs)
    entropy = -np.sum(p_log_p, axis=1)
    return entropy


# --- Main Execution ---

aln = load_msa(MSA_FILE)

if aln:
    valid_cols = get_valid_columns(aln)
    probs_matrix = get_column_probabilities(aln, valid_cols)
    entropy_scores = calculate_shannon_entropy(probs_matrix)

    # ==========================================
    # VISUALIZATION 1: Zoomed-in Line Plot (Columns 200-250)
    # ==========================================
    print("Generating Zoomed-in Plot with Annotations...")

    start_col, end_col = 200, 250

    max_len = len(entropy_scores)
    if end_col > max_len: end_col = max_len
    if start_col >= max_len: start_col = max_len - 50 if max_len > 50 else 0

    subset_scores = entropy_scores[start_col:end_col]
    x_indices = np.arange(start_col, end_col)

    # --- מציאת נקודות הקיצון לשימוש בשני הגרפים ---
    local_min_rel_idx = np.argmin(subset_scores)
    local_max_rel_idx = np.argmax(subset_scores)

    # אינדקסים אמיתיים (בתוך הטווח 200-250)
    local_min_col = x_indices[local_min_rel_idx]
    local_min_val = subset_scores[local_min_rel_idx]

    local_max_col = x_indices[local_max_rel_idx]
    local_max_val = subset_scores[local_max_rel_idx]

    # --- יצירת גרף הזום ---
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(x_indices, subset_scores, color='#333333', linewidth=2, alpha=0.9)
    ax.fill_between(x_indices, subset_scores, color='skyblue', alpha=0.5)
    ax.scatter(x_indices, subset_scores, color='#333333', s=20, alpha=0.6)

    # הגדרת גבול עליון לזום
    y_zoom_limit = max(subset_scores) * 1.4
    ax.set_ylim(0, y_zoom_limit)

    # כיתוב לנקודה הנמוכה (Conserved)
    ax.annotate('Conserved Region',
                xy=(local_min_col, local_min_val),
                xytext=(local_min_col, local_min_val + 0.5),
                arrowprops=dict(facecolor='darkred', shrink=0.05, width=1),
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="darkred", alpha=0.9),
                fontsize=10, color='darkred', ha='center', fontweight='bold')

    # כיתוב לנקודה הגבוהה (Variable)
    ax.annotate('Variable Region',
                xy=(local_max_col, local_max_val),
                xytext=(local_max_col, local_max_val + 0.3),
                arrowprops=dict(facecolor='darkgreen', shrink=0.05, width=1),
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="darkgreen", alpha=0.9),
                fontsize=10, color='darkgreen', ha='center')

    ax.set_title(f"TP53 Conservation Detail (Columns {start_col}-{end_col})", fontsize=16)
    ax.set_ylabel("Entropy (Bits)", fontsize=12)
    ax.set_xlabel("Alignment Column Index", fontsize=12)

    if len(x_indices) > 0:
        ax.set_xlim(x_indices[0] - 1, x_indices[-1] + 1)

    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

    plt.tight_layout()
    plt.savefig("tp53_zoomed_conservation_annotated.png", dpi=300)
    plt.close()
    print("✅ Saved: tp53_zoomed_conservation_annotated.png")


    # ==========================================
    # VISUALIZATION 2: Full Graph (Pointing to the SAME Zoom Points)
    # ==========================================
    print("Generating Full Annotated Plot...")

    plt.figure(figsize=(15, 6))

    plt.plot(entropy_scores, color='#333333', linewidth=1, alpha=0.8)
    plt.fill_between(range(len(entropy_scores)), entropy_scores, color='skyblue', alpha=0.4)

    # חישוב גבול עליון גלובלי
    global_max_val = np.max(entropy_scores)
    y_top_limit = global_max_val * 1.35
    plt.ylim(0, y_top_limit)

    # --- שינוי כאן ---
    # Annotation: Pointing to the specific Conserved point (Red)
    # הזזנו את הטקסט שמאלה
    plt.annotate('Conserved Region\n(From Zoom)',
                 xy=(local_min_col, local_min_val),
                 # xytext X index moved 60 positions left:
                 xytext=(local_min_col - 60, global_max_val * 0.8),
                 arrowprops=dict(facecolor='darkred', shrink=0.05, width=1),
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="darkred", alpha=0.9),
                 # Changed alignment to right:
                 fontsize=10, color='darkred', ha='right', fontweight='bold')

    # Annotation: Pointing to the specific Variable point (Green) - ללא שינוי
    plt.annotate('Variable Region\n(From Zoom)',
                 xy=(local_max_col, local_max_val),
                 xytext=(local_max_col, local_max_val + 1.0),
                 arrowprops=dict(facecolor='darkgreen', shrink=0.05, width=1),
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="darkgreen", alpha=0.9),
                 fontsize=10, color='darkgreen', ha='center')

    plt.title("TP53 Sequence Conservation (Full Gene)", fontsize=16)
    plt.ylabel("Entropy (Bits)", fontsize=12)
    plt.xlabel("Alignment Column Index", fontsize=12)

    plt.tight_layout()
    plt.savefig("tp53_full_conservation.png", dpi=300)
    plt.close()
    print("✅ Saved: tp53_full_conservation.png")

