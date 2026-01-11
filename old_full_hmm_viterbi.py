import numpy as np
import pandas as pd
from Bio import AlignIO
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import roc_curve, auc
import sys
import re

# --- הגדרות ---
MSA_FILE = "tp53_msa.fasta"
CLINVAR_FILE = "clinvar_tp53.txt"
AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
AA_TO_INDEX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}


class ProfileHMM:
    def __init__(self, msa_file):
        print(f"Loading MSA from {msa_file}...")
        self.msa = AlignIO.read(msa_file, "fasta")
        self.match_cols = self._identify_match_columns()
        self.L = len(self.match_cols)
        self.n_aa = len(AMINO_ACIDS)

        self.E = None
        self.T = None

        print(f"Model Length (L): {self.L} match states")
        self._train()

    def _identify_match_columns(self, threshold=0.5):
        n_seqs = len(self.msa)
        match_cols = []
        for i in range(self.msa.get_alignment_length()):
            gap_count = self.msa[:, i].count("-")
            if gap_count / n_seqs < threshold:
                match_cols.append(i)
        return match_cols

    def _train(self):
        print("Training HMM (Calculating Emissions & Transitions)...")
        counts_E = np.zeros((self.L, self.n_aa)) + 1
        counts_T = np.zeros((self.L - 1, 2)) + 1

        for record in self.msa:
            seq = str(record.seq).upper()
            for k, col_idx in enumerate(self.match_cols):
                aa = seq[col_idx]
                if aa in AA_TO_INDEX:
                    counts_E[k, AA_TO_INDEX[aa]] += 1

            for k in range(self.L - 1):
                col_curr = self.match_cols[k]
                col_next = self.match_cols[k + 1]
                if seq[col_next] == "-":
                    counts_T[k, 1] += 1
                else:
                    counts_T[k, 0] += 1

        self.E = np.log(counts_E / counts_E.sum(axis=1, keepdims=True))
        self.T = np.log(counts_T / counts_T.sum(axis=1, keepdims=True))
        print("Training complete.")

    def get_human_mapping(self):
        human_seq = None
        for rec in self.msa:
            if "HUMAN" in rec.id.upper() or "HOMO" in rec.description.upper():
                human_seq = str(rec.seq)
                break
        if not human_seq:
            human_seq = str(self.msa[0].seq)
            print("Warning: Using first sequence as reference (Human not found).")

        human_to_hmm_map = {}
        curr_hmm = 0
        curr_human = 1
        for i, char in enumerate(human_seq):
            if i in self.match_cols:
                if char != "-":
                    human_to_hmm_map[curr_human] = curr_hmm
                    curr_human += 1
                curr_hmm += 1
            elif char != "-":
                curr_human += 1
        return human_to_hmm_map


# --- פונקציה ראשית לניתוח ---

def run_analysis():
    # 1. בניית המודל
    hmm = ProfileHMM(MSA_FILE)
    human_map = hmm.get_human_mapping()

    # 2. טעינת ClinVar (מנגנון חכם לזיהוי עמודות)
    print(f"Loading ClinVar from {CLINVAR_FILE}...")
    try:
        df = pd.read_csv(CLINVAR_FILE, sep='\t')
        if df.shape[1] < 2:
            df = pd.read_csv(CLINVAR_FILE)
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    print("Columns found:", list(df.columns))

    # זיהוי שמות העמודות
    sig_col = None
    possible_sig = ['ClinicalSignificance', 'Clinical significance (Last reviewed)', 'Germline classification', 'Class',
                    'Significance']
    for col in df.columns:
        for p in possible_sig:
            if p.lower() in col.lower():
                sig_col = col
                break
        if sig_col: break

    name_col = None
    possible_name = ['Name', 'ProteinChange', 'HGVS.p', 'Mutant']
    for col in df.columns:
        if col in possible_name:
            name_col = col
            break
    if not name_col: name_col = df.columns[0]  # ברירת מחדל

    if not sig_col:
        print("❌ Error: Could not find 'ClinicalSignificance' column. Please check the file.")
        return

    print(f"Using columns: '{name_col}' (Mutation) and '{sig_col}' (Class)")

    # 3. חישוב ציונים
    scores_pathogenic = []
    scores_benign = []

    print("Scoring mutations...")
    for idx, row in df.iterrows():
        try:
            if pd.isna(row[name_col]) or pd.isna(row[sig_col]): continue

            val = str(row[name_col])
            # Regex to find p.Arg175His
            m = re.search(r'p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})', val)
            if not m: continue

            pos = int(m.group(2))
            new_aa_3 = m.group(3)
            from Bio.SeqUtils import seq1
            new_aa = seq1(new_aa_3)  # המרה ל-H

            if pos not in human_map: continue
            hmm_idx = human_map[pos]

            # --- חישוב הציון המשוקלל (Emissions + Transitions) ---
            if new_aa in AA_TO_INDEX:
                # 1. פליטה
                mutation_score = hmm.E[hmm_idx, AA_TO_INDEX[new_aa]]

                # 2. מעברים (Local Context)
                if hmm_idx < hmm.L - 1:
                    mutation_score += hmm.T[hmm_idx, 0]  # Match->Match prob

                # סיווג
                sig = str(row[sig_col]).lower()
                if 'pathogenic' in sig and 'benign' not in sig and 'conflicting' not in sig:
                    scores_pathogenic.append(mutation_score)
                elif 'benign' in sig and 'pathogenic' not in sig and 'conflicting' not in sig:
                    scores_benign.append(mutation_score)
        except Exception as e:
            continue

    print(f"Scored {len(scores_pathogenic)} Pathogenic, {len(scores_benign)} Benign.")

    # 4. יצירת גרף ROC
    if len(scores_pathogenic) > 0 and len(scores_benign) > 0:
        y_true = [1] * len(scores_pathogenic) + [0] * len(scores_benign)
        y_scores = scores_pathogenic + scores_benign
        y_scores_inv = [-x for x in y_scores]  # היפוך סימן

        fpr, tpr, _ = roc_curve(y_true, y_scores_inv)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Full Profile HMM (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.title('ROC Curve - Full Profile HMM (Emissions + Transitions)')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig("roc_full_hmm.png")
        print(f"✅ Success! ROC Curve saved to 'roc_full_hmm.png'. AUC: {roc_auc:.3f}")
    else:
        print("❌ Not enough data found to plot results.")


if __name__ == "__main__":
    run_analysis()