import numpy as np
import pandas as pd
from Bio import AlignIO
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import roc_curve, auc
import re
from Bio.SeqUtils import seq1

# --- הגדרות ---
# MSA_FILE = "tp53_msa.fasta"
# MSA_FILE = "tp53_30_msa.fasta"
MSA_FILE = "tp53_random_30_aligned.fasta"

CLINVAR_FILE = "clinvar_tp53.txt"
AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
AA_TO_INDEX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}


# class EvolutionaryModel:
#     def __init__(self):
#         self.emission_probs = None
#         self.match_cols = []
#         self.human_map = {}
#         self.optimal_threshold = None  # נלמד את זה בהמשך
#
#     def fit(self, msa_file):
#         """
#         שלב האימון: לומד את התפלגות חומצות האמינו מהאבולוציה
#         """
#         print(f"Training model on {msa_file}...")
#         alignment = AlignIO.read(msa_file, "fasta")
#
#         # 1. זיהוי עמודות ליבה
#         n_seqs = len(alignment)
#         threshold = 0.5
#         self.match_cols = []
#         for i in range(alignment.get_alignment_length()):
#             if alignment[:, i].count("-") / n_seqs < threshold:
#                 self.match_cols.append(i)
#
#         # 2. חישוב מטריצת הפליטה (Emissions)
#         emission_counts = np.zeros((len(self.match_cols), 20)) + 1  # Pseudocount
#         for m_idx, col_idx in enumerate(self.match_cols):
#             col = alignment[:, col_idx].upper()
#             for char in col:
#                 if char in AA_TO_INDEX:
#                     emission_counts[m_idx, AA_TO_INDEX[char]] += 1
#
#         # נרמול ללוג
#         row_sums = emission_counts.sum(axis=1, keepdims=True)
#         self.emission_probs = np.log(emission_counts / row_sums)
#
#         # 3. מיפוי לרצף האנושי (כדי שנוכל לחזות עליו)
#         self._map_human_sequence(alignment)
#         print("✅ Training complete.")
#
#     def _map_human_sequence(self, alignment):
#         """פונקציית עזר פנימית למיפוי אינדקסים"""
#         human_seq = None
#         for record in alignment:
#             if "HUMAN" in record.id.upper() or "HOMO" in record.description.upper():
#                 human_seq = str(record.seq)
#                 break
#         if not human_seq: human_seq = str(alignment[0].seq)
#
#         self.human_map = {}
#         curr_human = 1
#         curr_hmm = 0
#         for i, char in enumerate(human_seq):
#             is_match = (i in self.match_cols)
#             if char != "-":
#                 if is_match:
#                     self.human_map[curr_human] = curr_hmm
#                     curr_hmm += 1
#                 curr_human += 1
#             elif is_match:
#                 curr_hmm += 1
#
#     def predict_score(self, position, new_aa):
#         """
#         שלב החיזוי (Score Only): מחזיר ציון למוטציה ספציפית
#         """
#         if position not in self.human_map:
#             return None  # עמדה מחוץ למודל
#
#         hmm_idx = self.human_map[position]
#         aa_idx = AA_TO_INDEX.get(new_aa)
#
#         if aa_idx is None: return None
#
#         # שליפת הציון מהמטריצה שנלמדה
#         return self.emission_probs[hmm_idx, aa_idx]
#
#     def set_optimal_threshold(self, y_true, y_scores):
#         """
#         חישוב הסף האופטימלי שנותן את האיזון הכי טוב בין רגישות לספציפיות
#         """
#         fpr, tpr, thresholds = roc_curve(y_true, [-x for x in y_scores])  # היפוך סימן
#         # Youden's J statistic = TPR - FPR
#         optimal_idx = np.argmax(tpr - fpr)
#         self.optimal_threshold = -thresholds[optimal_idx]  # החזרה לסימן המקורי
#
#         print(f"🎯 Optimal Cutoff Score found: {self.optimal_threshold:.2f}")
#         return self.optimal_threshold
#
#     def predict_class(self, score):
#         """
#         מחזיר תשובה סופית: 'Pathogenic' או 'Benign' לפי הסף
#         """
#         if self.optimal_threshold is None:
#             return "Unknown (Threshold not set)"
#
#         if score < self.optimal_threshold:
#             return "Pathogenic"
#         else:
#             return "Benign"
#


class EvolutionaryModel:
    def __init__(self):
        self.emission_probs = None
        self.consensus_scores = None  # נשמור את הציון "הכי טוב" לכל עמודה
        self.match_cols = []
        self.human_map = {}
        self.optimal_threshold = None

    def fit(self, msa_file):
        """אימון המודל + חישוב הקונצנזוס"""
        print(f"Training model on {msa_file}...")
        alignment = AlignIO.read(msa_file, "fasta")

        # 1. זיהוי עמודות ליבה
        n_seqs = len(alignment)
        threshold = 0.5
        self.match_cols = []
        for i in range(alignment.get_alignment_length()):
            if alignment[:, i].count("-") / n_seqs < threshold:
                self.match_cols.append(i)

        # 2. חישוב מטריצת הפליטה
        emission_counts = np.zeros((len(self.match_cols), 20)) + 1
        for m_idx, col_idx in enumerate(self.match_cols):
            col = alignment[:, col_idx].upper()
            for char in col:
                if char in AA_TO_INDEX:
                    emission_counts[m_idx, AA_TO_INDEX[char]] += 1

        row_sums = emission_counts.sum(axis=1, keepdims=True)
        self.emission_probs = np.log(emission_counts / row_sums)

        # --- השיפור: שמירת הציון המקסימלי לכל עמודה ---
        # זה מייצג את "האידיאל האבולוציוני" בעמדה זו
        self.consensus_scores = np.max(self.emission_probs, axis=1)

        # 3. מיפוי
        self._map_human_sequence(alignment)
        print("✅ Training complete (Normalized by Consensus).")

    def _map_human_sequence(self, alignment):
        human_seq = None
        for record in alignment:
            if "HUMAN" in record.id.upper() or "HOMO" in record.description.upper():
                human_seq = str(record.seq)
                break
        if not human_seq: human_seq = str(alignment[0].seq)

        self.human_map = {}
        curr_human = 1
        curr_hmm = 0
        for i, char in enumerate(human_seq):
            is_match = (i in self.match_cols)
            if char != "-":
                if is_match:
                    self.human_map[curr_human] = curr_hmm
                    curr_hmm += 1
                curr_human += 1
            elif is_match:
                curr_hmm += 1

    def predict_score(self, position, new_aa):
        """
        חישוב הציון החדש: המרחק מהקונצנזוס
        """
        if position not in self.human_map:
            return None

        hmm_idx = self.human_map[position]
        aa_idx = AA_TO_INDEX.get(new_aa)

        if aa_idx is None: return None

        # 1. הציון של המוטציה
        mutant_score = self.emission_probs[hmm_idx, aa_idx]

        # 2. הציון של האות הכי טובה בעמודה זו (Consensus)
        consensus_score = self.consensus_scores[hmm_idx]

        # 3. הציון הסופי הוא ההפרש (תמיד שלילי או אפס)
        # ככל שהמספר נמוך יותר (שלילי יותר), כך המוטציה "רחוקה יותר מהאידיאל"
        normalized_score = mutant_score - consensus_score

        return normalized_score

    def set_optimal_threshold(self, y_true, y_scores):
        fpr, tpr, thresholds = roc_curve(y_true, [-x for x in y_scores])
        optimal_idx = np.argmax(tpr - fpr)
        self.optimal_threshold = -thresholds[optimal_idx]
        print(f"🎯 Optimal Cutoff Score found: {self.optimal_threshold:.2f}")
        return self.optimal_threshold

    def predict_class(self, score):
        if self.optimal_threshold is None: return "Unknown"
        if score < self.optimal_threshold:
            return "Pathogenic"
        else:
            return "Benign"
# --- Main Logic ---

# 1. יצירת המודל ואימון
model = EvolutionaryModel()
model.fit(MSA_FILE)

# 2. טעינת נתונים למבחן (ClinVar)
print("Loading ClinVar for evaluation...")
try:
    df = pd.read_csv(CLINVAR_FILE, sep='\t')
    if df.shape[1] < 2: df = pd.read_csv(CLINVAR_FILE)
except:
    print("Error reading ClinVar.")
    exit()

# זיהוי עמודות (הקוד החכם שלך)
name_col = next((c for c in df.columns if c in ['Name', 'ProteinChange', 'HGVS.p']), df.columns[0])
sig_col = next((c for c in df.columns if 'ClinicalSignificance' in c or 'Germline' in c or 'Significance' in c), None)

scores_pathogenic = []
scores_benign = []

# 3. הרצת חיזויים על הדאטה
print("Predicting scores...")
for idx, row in df.iterrows():
    try:
        if pd.isna(row[name_col]) or pd.isna(row[sig_col]): continue

        val = str(row[name_col])
        m = re.search(r'p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})', val)
        if not m: continue

        pos = int(m.group(2))

        # המרה מ-Arg ל-R
        aa_3_letter = m.group(3)
        new_aa = seq1(aa_3_letter)

        # --- כאן מתבצע החיזוי ---
        score = model.predict_score(pos, new_aa)

        if score is not None:
            sig = str(row[sig_col]).lower()
            if 'pathogenic' in sig and 'benign' not in sig:
                scores_pathogenic.append(score)
            elif 'benign' in sig and 'pathogenic' not in sig:
                scores_benign.append(score)
    except:
        continue

# 4. מציאת הסף האופטימלי (Fine Tuning)
y_true = [1] * len(scores_pathogenic) + [0] * len(scores_benign)
y_scores = scores_pathogenic + scores_benign
cutoff = model.set_optimal_threshold(y_true, y_scores)

# 5. דוגמה לשימוש בחיזוי בזמן אמת (למצגת)
print("-" * 30)
print("🩺 Real-time Prediction Example:")
test_mutation_pos = 175
test_mutation_aa = 'H'  # Arg175His (המפורסמת)
my_score = model.predict_score(test_mutation_pos, test_mutation_aa)
diagnosis = model.predict_class(my_score)

print(f"Mutation: p.Arg{test_mutation_pos}His")
print(f"Model Score: {my_score:.2f}")
print(f"Model Diagnosis: {diagnosis} (Cutoff: {cutoff:.2f})")
print("-" * 30)

# 6. ציור ROC סופי
fpr, tpr, _ = roc_curve(y_true, [-x for x in y_scores])
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.title(f'Model Performance (AUC={roc_auc:.2f})')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.savefig("final_optimized_roc.png")
print(f"Final AUC: {roc_auc:.3f}")