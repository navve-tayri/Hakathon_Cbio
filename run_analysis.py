import pandas as pd
import numpy as np
from Bio import AlignIO
import matplotlib.pyplot as plt
import seaborn as sns
import re

from scipy import stats
from sklearn.metrics import roc_curve, auc

# --- קבצים ---
MSA_FILE = "tp53_msa.fasta"
CLINVAR_FILE = "clinvar_tp53.txt"  # וודאי שזה השם הנכון

# --- הגדרות ---
AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
AA_TO_INDEX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}


def get_profile_hmm_score(seq, emission_matrix, match_cols_indices):
    """
    מחשב ציון פשוט (Log-Odds) לרצף בהינתן מטריצת הפליטה.
    (בגרסה פשוטה זו להאקתון, אנחנו מתמקדים בפליטה כי היא הדומיננטית ב-Missense)
    """
    score = 0
    # אנחנו רצים רק על העמודות שנחשבות Match States
    for i, m_idx in enumerate(match_cols_indices):
        if i >= len(seq): break  # הגנה מחריגה

        aa = seq[i]
        if aa in AA_TO_INDEX:
            # הוספת הניקוד (Log Probability) מהמטריצה
            score += emission_matrix[i, AA_TO_INDEX[aa]]
        else:
            # קנס קטן על אותיות לא ידועות או רווחים ברצף המטרה
            score += -5.0
    return score


def parse_clinvar_mutation(protein_change):
    """
    מפענח מחרוזת כמו 'p.Arg175His' ומחזיר: (עמדה מקורית, אות ישנה, אות חדשה)
    """
    try:
        # שימוש בביטוי רגולרי למצוא את המבנה: p.Ala123Val
        match = re.search(r'p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})', str(protein_change))
        if match:
            old_aa_3 = match.group(1)
            pos = int(match.group(2))
            new_aa_3 = match.group(3)

            # המרה מ-3 אותיות לאות אחת (Ala -> A)
            from Bio.SeqUtils import seq1
            old_aa = seq1(old_aa_3)
            new_aa = seq1(new_aa_3)

            return pos, old_aa, new_aa
    except:
        pass
    return None, None, None


# 1. טעינת ה-MSA ובניית המטריצה (כמו שעשינו קודם)
print("Loading MSA and building model...")
alignment = AlignIO.read(MSA_FILE, "fasta")
match_cols = []
threshold = 0.5
for i in range(alignment.get_alignment_length()):
    if alignment[:, i].count("-") / len(alignment) < threshold:
        match_cols.append(i)

# חישוב מטריצת הפליטה מחדש
emission_counts = np.zeros((len(match_cols), 20)) + 1  # Pseudocount
for m_idx, col_idx in enumerate(match_cols):
    col = alignment[:, col_idx].upper()
    for char in col:
        if char in AA_TO_INDEX:
            emission_counts[m_idx, AA_TO_INDEX[char]] += 1
emission_probs = np.log(emission_counts / emission_counts.sum(axis=1, keepdims=True))

# 2. מציאת הרצף האנושי בתוך ה-MSA (כדי למפות אינדקסים)
human_seq_aligned = None
for record in alignment:
    if "HUMAN" in record.id.upper() or "HOMO" in record.description.upper():
        human_seq_aligned = str(record.seq)
        break

if human_seq_aligned is None:
    # אם לא מצא לפי שם, ניקח את הראשון כברירת מחדל (פחות טוב)
    human_seq_aligned = str(alignment[0].seq)
    print("Warning: Could not auto-detect Human sequence. Using first sequence.")
else:
    print("Found Human reference sequence in alignment.")

# מיפוי: אינדקס בחלבון האנושי (1, 2, 3...) -> אינדקס ב-Match State של ה-HMM
human_pos_to_hmm_index = {}
current_human_pos = 1  # ביולוגים סופרים מ-1
hmm_state_counter = 0

for i, char in enumerate(human_seq_aligned):
    is_match_col = (i in match_cols)

    if char != "-":
        # זו אות בחלבון האנושי
        if is_match_col:
            human_pos_to_hmm_index[current_human_pos] = hmm_state_counter
            hmm_state_counter += 1
        current_human_pos += 1
    else:
        # זה רווח בחלבון האנושי, אם העמודה היא match state אנחנו מתקדמים ב-HMM
        if is_match_col:
            hmm_state_counter += 1

# 3. טעינת ClinVar וחישוב ציונים
print("Scoring mutations...")

# נסיון טעינה חכם יותר - מזהה לבד אם זה טאבים או פסיקים
try:
    df = pd.read_csv(CLINVAR_FILE, sep="\t")
    if len(df.columns) < 2:  # אם לא הצליח לזהות טאבים
        df = pd.read_csv(CLINVAR_FILE)
except Exception as e:
    print(f"Error loading CSV: {e}")
    exit()

print("Available columns in file:", list(df.columns))  # הדפסה לדיבוג

# זיהוי אוטומטי של שם עמודת הסיווג (Clinical Significance)
sig_col = None
possible_sig_names = [
    'ClinicalSignificance',
    'Clinical significance (Last reviewed)',
    'Germline classification',
    'Class'
]

# נסה למצוא התאמה מדויקת או חלקית
for col in df.columns:
    for name in possible_sig_names:
        if name.lower() in col.lower():
            sig_col = col
            break
    if sig_col: break

if not sig_col:
    print("❌ Error: Could not find 'ClinicalSignificance' column automatically.")
    print("Please look at the 'Available columns' list above and update the code manually.")
    exit()

print(f"✅ Using column '{sig_col}' for clinical significance.")

# זיהוי אוטומטי של שם עמודת המוטציה
name_col = None
possible_name_cols = ['Name', 'ProteinChange', 'HGVS.p']
for col in df.columns:
    if col in possible_name_cols:
        name_col = col
        break

if not name_col:
    # ברירת מחדל אם לא מצא
    name_col = df.columns[0]

scores_pathogenic = []
scores_benign = []

for index, row in df.iterrows():
    # דילוג על שורות ללא מידע
    if pd.isna(row[name_col]) or pd.isna(row[sig_col]):
        continue

    pos, old_aa, new_aa = parse_clinvar_mutation(row[name_col])

    if pos and pos in human_pos_to_hmm_index:
        hmm_idx = human_pos_to_hmm_index[pos]

        # חישוב הציון
        new_aa_idx = AA_TO_INDEX.get(new_aa)
        if new_aa_idx is not None:
            mutation_score = emission_probs[hmm_idx, new_aa_idx]

            # סיווג לפי ClinVar
            sig = str(row[sig_col]).lower()

            # לוגיקה קצת יותר מתירנית כדי לתפוס יותר דאטה
            if 'pathogenic' in sig and 'benign' not in sig and 'conflicting' not in sig:
                scores_pathogenic.append(mutation_score)
            elif 'benign' in sig and 'pathogenic' not in sig and 'conflicting' not in sig:
                scores_benign.append(mutation_score)

print(f"Scored {len(scores_pathogenic)} Pathogenic and {len(scores_benign)} Benign mutations.")


# 4. ויזואליזציה (Boxplot)
plt.figure(figsize=(10, 6))
data_to_plot = [scores_pathogenic, scores_benign]
plt.boxplot(data_to_plot, patch_artist=True, labels=['Pathogenic', 'Benign'],
            boxprops=dict(facecolor='#ff9999', color='red'),
            medianprops=dict(color='black'))
plt.title('HMM Log-Likelihood Scores: Pathogenic vs Benign')
plt.ylabel('HMM Emission Score (Log Probability)')
plt.grid(True, alpha=0.3)

output_img = "hmm_log_likelihood_score.png"
plt.savefig(output_img, dpi=300, bbox_inches='tight')
plt.close() # סוגר את הזיכרון הגרפי


print(f"Scored {len(scores_pathogenic)} Pathogenic and {len(scores_benign)} Benign mutations.")




# --- חלק 5: סטטיסטיקה ו-ROC (התוספת החדשה - שמירה בלבד) ---

print("-" * 30)
print("Running Statistical Analysis...")

# 1. מבחן T (בודק אם ההבדל בין הקבוצות מובהק)
t_stat, p_val = stats.ttest_ind(scores_pathogenic, scores_benign, equal_var=False)
print(f"T-test results: t={t_stat:.2f}, p-value={p_val:.2e}")

if p_val < 0.05:
    print("✅ The difference is Statistically Significant!")
else:
    print("❌ No significant difference found.")

# 2. עקומת ROC (בודקת את הדיוק)
y_true = [1] * len(scores_pathogenic) + [0] * len(scores_benign)
y_scores = scores_pathogenic + scores_benign

# הופכים סימן כי אצלנו ציון נמוך = חולה
y_scores_inverted = [-x for x in y_scores]

fpr, tpr, thresholds = roc_curve(y_true, y_scores_inverted)
roc_auc = auc(fpr, tpr)

print(f"ROC AUC Score: {roc_auc:.3f}")

# 3. ציור הגרף ושמירה לקובץ (בלי להציג חלון)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)

# שמירה לקובץ וסגירה מיידית
output_file = "roc_curve.png"
plt.savefig(output_file, dpi=300)
plt.close() # סוגר את הזיכרון הגרפי בלי לפתוח חלון

print(f"✅ ROC Curve saved successfully to '{output_file}'")