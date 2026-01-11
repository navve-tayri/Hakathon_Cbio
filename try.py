import pandas as pd
import numpy as np
from Bio import AlignIO
from hmmlearn.hmm import CategoricalHMM


### algorithm for building profile HMM(alignment*, theta)

def profileHMM(alignment, alphabet, theta):
    # # 'seed' indicates which columns pass threshold theta for align*
    def get_seed_alignment(alignment, theta, alphabet):
        """'seed': True if gaps < theta; included in align*
            returns seed alignment columns for HMM"""
        k = len(alignment[0])
        a = len(alphabet.keys())
        #         print("Length of alignment(k) = ",k)
        #         print("size of alphabet = ",len(alphabet),'\n')
        freq = np.zeros(shape=(a + 1, k))

        for seq in alignment:
            for i in range(k):
                if seq[i] == '-':
                    #                     print('\tgap', seq[i])
                    freq[a][i] += 1
                else:
                    freq[alphabet[seq[i]]][i] += 1
        #                     print('\tbase', seq[i])
        n = len(alignment)
        print('\nFrequency matrix of alignment and seed columns:\n', freq)
        seed = [x / n < theta for x in freq[a]]
        #         print('seed:',seed)
        print(" ", '  '.join(list('+' if i else '-' for i in seed)))
        return seed

    def normalize_matrices(T, E):
        """counts to probabilities: normalize across rows"""
        for state in range(len(S)):
            if sum(T[state]) > 0:
                T[state] = T[state] / sum(T[state])
            if sum(E[state]) > 0:
                E[state] = E[state] / sum(E[state])
        return T, E

    def state_transition(T, prev, kind, S):
        """return next Match, Del, Ins, or End state in [states] """
        x = 0
        if S[prev][0] == 'M':
            x = 1
        for nxt in range(prev + 1 + x, len(T)):
            if S[nxt][0] == kind[0]:
                T[prev][nxt] += 1
                #                 print('  >Transition added:',((S[prev],prev),(S[nxt], nxt)))
                return T, nxt

    ### Walk each sequence through it's hidden states and count transitions & emissions

    seed = get_seed_alignment(alignment, theta, alphabet)
    n = len(alignment)
    k = len(alignment[0])
    S = ['S', 'I0'] + list(c + str(n) for n in range(1, sum(seed) + 1) for c in "MDI") + ['E']
    E = np.zeros(shape=(len(S), len(alphabet.keys())))
    T = np.zeros(shape=(len(S), len(S)))

    for seq in alignment:
        # 'state' is the hidden state col/row; i is pos in alignment
        state = 0;
        i = 0
        while i < k:

            # seed: either match or del as hidden state
            if seed[i]:
                if seq[i] in alphabet:
                    T, state = state_transition(T, state, 'Match', S)
                    E[state][alphabet[seq[i]]] += 1
                else:
                    T, state = state_transition(T, state, 'Deletion', S)

            # not seed: either insert or nothing
            else:
                # count any emissions before next seed column
                emits = []
                while not seed[i]:
                    if seq[i] in alphabet:
                        emits.append(seq[i])
                    i += 1
                    if i == k:
                        break  # hit end of sequence
                i -= 1
                if len(emits) > 0:
                    # count the transition(state, insertion)
                    T, state = state_transition(T, state, 'Insert', S)
                    # get all emissions(state) in insertion
                    for symbol in emits:
                        E[state][alphabet[symbol]] += 1
                    # count all symbols as cyclic transition t(ins_x, ins_x)
                    if len(emits) > 1:
                        T[state][state] += len(emits) - 1
                # else do nothing, just gaps in align

            i += 1
        # from last state to 'End'
        T, state = state_transition(T, state, 'End', S)

    T, E = normalize_matrices(T, E)
    return S, T, E


# new
NEG_INF = -1e18

def _state_index(j: int, kind: str) -> int:
    """
    kind in {"M","D","I"}.
    states layout: 0:S, 1:I0, then for j>=1: Mj, Dj, Ij (3 per j), last:E
    """
    # base index for j>=1 block starts at 2
    base = 2 + 3*(j-1)
    if kind == "M":
        return base
    if kind == "D":
        return base + 1
    if kind == "I":
        return base + 2
    raise ValueError("kind must be M/D/I")

def viterbi_profile_hmm(states, T, E, alphabet, sequence):
    """
    Viterbi decoding for Profile-HMM with silent Deletion states.
    Returns:
        best_logprob, path_states (list[str])
    """
    seq = sequence.upper()
    L = len(seq)

    # number of match states k inferred from states list:
    # states = ['S','I0'] + [M1,D1,I1,...,Mk,Dk,Ik] + ['E']
    k = (len(states) - 3) // 3  # remove S,I0,E -> remaining 3k
    end_idx = len(states) - 1

    eps = 1e-300
    Tlog = np.log(np.maximum(T, eps))
    Elog = np.log(np.maximum(E, eps))

    # DP tables: dimensions (k+1) x (L+1)
    VM = np.full((k+1, L+1), NEG_INF, dtype=float)
    VI = np.full((k+1, L+1), NEG_INF, dtype=float)
    VD = np.full((k+1, L+1), NEG_INF, dtype=float)

    # backpointers store tuples: (prev_kind, prev_j, prev_i)
    bM = [[None]*(L+1) for _ in range(k+1)]
    bI = [[None]*(L+1) for _ in range(k+1)]
    bD = [[None]*(L+1) for _ in range(k+1)]

    S_idx = 0
    I0_idx = 1

    def emit_log(state_idx, ch):
        if ch not in alphabet:
            return np.log(eps)
        return Elog[state_idx, alphabet[ch]]

    # --- Initialization ---
    # Start at S with 0 consumed symbols, j=0
    # Insertions before first match: I0 can emit
    # VI[0,1] from S->I0
    if L >= 1:
        VI[0, 1] = Tlog[S_idx, I0_idx] + emit_log(I0_idx, seq[0])
        bI[0][1] = ("S", 0, 0)  # from S

        # I0 self-loop for more leading inserts
        for i in range(2, L+1):
            VI[0, i] = VI[0, i-1] + Tlog[I0_idx, I0_idx] + emit_log(I0_idx, seq[i-1])
            bI[0][i] = ("I", 0, i-1)

    # Deletions before consuming anything: S -> D1 -> D2 -> ... possible
    if k >= 1:
        D1 = _state_index(1, "D")
        VD[1, 0] = Tlog[S_idx, D1]
        bD[1][0] = ("S", 0, 0)

        for j in range(2, k+1):
            Dj = _state_index(j, "D")
            Dprev = _state_index(j-1, "D")
            VD[j, 0] = VD[j-1, 0] + Tlog[Dprev, Dj]
            bD[j][0] = ("D", j-1, 0)

    # First match M1 at i=1: from S->M1 OR I0->M1
    if k >= 1 and L >= 1:
        M1 = _state_index(1, "M")
        # from S
        cand1 = Tlog[S_idx, M1]
        # from I0 (if leading inserts exist)
        cand2 = VI[0, 0] + Tlog[I0_idx, M1] if VI[0, 0] > NEG_INF/2 else NEG_INF

        best = max(cand1, cand2)
        VM[1, 1] = best + emit_log(M1, seq[0])
        bM[1][1] = ("S", 0, 0) if best == cand1 else ("I", 0, 0)

    # --- Recurrence ---
    for j in range(1, k+1):
        Mj = _state_index(j, "M")
        Dj = _state_index(j, "D")
        Ij = _state_index(j, "I")

        # I_j depends on same j, i-1
        for i in range(1, L+1):
            ch = seq[i-1]

            # ---- Compute VM[j,i] (needs j-1, i-1) ----
            if j >= 2:
                Mprev = _state_index(j-1, "M")
                Dprev = _state_index(j-1, "D")
                Iprev = _state_index(j-1, "I")

                cands = [
                    (VM[j-1, i-1] + Tlog[Mprev, Mj], ("M", j-1, i-1)),
                    (VI[j-1, i-1] + Tlog[Iprev, Mj], ("I", j-1, i-1)),
                    (VD[j-1, i-1] + Tlog[Dprev, Mj], ("D", j-1, i-1)),
                ]
                best_val, best_ptr = max(cands, key=lambda x: x[0])
                VM[j, i] = best_val + emit_log(Mj, ch)
                bM[j][i] = best_ptr

            # ---- Compute VI[j,i] (needs same j, i-1) ----
            # transitions from Mj->Ij, Dj->Ij, Ij->Ij
            candsI = [
                (VM[j, i-1] + Tlog[Mj, Ij], ("M", j, i-1)),
                (VD[j, i-1] + Tlog[Dj, Ij], ("D", j, i-1)),
                (VI[j, i-1] + Tlog[Ij, Ij], ("I", j, i-1)),
            ]
            best_val, best_ptr = max(candsI, key=lambda x: x[0])
            VI[j, i] = best_val + emit_log(Ij, ch)
            bI[j][i] = best_ptr

        # ---- Compute VD[j,i] (silent; needs j-1, same i) ----
        if j >= 2:
            Mprev = _state_index(j-1, "M")
            Dprev = _state_index(j-1, "D")
            Iprev = _state_index(j-1, "I")
            for i in range(0, L+1):
                candsD = [
                    (VM[j-1, i] + Tlog[Mprev, Dj], ("M", j-1, i)),
                    (VI[j-1, i] + Tlog[Iprev, Dj], ("I", j-1, i)),
                    (VD[j-1, i] + Tlog[Dprev, Dj], ("D", j-1, i)),
                ]
                best_val, best_ptr = max(candsD, key=lambda x: x[0])
                VD[j, i] = best_val
                bD[j][i] = best_ptr

    # --- Termination to E after consuming all L symbols ---
    Mk = _state_index(k, "M")
    Dk = _state_index(k, "D")
    Ik = _state_index(k, "I")

    end_cands = [
        (VM[k, L] + Tlog[Mk, end_idx], ("M", k, L)),
        (VI[k, L] + Tlog[Ik, end_idx], ("I", k, L)),
        (VD[k, L] + Tlog[Dk, end_idx], ("D", k, L)),
    ]
    best_logprob, best_ptr = max(end_cands, key=lambda x: x[0])

    # --- Backtrack ---
    path = ["E"]
    kind, j, i = best_ptr
    while True:
        if kind == "S":
            path.append("S")
            break
        if kind == "M":
            path.append(f"M{j}")
            prev = bM[j][i]
        elif kind == "I":
            path.append(f"I{j}")
            prev = bI[j][i]
        elif kind == "D":
            path.append(f"D{j}")
            prev = bD[j][i]
        else:
            raise RuntimeError("Unknown backpointer kind")

        if prev is None:
            # safety stop (shouldn't happen if model is connected)
            break
        kind, j, i = prev

    path.reverse()
    return best_logprob, path


def clean_protein_sequence(seq: str, alphabet: dict) -> str:
    """
    מנקה רצף: מסיר רווחים/תווי שורה, הופך ל-Upper,
    ומחליף תווים לא מוכרים ב-'X' אם קיים באלפבית, אחרת מסנן אותם.
    """
    s = str(seq).strip().upper().replace(" ", "").replace("\n", "").replace("\r", "")
    allowed = set(alphabet.keys())

    if "X" in allowed:
        return "".join(ch if ch in allowed else "X" for ch in s)
    else:
        return "".join(ch for ch in s if ch in allowed)

def score_sequence_viterbi(states, T, E, alphabet, seq: str) -> float:
    """
    מחזיר ציון Viterbi log-prob לרצף (ככל שגבוה יותר -> מתאים יותר למודל).
    """
    seq_clean = clean_protein_sequence(seq, alphabet)
    if len(seq_clean) == 0:
        return float("-inf")
    best_ll, _ = viterbi_profile_hmm(states, T, E, alphabet, seq_clean)
    return float(best_ll)


def choose_threshold(scores: np.ndarray, labels: np.ndarray, method: str = "means") -> float:
    """
    labels: 0=בריא, 1=חולה
    scores: log-likelihood (גבוה=יותר "דומה לחולה" לפי ההגדרה שנעשה בהמשך)

    method:
      - "means": threshold = (mean_healthy + mean_sick)/2
      - "youden": threshold שממקסם Youden's J = TPR - FPR
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)

    s0 = scores[labels == 0]
    s1 = scores[labels == 1]
    if len(s0) == 0 or len(s1) == 0:
        raise ValueError("צריך שיהיו גם בריאים (0) וגם חולים (1) כדי לקבוע threshold.")

    if method == "means":
        return 0.5 * (np.mean(s0) + np.mean(s1))

    if method == "youden":
        # סורקים מועמדים טבעיים: כל ערך ציון + מעט
        uniq = np.unique(scores[~np.isinf(scores)])
        if len(uniq) == 0:
            return float("nan")

        # מועמדים: midpoints בין ערכים עוקבים (סטנדרטי ל-ROC threshold)
        candidates = []
        candidates.append(uniq[0] - 1.0)
        for a, b in zip(uniq[:-1], uniq[1:]):
            candidates.append(0.5 * (a + b))
        candidates.append(uniq[-1] + 1.0)

        best_thr = candidates[0]
        best_J = -1e18

        P = np.sum(labels == 1)
        N = np.sum(labels == 0)

        for thr in candidates:
            pred = (scores >= thr).astype(int)  # 1 אם מעל thr
            TP = np.sum((pred == 1) & (labels == 1))
            FP = np.sum((pred == 1) & (labels == 0))
            TPR = TP / P if P else 0.0
            FPR = FP / N if N else 0.0
            J = TPR - FPR
            if J > best_J:
                best_J = J
                best_thr = thr

        return float(best_thr)

    raise ValueError("method must be 'means' or 'youden'")



def predict_health_label_from_score(score: float, threshold: float) -> int:
    """
    מחזיר 1=חולה אם score >= threshold, אחרת 0=בריא
    """
    return int(score >= threshold)


def predict_sequence(states, T, E, alphabet, threshold: float, seq: str) -> dict:
    """
    API נוח: מחזיר score + label
    """
    s = score_sequence_viterbi(states, T, E, alphabet, seq)
    y = predict_health_label_from_score(s, threshold)
    return {"score": s, "pred_label": y}  # 0/1


def evaluate_threshold(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    pred = (scores >= threshold).astype(int)

    TP = np.sum((pred == 1) & (labels == 1))
    TN = np.sum((pred == 0) & (labels == 0))
    FP = np.sum((pred == 1) & (labels == 0))
    FN = np.sum((pred == 0) & (labels == 1))

    acc = (TP + TN) / max(1, (TP + TN + FP + FN))
    prec = TP / max(1, (TP + FP))
    rec = TP / max(1, (TP + FN))
    f1 = 2 * prec * rec / max(1e-12, (prec + rec))

    return {
        "threshold": float(threshold),
        "TP": int(TP), "TN": int(TN), "FP": int(FP), "FN": int(FN),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
    }


from Bio import SeqIO

def load_mutant_fasta(fasta_path: str):
    """
    Expects headers like:
      >TP53_D234Y|label=Likely_benign
    Returns list of (seq, y) where y in {0,1}
    """
    X = []
    y = []
    for rec in SeqIO.parse(fasta_path, "fasta"):
        header = rec.description
        seq = str(rec.seq).upper()

        if "|label=" not in header:
            continue
        label_str = header.split("|label=", 1)[1].strip()

        if label_str in {"Benign", "Likely_benign"}:
            y.append(0)
        elif label_str in {"Pathogenic", "Likely_pathogenic", "Pathogenic_or_Likely_pathogenic"}:
            y.append(1)
        else:
            continue

        X.append(seq)
    return X, np.array(y, dtype=int)


# if __name__ == "__main__":
    # aln = AlignIO.read("tp53_msa.fasta", "fasta")
    # alignment = [str(rec.seq).upper() for rec in aln]  # list[str]
    #
    # AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWYX")
    # alphabet = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
    # theta = 0.35
    #
    # states, transition, emission = profileHMM(alignment, alphabet, theta)
    # test_seq = alignment[0].replace("-", "")
    # best_ll, v_path = viterbi_profile_hmm(states, transition, emission, alphabet, test_seq)
    #
    # print("\nStates:\n", states)
    # print("\nTransition Matrix:\n", pd.DataFrame(transition, index=states, columns=states))
    # print("\nEmission Matrix:\n", pd.DataFrame(emission, index=states, columns=list(alphabet.keys())))
    # print("\nViterbi best log-likelihood:", best_ll)
    # print("Viterbi path length:", len(v_path))
    # print("First 50 states in path:", v_path[:50])

#
# if __name__ == "__main__":
#     import pandas as pd
#     import numpy as np
#
#     # ---------------------------------------------------------
#     # 1) בניית המודל
#     # ---------------------------------------------------------
#     print("--- Step 1: Building Profile HMM ---")
#     aln = AlignIO.read("tp53_msa.fasta", "fasta")
#     alignment = [str(rec.seq).upper() for rec in aln]
#
#     AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWYX")
#     alphabet = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
#     theta = 0.35
#
#     states, transition, emission = profileHMM(alignment, alphabet, theta)
#     print(f"HMM Model built successfully ({len(states)} states).")
#
#     # ---------------------------------------------------------
#     # 2) טעינת הנתונים
#     # ---------------------------------------------------------
#     print("\n--- Step 2: Loading Data ---")
#     mutants_file = "mutants.fasta"
#     try:
#         X_seqs, y_labels = load_mutant_fasta(mutants_file)
#     except FileNotFoundError:
#         print(f"Error: '{mutants_file}' not found.")
#         exit()
#
#     df = pd.DataFrame({"sequence": X_seqs, "label_bin": y_labels})
#
#     # מציאת רצף ה-WT מתוך הקובץ (לפי הכותרת או שהוא הראשון)
#     # הנחה: הרצף הראשון בקובץ mutants.fasta הוא ה-WT (כך הקוד של חברה שלך בנוי)
#     wt_seq = df.iloc[0]["sequence"]
#     print(f"Identified WT sequence (length {len(wt_seq)})")
#
#     # חישוב ציון ה-WT פעם אחת
#     wt_score = score_sequence_viterbi(states, transition, emission, alphabet, wt_seq)
#     print(f"WT Log-Likelihood: {wt_score:.4f}")
#
#     # ערבוב
#     df = df.sample(frac=1, random_state=42).reset_index(drop=True)
#
#     # חלוקה 80/20
#     split_idx = int(len(df) * 0.80)
#     df_train = df.iloc[:split_idx].copy()
#     df_test = df.iloc[split_idx:].copy()
#
#     # ---------------------------------------------------------
#     # 3) חישוב ציוני Delta (השינוי הגדול!)
#     # ---------------------------------------------------------
#     print("\n--- Step 3: Calibrating with Delta Score ---")
#
#
#     def get_delta_scores(sequences, wt_val):
#         """
#         מחשב: ציון_WT פחות ציון_מוטנט.
#         ערך גבוה = המוטנט הרבה פחות סביר מה-WT = כנראה פתוגני.
#         """
#         raw_scores = np.array([
#             score_sequence_viterbi(states, transition, emission, alphabet, seq)
#             for seq in sequences
#         ])
#         # Delta Score: כמה המוטציה "קלקלה" את הציון ביחס ל-WT
#         return wt_val - raw_scores
#
#
#     # חישוב ציונים לקבוצת האימון
#     train_scores = get_delta_scores(df_train["sequence"].astype(str).tolist(), wt_score)
#     train_labels = df_train["label_bin"].astype(int).to_numpy()
#
#     # שימוש ב-Youden (עכשיו זה יעבוד כי הכיוון נכון)
#     best_thr = choose_threshold(train_scores, train_labels, method="youden")
#
#     print(f"Optimal Delta Threshold: {best_thr:.4f}")
#     # הסבר: אם ההפרש גדול מהסף הזה -> המודל יגיד "חולה"
#
#     # ---------------------------------------------------------
#     # 4) בדיקה סופית
#     # ---------------------------------------------------------
#     print("\n--- Step 4: Final Evaluation ---")
#
#     test_scores = get_delta_scores(df_test["sequence"].astype(str).tolist(), wt_score)
#     test_labels = df_test["label_bin"].astype(int).to_numpy()
#
#     final_metrics = evaluate_threshold(test_scores, test_labels, best_thr)
#
#     print("Results on Test Set:")
#     print(f"  Accuracy:  {final_metrics['accuracy']:.4f}")
#     print(f"  Precision: {final_metrics['precision']:.4f}")
#     print(f"  Recall:    {final_metrics['recall']:.4f}")
#     print(f"  F1 Score:  {final_metrics['f1']:.4f}")
#
#     print("\nConfusion Matrix:")
#     print(f"  TP: {final_metrics['TP']} (Sick correctly identified)")
#     print(f"  TN: {final_metrics['TN']} (Healthy correctly identified)")
#     print(f"  FP: {final_metrics['FP']} (Healthy called Sick)")
#     print(f"  FN: {final_metrics['FN']} (Sick called Healthy)")


if __name__ == "__main__":
    import pandas as pd
    import numpy as np

    # ---------------------------------------------------------
    # 1) בניית המודל (Profile-HMM)
    # ---------------------------------------------------------
    print("--- Step 1: Building Profile HMM ---")
    aln = AlignIO.read("tp53_msa.fasta", "fasta")
    alignment = [str(rec.seq).upper() for rec in aln]

    AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWYX")
    alphabet = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
    theta = 0.35

    states, transition, emission = profileHMM(alignment, alphabet, theta)
    print(f"HMM Model built successfully ({len(states)} states).")

    # ---------------------------------------------------------
    # 2) טעינת הנתונים
    # ---------------------------------------------------------
    print("\n--- Step 2: Loading Data ---")
    mutants_file = "mutants.fasta"
    try:
        X_seqs, y_labels = load_mutant_fasta(mutants_file)
    except FileNotFoundError:
        print(f"Error: '{mutants_file}' not found.")
        exit()

    df = pd.DataFrame({"sequence": X_seqs, "label_bin": y_labels})

    # מציאת רצף ה-WT וחישוב הציון שלו (לשימוש ב-Delta בהמשך)
    wt_seq = df.iloc[0]["sequence"]
    wt_score = score_sequence_viterbi(states, transition, emission, alphabet, wt_seq)
    print(f"WT Sequence identified. Log-Likelihood: {wt_score:.4f}")

    # ---------------------------------------------------------
    # 3) חישוב ציונים גולמיים (Raw Scores) ובדיקת ממוצעים - [החלק שביקשת]
    # ---------------------------------------------------------
    print("\n--- Step 3: Raw Scores Analysis (Debug) ---")

    # חישוב ציונים לכל הרצפים בדאטה-בייס (לפני חלוקה)
    all_raw_scores = []
    for seq in df["sequence"].astype(str).tolist():
        sc = score_sequence_viterbi(states, transition, emission, alphabet, seq)
        all_raw_scores.append(sc)

    all_raw_scores = np.array(all_raw_scores)
    labels_all = df["label_bin"].astype(int).to_numpy()

    # הדפסת ממוצעים לבדיקה )
    mean_healthy = np.mean(all_raw_scores[labels_all == 0])
    mean_sick = np.mean(all_raw_scores[labels_all == 1])

    print(f"Mean Score Healthy: {mean_healthy:.4f}")
    print(f"Mean Score Sick:    {mean_sick:.4f}")
    print(f"Gap (Signal):       {mean_healthy - mean_sick:.4f}")

    # אם ה-Gap קטן מאוד (למשל פחות מ-2.0), המודל מתקשה להבדיל

    # ---------------------------------------------------------
    # 4) הכנת Delta Scores וחלוקה ל-Train/Test
    # ---------------------------------------------------------
    print("\n--- Step 4: Preparing Delta Scores & Splitting ---")

    # הוספת הנתונים ל-DataFrame כדי שנוכל לערבב ולחלק בנוחות
    df["raw_score"] = all_raw_scores
    # חישוב Delta: כמה המוטציה הורידה את הציון ביחס ל-WT
    # ערך גבוה = שינוי דרסטי (חולה), ערך נמוך/שלילי = דומה ל-WT (בריא)
    df["delta_score"] = wt_score - df["raw_score"]

    # ערבוב הדאטה
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # חלוקה 80/20
    split_idx = int(len(df) * 0.80)
    df_train = df.iloc[:split_idx].copy()
    df_test = df.iloc[split_idx:].copy()

    print(f"Training on {len(df_train)} samples, Testing on {len(df_test)} samples.")

    # ---------------------------------------------------------
    # 5) קביעת סף (Threshold) על סמך Delta Scores
    # ---------------------------------------------------------
    print("\n--- Step 5: Finding Threshold (on Train Set) ---")

    train_deltas = df_train["delta_score"].to_numpy()
    train_labels = df_train["label_bin"].to_numpy()

    # שימוש ב-Youden למציאת הסף האופטימלי להפרש (Delta)
    best_thr = choose_threshold(train_deltas, train_labels, method="youden")
    print(f"Optimal Delta Threshold: {best_thr:.4f}")
    print("(Sequences with Delta > Threshold will be predicted as Sick)")

    # ---------------------------------------------------------
    # 6) בדיקה סופית (Evaluation)
    # ---------------------------------------------------------
    print("\n--- Step 6: Final Evaluation (on Test Set) ---")

    test_deltas = df_test["delta_score"].to_numpy()
    test_labels = df_test["label_bin"].to_numpy()

    final_metrics = evaluate_threshold(test_deltas, test_labels, best_thr)

    print("Results on Test Set:")
    print(f"  Accuracy:  {final_metrics['accuracy']:.4f}")
    print(f"  Precision: {final_metrics['precision']:.4f}")
    print(f"  Recall:    {final_metrics['recall']:.4f}")
    print(f"  F1 Score:  {final_metrics['f1']:.4f}")

    print("\nConfusion Matrix:")
    print(f"  TP: {final_metrics['TP']} | FP: {final_metrics['FP']}")
    print(f"  FN: {final_metrics['FN']} | TN: {final_metrics['TN']}")