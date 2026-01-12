import sys
import pandas as pd
import numpy as np
from Bio import AlignIO, SeqIO
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# --- 1. CONFIGURATION & CONSTANTS ---
NEG_INF = -1e18

# The Official UniProt P04637 Sequence (P53_HUMAN)
# Using this ensures mathematically accurate Delta Scores.
WT_SEQUENCE = """
MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGP
DEAPRMPEAAPPVAPAPAAPTPAAPAPAPSWPLSSSVPSQKTYQGSYGFRLGFLHSGTAK
SVTCTYSPALNKMFCQLAKTCPVQLWVDSTPPPGTRVRAMAIYKQSQHMTEVVRRCPHHE
RCSDSDGLAPPQHLIRVEGNLRVEYLDDRNTFRHSVVVPYEPPEVGSDCTTIHYNYMCNS
SCMGGMNRRPILTIITLEDSSGNLLGRNSFEVRVCACPGRDRRTEEENLRKKGEPHHELP
PGSTKRALPNNTSSSPQPKKKPLDGEYFTLQIRGRERFEMFRELNEALELKDAQAGKEPG
GSRAHSSHLKSKKGQSTSRHKKLMFKTEGPDSD
""".replace("\n", "").replace(" ", "").strip()


# --- 2. HMM CORE FUNCTIONS ---

def profileHMM(alignment, alphabet, theta):
    def get_seed_alignment(alignment, theta, alphabet):
        k = len(alignment[0])
        a = len(alphabet.keys())
        freq = np.zeros(shape=(a + 1, k))
        for seq in alignment:
            for i in range(k):
                if seq[i] == '-':
                    freq[a][i] += 1
                else:
                    freq[alphabet[seq[i]]][i] += 1
        n = len(alignment)
        seed = [x / n < theta for x in freq[a]]
        return seed

    def normalize_matrices(T, E):
        for state in range(len(S)):
            if sum(T[state]) > 0:
                T[state] = T[state] / sum(T[state])
            if sum(E[state]) > 0:
                E[state] = E[state] / sum(E[state])
        return T, E

    def state_transition(T, prev, kind, S):
        x = 0
        if S[prev][0] == 'M': x = 1
        for nxt in range(prev + 1 + x, len(T)):
            if S[nxt][0] == kind[0]:
                T[prev][nxt] += 1
                return T, nxt

    seed = get_seed_alignment(alignment, theta, alphabet)
    n = len(alignment)
    k = len(alignment[0])
    S = ['S', 'I0'] + list(c + str(n) for n in range(1, sum(seed) + 1) for c in "MDI") + ['E']
    E = np.zeros(shape=(len(S), len(alphabet.keys())))
    T = np.zeros(shape=(len(S), len(S)))

    for seq in alignment:
        state = 0
        i = 0
        while i < k:
            if seed[i]:
                if seq[i] in alphabet:
                    T, state = state_transition(T, state, 'Match', S)
                    E[state][alphabet[seq[i]]] += 1
                else:
                    T, state = state_transition(T, state, 'Deletion', S)
            else:
                emits = []
                while not seed[i]:
                    if seq[i] in alphabet: emits.append(seq[i])
                    i += 1
                    if i == k: break
                i -= 1
                if len(emits) > 0:
                    T, state = state_transition(T, state, 'Insert', S)
                    for symbol in emits: E[state][alphabet[symbol]] += 1
                    if len(emits) > 1: T[state][state] += len(emits) - 1
            i += 1
        T, state = state_transition(T, state, 'End', S)

    T, E = normalize_matrices(T, E)
    return S, T, E


def _state_index(j: int, kind: str) -> int:
    base = 2 + 3 * (j - 1)
    if kind == "M": return base
    if kind == "D": return base + 1
    if kind == "I": return base + 2
    raise ValueError("kind must be M/D/I")


def viterbi_profile_hmm(states, T, E, alphabet, sequence):
    seq = sequence.upper()
    L = len(seq)
    k = (len(states) - 3) // 3
    end_idx = len(states) - 1
    eps = 1e-300
    Tlog = np.log(np.maximum(T, eps))
    Elog = np.log(np.maximum(E, eps))

    VM = np.full((k + 1, L + 1), NEG_INF, dtype=float)
    VI = np.full((k + 1, L + 1), NEG_INF, dtype=float)
    VD = np.full((k + 1, L + 1), NEG_INF, dtype=float)

    S_idx = 0;
    I0_idx = 1

    def emit_log(state_idx, ch):
        if ch not in alphabet: return np.log(eps)
        return Elog[state_idx, alphabet[ch]]

    if L >= 1:
        VI[0, 1] = Tlog[S_idx, I0_idx] + emit_log(I0_idx, seq[0])
        for i in range(2, L + 1):
            VI[0, i] = VI[0, i - 1] + Tlog[I0_idx, I0_idx] + emit_log(I0_idx, seq[i - 1])

    if k >= 1:
        D1 = _state_index(1, "D")
        VD[1, 0] = Tlog[S_idx, D1]
        for j in range(2, k + 1):
            Dj = _state_index(j, "D")
            Dprev = _state_index(j - 1, "D")
            VD[j, 0] = VD[j - 1, 0] + Tlog[Dprev, Dj]

    if k >= 1 and L >= 1:
        M1 = _state_index(1, "M")
        cand1 = Tlog[S_idx, M1]
        cand2 = VI[0, 0] + Tlog[I0_idx, M1] if VI[0, 0] > NEG_INF / 2 else NEG_INF
        VM[1, 1] = max(cand1, cand2) + emit_log(M1, seq[0])

    for j in range(1, k + 1):
        Mj = _state_index(j, "M")
        Dj = _state_index(j, "D")
        Ij = _state_index(j, "I")
        for i in range(1, L + 1):
            ch = seq[i - 1]
            if j >= 2:
                Mprev = _state_index(j - 1, "M")
                Dprev = _state_index(j - 1, "D")
                Iprev = _state_index(j - 1, "I")
                cands = [VM[j - 1, i - 1] + Tlog[Mprev, Mj], VI[j - 1, i - 1] + Tlog[Iprev, Mj],
                         VD[j - 1, i - 1] + Tlog[Dprev, Mj]]
                VM[j, i] = max(cands) + emit_log(Mj, ch)

            candsI = [VM[j, i - 1] + Tlog[Mj, Ij], VD[j, i - 1] + Tlog[Dj, Ij], VI[j, i - 1] + Tlog[Ij, Ij]]
            VI[j, i] = max(candsI) + emit_log(Ij, ch)

        if j >= 2:
            Mprev = _state_index(j - 1, "M");
            Dprev = _state_index(j - 1, "D");
            Iprev = _state_index(j - 1, "I")
            for i in range(0, L + 1):
                candsD = [VM[j - 1, i] + Tlog[Mprev, Dj], VI[j - 1, i] + Tlog[Iprev, Dj],
                          VD[j - 1, i] + Tlog[Dprev, Dj]]
                VD[j, i] = max(candsD)

    Mk = _state_index(k, "M");
    Dk = _state_index(k, "D");
    Ik = _state_index(k, "I")
    end_cands = [VM[k, L] + Tlog[Mk, end_idx], VI[k, L] + Tlog[Ik, end_idx], VD[k, L] + Tlog[Dk, end_idx]]
    return float(max(end_cands)), []  # Path not returned to save time


def clean_protein_sequence(seq: str, alphabet: dict) -> str:
    s = str(seq).strip().upper().replace(" ", "").replace("\n", "").replace("\r", "")
    return "".join(ch for ch in s if ch in alphabet)


def score_sequence_viterbi(states, T, E, alphabet, seq: str) -> float:
    seq_clean = clean_protein_sequence(seq, alphabet)
    if len(seq_clean) == 0: return float("-inf")
    best_ll, _ = viterbi_profile_hmm(states, T, E, alphabet, seq_clean)
    return best_ll


# --- 3. EVALUATION HELPERS ---

def choose_threshold(scores, labels, method="youden"):
    scores = np.asarray(scores);
    labels = np.asarray(labels)
    if method == "means":
        return 0.5 * (np.mean(scores[labels == 0]) + np.mean(scores[labels == 1]))

    # Youden's Index
    uniq = np.unique(scores[~np.isinf(scores)])
    candidates = [(uniq[i] + uniq[i + 1]) / 2 for i in range(len(uniq) - 1)]
    candidates = [uniq[0] - 1] + candidates + [uniq[-1] + 1]

    best_thr, best_J = candidates[0], -1e18
    P = np.sum(labels == 1);
    N = np.sum(labels == 0)

    for thr in candidates:
        pred = (scores >= thr).astype(int)
        TP = np.sum((pred == 1) & (labels == 1))
        FP = np.sum((pred == 1) & (labels == 0))
        J = (TP / P if P else 0) - (FP / N if N else 0)
        if J > best_J: best_J = J; best_thr = thr
    return best_thr


# ------ new ------
def load_fasta_unlabeled(fasta_path: str):
    """
    Load sequences from a FASTA file that may have no labels.
    Returns:
        ids: list[str]  (record.id)
        descs: list[str] (record.description)
        seqs: list[str] (sequence)
    """
    ids, descs, seqs = [], [], []
    for rec in SeqIO.parse(fasta_path, "fasta"):
        seq = str(rec.seq).upper()
        ids.append(rec.id)
        descs.append(rec.description)
        seqs.append(seq)
    return ids, descs, seqs

def wrap_fasta(seq: str, width: int = 60) -> str:
    return "\n".join(seq[i:i+width] for i in range(0, len(seq), width))


def predict_labels_for_fasta(
    fasta_in: str,
    states, transition, emission, alphabet,
    wt_score: float,
    threshold: float,
    output_csv: str = "predictions.csv",
    output_fasta_with_labels: str | None = None
):
    """
    Predict BENIGN/PATHOGENIC for each sequence in an unlabeled FASTA,
    based on delta_score = wt_score - raw_score and a learned threshold.
    Writes a CSV summary; optionally writes a FASTA with |label=... headers.
    """
    ids, descs, seqs = load_fasta_unlabeled(fasta_in)

    results = []
    labeled_records = []

    for rec_id, desc, seq in zip(ids, descs, seqs):
        raw = score_sequence_viterbi(states, transition, emission, alphabet, seq)
        delta = wt_score - raw

        pred_bin = 1 if delta >= threshold else 0
        pred_label = "PATHOGENIC" if pred_bin == 1 else "BENIGN"

        results.append({
            "id": rec_id,
            "description": desc,
            "raw_score": raw,
            "delta_score": delta,
            "threshold": threshold,
            "pred_label": pred_label
        })

        if output_fasta_with_labels is not None:
            # Add label field to header without destroying the original description
            new_header = f"{desc}|label={pred_label}"
            labeled_records.append((new_header, seq))

    df_pred = pd.DataFrame(results)
    df_pred.to_csv(output_csv, index=False)
    print(f"Saved predictions table to: {output_csv}")

    if output_fasta_with_labels is not None:
        with open(output_fasta_with_labels, "w", encoding="utf-8") as f:
            for header, seq in labeled_records:
                f.write(">" + header + "\n")
                f.write(wrap_fasta(seq) + "\n")
        print(f"Saved labeled FASTA to: {output_fasta_with_labels}")

    return df_pred


def build_model_from_msa(msa_path: str = "tp53_msa.fasta", theta: float = 0.35):
    aln = AlignIO.read(msa_path, "fasta")
    alignment = [str(rec.seq).upper() for rec in aln]
    AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWYX")
    alphabet = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
    states, transition, emission = profileHMM(alignment, alphabet, theta=theta)
    return states, transition, emission, alphabet


# --- YOUR REQUESTED FUNCTION IS HERE ---
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


# ---------------------------------------

def load_mutant_fasta(fasta_path):
    X, y = [], []
    for rec in SeqIO.parse(fasta_path, "fasta"):
        if "|label=" not in rec.description: continue
        label = rec.description.split("|label=", 1)[1].strip()
        if label in {"Benign", "Likely_benign"}:
            y.append(0)
        elif label in {"Pathogenic", "Likely_pathogenic", "Pathogenic_or_Likely_pathogenic"}:
            y.append(1)
        else:
            continue
        X.append(str(rec.seq).upper())
    return X, np.array(y, dtype=int)


def plot_roc_curve(y_true, y_scores, output_file="roc_curve.png"):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0]);
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate');
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right");
    plt.grid(True, alpha=0.3)
    plt.savefig(output_file, dpi=300);
    plt.close()
    print(f"✅ ROC Curve saved to '{output_file}' (AUC: {roc_auc:.4f})")


# --- 4. MAIN EXECUTION ---

def evaluation_mode():
    print("--- Step 1: Building Profile HMM ---")
    states, transition, emission, alphabet = build_model_from_msa("tp53_msa.fasta", theta=0.35)
    print(f"HMM Model built ({len(states)} states).")

    print("\n--- Step 2: Calibrating with Wild Type ---")
    wt_score = score_sequence_viterbi(states, transition, emission, alphabet, WT_SEQUENCE)
    print(f"WT (UniProt P04637) Score: {wt_score:.4f}")

    print("\n--- Step 3: Loading & Scoring Mutants ---")
    X_seqs, y_labels = load_mutant_fasta("tp53_clinvar_labeled.fasta")
    df = pd.DataFrame({"sequence": X_seqs, "label_bin": y_labels})

    print("Calculating scores for all mutants (this might take a moment)...")
    df["raw_score"] = [score_sequence_viterbi(states, transition, emission, alphabet, s) for s in df["sequence"]]
    df["delta_score"] = wt_score - df["raw_score"]

    print(f"Mean Healthy Delta: {np.mean(df[df.label_bin == 0].delta_score):.4f}")
    print(f"Mean Sick Delta:    {np.mean(df[df.label_bin == 1].delta_score):.4f}")

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    split_idx = int(len(df) * 0.60)
    df_train = df.iloc[:split_idx]
    df_test = df.iloc[split_idx:]

    print(f"\nTraining on {len(df_train)} samples, Testing on {len(df_test)} samples.")

    print("\n--- Step 4: Optimization & Evaluation ---")
    best_thr = choose_threshold(df_train.delta_score.values, df_train.label_bin.values, method="youden")
    print(f"Optimal Delta Threshold (calculated on Train): {best_thr:.4f}")

    final_metrics = evaluate_threshold(df_test.delta_score.values, df_test.label_bin.values, best_thr)

    print("\nResults on Test Set:")
    print(f"  Accuracy:  {final_metrics['accuracy']:.4f}")
    print(f"  Precision: {final_metrics['precision']:.4f}")
    print(f"  Recall:    {final_metrics['recall']:.4f}")
    print(f"  F1 Score:  {final_metrics['f1']:.4f}")
    print("\nConfusion Matrix:")
    print(f"  TP: {final_metrics['TP']} | FP: {final_metrics['FP']}")
    print(f"  FN: {final_metrics['FN']} | TN: {final_metrics['TN']}")

    print("\n--- Step 5: Generating Plots ---")
    plot_roc_curve(df_test.label_bin.values, df_test.delta_score.values)

    # נחזיר גם את המודל וה-threshold למקרה שתרצי להשתמש בזה בלי CLI
    return states, transition, emission, alphabet, wt_score, best_thr


def inference_mode(fasta_path: str):
    print("--- Step 1: Building Profile HMM ---")
    states, transition, emission, alphabet = build_model_from_msa("tp53_msa.fasta", theta=0.35)
    print(f"HMM Model built ({len(states)} states).")

    print("\n--- Step 2: Calibrating with Wild Type ---")
    wt_score = score_sequence_viterbi(states, transition, emission, alphabet, WT_SEQUENCE)
    print(f"WT (UniProt P04637) Score: {wt_score:.4f}")

    print("\n--- Step 3: Learning Threshold from ClinVar Labeled Set ---")
    X_seqs, y_labels = load_mutant_fasta("tp53_clinvar_labeled.fasta")
    df = pd.DataFrame({"sequence": X_seqs, "label_bin": y_labels})
    df["raw_score"] = [score_sequence_viterbi(states, transition, emission, alphabet, s) for s in df["sequence"]]
    df["delta_score"] = wt_score - df["raw_score"]

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    split_idx = int(len(df) * 0.60)
    df_train = df.iloc[:split_idx]

    best_thr = choose_threshold(df_train.delta_score.values, df_train.label_bin.values, method="youden")
    print(f"Using learned threshold (Train/Youden): {best_thr:.4f}")

    print("\n--- Step 4: Predicting on Unlabeled FASTA ---")
    df_pred = predict_labels_for_fasta(
        fasta_in=fasta_path,
        states=states, transition=transition, emission=emission, alphabet=alphabet,
        wt_score=wt_score,
        threshold=best_thr,
        output_csv="unlabeled_predictions.csv",
        output_fasta_with_labels="unlabeled_with_predicted_labels.fasta"
    )

    # הדפסה קצרה למסך
    print("\nTop predictions:")
    print(df_pred[["id", "pred_label", "delta_score"]].head(20).to_string(index=False))





if __name__ == "__main__":
    if __name__ == "__main__":
        # Usage:
        #   python your_script.py                -> evaluation mode (as before)
        #   python your_script.py input.fasta    -> inference mode on input.fasta
        if len(sys.argv) == 1:
            evaluation_mode()
        elif len(sys.argv) == 2:
            fasta_path = sys.argv[1]
            inference_mode(fasta_path)
        else:
            raise SystemExit("Usage: python your_script.py [optional_input.fasta]")