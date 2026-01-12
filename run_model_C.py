import sys
import pandas as pd
import numpy as np
from Bio import AlignIO, SeqIO
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# --- 1. CONFIGURATION & CONSTANTS ---
NEG_INF = -1e18

# The Official UniProt P04637 Sequence (P53_HUMAN)
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
    return float(max(end_cands)), []


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


def load_fasta_unlabeled(fasta_path: str):
    ids, descs, seqs = [], [], []
    for rec in SeqIO.parse(fasta_path, "fasta"):
        seq = str(rec.seq).upper()
        ids.append(rec.id)
        descs.append(rec.description)
        seqs.append(seq)
    return ids, descs, seqs


def build_model_from_msa(msa_path: str = "tp53_msa.fasta", theta: float = 0.35):
    aln = AlignIO.read(msa_path, "fasta")
    alignment = [str(rec.seq).upper() for rec in aln]
    AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWYX")
    alphabet = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
    states, transition, emission = profileHMM(alignment, alphabet, theta=theta)
    return states, transition, emission, alphabet


# ---------------------------------------------------------
# NEW: Modified loader to keep IDs (needed for the tbl output)
# ---------------------------------------------------------
def load_mutant_fasta_with_ids(fasta_path):
    ids, X, y = [], [], []
    for rec in SeqIO.parse(fasta_path, "fasta"):
        if "|label=" not in rec.description: continue
        label_part = rec.description.split("|label=", 1)[1].strip()

        label_val = -1
        if "benign" in label_part.lower():
            label_val = 0
        elif "pathogenic" in label_part.lower():
            label_val = 1

        if label_val != -1:
            # We construct a clean ID that includes the label for the .tbl file
            # e.g., "Var123|label=Pathogenic"
            clean_label = "Pathogenic" if label_val == 1 else "Benign"
            # Removing spaces from ID to ensure column alignment in .tbl
            safe_id = f"{rec.id}|label={clean_label}"

            ids.append(safe_id)
            X.append(str(rec.seq).upper())
            y.append(label_val)

    return ids, X, np.array(y, dtype=int)


# ---------------------------------------------------------
# NEW: Function to save Pseudo-HMMER .tbl
# ---------------------------------------------------------
def save_to_tbl(df, output_file="scores_C.tbl"):
    """
    Saves the results in a whitespace-delimited format compatible with 'roc_from_tbl.py'.
    Format requirements for the parser:
      - Column 0: Target Name (must contain '|label=...')
      - Column 5: Score (Bitscore or Log-Likelihood)
    """
    print(f"\nGeneratng .tbl file: {output_file} ...")
    with open(output_file, "w") as f:
        # Header (mimicking HMMER style, though parser ignores comments)
        f.write("# target name        accession  query name           accession  e-value  score  bias\n")
        f.write("# ------------------ ---------- -------------------- ---------- ------- ------ -----\n")

        for _, row in df.iterrows():
            target = row['id_with_label']
            score = row['raw_score']

            # Writing columns.
            # Col 0: target
            # Col 1-4: Placeholders ("-")
            # Col 5: Score
            # Col 6+: Placeholders
            # Using f-string alignment (<30) to make it look neat, though split() handles any whitespace.
            f.write(f"{target:<30} -          -                    -          -       {score:.4f}   -\n")

    print(f"✅ Saved .tbl file to '{output_file}'.")
    print("   You can now run: python roc_from_tbl.py --tbl scores_C.tbl")


# --- 4. MAIN EXECUTION ---

def evaluation_mode():
    print("--- Step 1: Building Profile HMM ---")
    states, transition, emission, alphabet = build_model_from_msa("tp53_msa.fasta", theta=0.35)
    print(f"HMM Model built ({len(states)} states).")

    print("\n--- Step 2: Calibrating with Wild Type ---")
    wt_score = score_sequence_viterbi(states, transition, emission, alphabet, WT_SEQUENCE)
    print(f"WT (UniProt P04637) Score: {wt_score:.4f}")

    print("\n--- Step 3: Loading & Scoring Mutants ---")
    # UPDATED: Using the new loader that returns IDs
    ids, X_seqs, y_labels = load_mutant_fasta_with_ids("tp53_clinvar_labeled.fasta")

    df = pd.DataFrame({
        "id_with_label": ids,
        "sequence": X_seqs,
        "label_bin": y_labels
    })

    print("Calculating scores for all mutants (this might take a moment)...")
    df["raw_score"] = [score_sequence_viterbi(states, transition, emission, alphabet, s) for s in df["sequence"]]
    df["delta_score"] = wt_score - df["raw_score"]

    # --- NEW STEP: Save the TBL file for your external script ---
    save_to_tbl(df, "scores_C.tbl")
    # ------------------------------------------------------------

    print(f"\nMean Healthy Delta: {np.mean(df[df.label_bin == 0].delta_score):.4f}")
    print(f"Mean Sick Delta:    {np.mean(df[df.label_bin == 1].delta_score):.4f}")

    # Standard internal evaluation (optional now if you use the external script)
    best_thr = choose_threshold(df.delta_score.values, df.label_bin.values, method="youden")
    print(f"Optimal Delta Threshold: {best_thr:.4f}")


def inference_mode(fasta_path: str):
    # This remains largely the same, but you could add tbl export here too if needed.
    print("--- Step 1: Building Profile HMM ---")
    states, transition, emission, alphabet = build_model_from_msa("tp53_msa.fasta", theta=0.35)
    wt_score = score_sequence_viterbi(states, transition, emission, alphabet, WT_SEQUENCE)

    # ... (Rest of inference logic) ...
    print("Inference mode logic (omitted for brevity, focus was on evaluation tbl export).")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        evaluation_mode()
    elif len(sys.argv) == 2:
        fasta_path = sys.argv[1]
        inference_mode(fasta_path)
    else:
        raise SystemExit("Usage: python your_script.py [optional_input.fasta]")