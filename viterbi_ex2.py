from Bio import SeqIO  # pip install biopython
from collections import Counter, defaultdict
import argparse
import gzip
import numpy as np
from hmmlearn.hmm import CategoricalHMM
import random
import plotly.graph_objects as go
import time

# Define dinucleotide symbols and mapping
DINUCS = [a + b for a in "ACGT" for b in "ACGT"]  # ['AA','AC',...,'TT']
DINUC_TO_IDX = {dinuc: i for i, dinuc in enumerate(DINUCS)}

def parse_fasta_file(file_path: str):
    """
    Parses a FASTA file (plain or gzipped) and returns a mapping of sequence identifiers to nucleotide sequences.

    Parameters:
        file_path (str): The path to the FASTA file.

    Returns:
        dict: A dictionary with sequence IDs as keys and DNA sequences as values.
    """
    sequences = {}

    if file_path.endswith('.gz'):
        with gzip.open(file_path, 'rt') as file_handle:
            for record in SeqIO.parse(file_handle, "fasta"):
                sequences[record.id] = str(record.seq)
    else:
        with open(file_path, 'r') as file_handle:
            for record in SeqIO.parse(file_handle, "fasta"):
                sequences[record.id] = str(record.seq)

    return sequences

def prepare_training_data(sequence_file: str, label_file: str):
    """
    Aligns nucleotide sequences with corresponding labels to create a training dataset.

    Parameters:
        sequence_file (str): Path to the FASTA file containing sequences.
        label_file (str): Path to the FASTA file containing labels.

    Returns:
        list[tuple[str, str]]: A list of tuples where each tuple contains a DNA sequence and its label.
    """
    sequences = parse_fasta_file(sequence_file)
    labels = parse_fasta_file(label_file)

    if sequences.keys() != labels.keys():
        raise ValueError("Mismatch between sequence IDs and label IDs in the provided files.")

    return [(sequences[seq_id], labels[seq_id]) for seq_id in sequences]


def encode_sequence_to_dinucs(seq: str) -> np.ndarray:
    """
    Convert a nucleotide sequence into dinucleotide observations for hmmlearn.
    Output shape is (L-1, 1).
    """
    seq = seq.upper()
    obs = []
    for i in range(len(seq) - 1):
        dinuc = seq[i:i+2]
        if dinuc in DINUC_TO_IDX:
            obs.append(DINUC_TO_IDX[dinuc])
        else:
            # Unknown characters → map to 0 (AA), simple fallback
            obs.append(0)
    return np.array(obs, dtype=int).reshape(-1, 1)


def train_classifier(training_data):
    """
    Trains a classifier to identify CpG islands in DNA sequences.

    Parameters:
        training_data (list[tuple[str, str]]): Training data consisting of sequences and their labels.

    Returns:
        object: Your trained classifier model.
    """

    states = ['N', 'C']
    symbols = DINUCS  # 16 dinucleotides

    # counts
    init_counts = Counter()
    trans_counts = {s: Counter() for s in states}
    emit_counts = {s: Counter() for s in states}

    # loop over training data to collect counts
    for seq, lbl in training_data:
        seq = seq.upper()
        lbl = lbl.upper()

        assert len(seq) == len(lbl), "sequence and label lengths must match"

        # initial state at position 0
        init_counts[lbl[0]] += 1

        # emission at position 0 (dinuc of pos 0 and 1)
        prev_state = lbl[0]
        o = "".join([seq[0], seq[1]])
        emit_counts[prev_state][o] += 1

        for i in range(1, len(seq)-1):
            s = lbl[i]
            o = "".join([seq[i], seq[i+1]])
            emit_counts[s][o] += 1
            trans_counts[prev_state][s] += 1
            prev_state = s

    # Laplace smoothing
    alpha = 1.0

    # Initial probabilities
    initial = {}
    total_init = sum(init_counts[s] + alpha for s in states)
    for s in states:
        initial[s] = (init_counts[s] + alpha) / total_init

    # transition probabilities
    transition = {}
    for s_from in states:
        transition[s_from] = {}
        total = sum(trans_counts[s_from][s_to] + alpha for s_to in states)
        for s_to in states:
            transition[s_from][s_to] = (trans_counts[s_from][s_to] + alpha) / total

    # emission probabilities
    emission = {}
    for s in states:
        emission[s] = {}
        total = sum(emit_counts[s][sym] + alpha for sym in symbols)
        for sym in symbols:
            emission[s][sym] = (emit_counts[s][sym] + alpha) / total

    # model = MultinomialHMM(n_components=2, n_trials=len(training_data[0]), init_params="")
    model = CategoricalHMM(n_components=2, n_features=16, init_params="")
    model.startprob_ = np.array([initial['N'], initial['C']])
    model.transmat_ = np.array([[transition['N']['N'], transition['N']['C']], [transition['C']['N'], transition['C']['C']]])
    model.emissionprob_ = np.array([
        [emission['N'][sym] for sym in symbols],
        [emission['C'][sym] for sym in symbols]])
    # print("Initial: \n",initial)
    # print()
    # print("Transition: \n",model.transmat_)
    # print()
    # print("Emission: \n",model.emissionprob_)
    # print()
    return model


def annotate_sequence(model, sequence):
    """
    Annotates a DNA sequence with CpG island predictions.

    Parameters:
        model (object): Your trained classifier model.
        sequence (str): A DNA sequence to be annotated.

    Returns:
        str: A string of annotations, where 'C' marks a CpG island region and 'N' denotes non-CpG regions.
    """
    seq = sequence.upper()

    # if sequence is too short, return all 'N's
    if len(seq) < 2:
        return "N" * len(seq)

    # encode to dinucleotide observations
    X = encode_sequence_to_dinucs(seq)  # shape (L-1, 1)

    # predict states using Viterbi
    logprob, states = model.decode(X, algorithm="viterbi")

    # convert states from 0/1 to 'N'/'C'
    state_map = {0: 'N', 1: 'C'}
    annotations = ''.join(state_map[state] for state in states)

    return annotations

def annotate_fasta_file(model, input_path, output_path):
    """
    Annotates all sequences in a FASTA file with CpG island predictions.

    Parameters:
        model (object): A trained classifier model.
        input_path (str): Path to the input FASTA file.
        output_path (str): Path to the output FASTA file where annotations will be saved.

    Writes:
        A gzipped FASTA file containing predicted annotations for each input sequence.
    """
    sequences = parse_fasta_file(input_path)

    with gzip.open(output_path, 'wt') as gzipped_file:
        for seq_id, sequence in sequences.items():
            annotation = annotate_sequence(model, sequence)
            gzipped_file.write(f">{seq_id}\n{annotation}\n")


def write_sequences_to_fasta(pairs, fasta_path):
    """
    write sequences to a FASTA file.
    """
    with open(fasta_path, "w") as f:
        for i, (seq, lbl) in enumerate(pairs):
            f.write(f">seq_{i}\n")
            f.write(seq + "\n")



def evaluate_model(model, eval_set, verbose=True):
    """
    Evaluate the HMM classifier on a labeled evaluation set.

    eval_set: list of (sequence, labels), where labels are 'C'/'N'.

    Prints:
        - Accuracy
        - Precision (for class 'C')
        - Recall (for class 'C')
        - F1 score (for class 'C')
        - Confusion matrix counts: TP, FP, TN, FN
    """
    tp = fp = tn = fn = 0

    for seq, lbl in eval_set:
        seq = seq.upper()
        lbl = lbl.upper()

        pred = annotate_sequence(model, seq)

        # sanity check
        if len(pred) != len(lbl)-1:
            raise ValueError(
                f"Prediction and label length mismatch: "
                f"{len(pred)} vs {len(lbl)}"
            )

        for y_true, y_pred in zip(lbl, pred):
            if y_true == 'C':
                if y_pred == 'C':
                    tp += 1
                else:
                    fn += 1
            elif y_true == 'N':
                if y_pred == 'C':
                    fp += 1
                else:
                    tn += 1
            else:
                # in case there are other label chars, just ignore them
                continue

    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    if verbose:
        print("Evaluation on eval_set")
        print(f"Total bases: {total}")
        print(f"TP (true C predicted C): {tp}")
        print(f"FP (true N predicted C): {fp}")
        print(f"TN (true N predicted N): {tn}")
        print(f"FN (true C predicted N): {fn}")
        print()
        print(f"Accuracy : {accuracy:.4f}")
        print(f"sensitivity   : {sensitivity:.4f} (Recall for class 'C')")
        print(f"specificity   : {specificity:.4f} (Recall for class 'N')")

    return sensitivity, specificity


def split_train_val(training_data, val_fraction=0.2, seed=9):
    """
    Split data into a training pool (for incremental training) and
    a fixed validation set.

    val_fraction: fraction of data used as validation (e.g. 0.2)
    """
    random.seed(seed)
    random.shuffle(training_data)

    num_sequences = len(training_data)
    split_val_idx = int((1.0 - val_fraction) * num_sequences)

    train_pool = training_data[:split_val_idx]   # 80%
    val_set    = training_data[split_val_idx:]   # 20%

    return train_pool, val_set


def run_sensitivity_specificity_curve(train_pool, val_set, num_runs=10):
    """
    Runs num_runs trainings with increasing fractions of train_pool
    and evaluates on the fixed validation set.

    Returns:
        sensitivities: list of floats
        specificities: list of floats
    """
    sensitivities = []
    specificities = []
    train_sizes = []
    runtimes = []

    pool_size = len(train_pool)

    for k in range(1, num_runs + 1):
        frac = k / num_runs          # 0.1, 0.2, ..., 1.0
        curr_train_size = int(frac * pool_size)
        curr_train_set = train_pool[:curr_train_size]

        print(f"\nRun {k}: using {curr_train_size} sequences "
              f"({frac*100:.0f}% of the 80% train pool)")

        start_time = time.perf_counter()

        # Train model on current subset
        classifier = train_classifier(curr_train_set)

        # Evaluate on fixed validation set
        sens, spec = evaluate_model(classifier, val_set, verbose=False)

        end_time = time.perf_counter()
        runtime = end_time - start_time

        print(f"Run {k}: sensitivity={sens:.4f}, specificity={spec:.4f}")

        sensitivities.append(sens)
        specificities.append(spec)
        train_sizes.append(curr_train_size)
        runtimes.append(runtime)

    return sensitivities, specificities, train_sizes, runtimes


def plot_sens_spec(sensitivities, specificities, output_path="sensitivity_vs_specificity06.html"):
    """
    Plot sensitivity vs specificity, each run as a point.
    """
    runs = list(range(1, len(sensitivities) + 1))

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=sensitivities,
        y=specificities,
        mode='markers+text',
        text=[f"{i}" for i in runs],
        textposition="top center",
        hovertemplate="Run %{text}<br>Sensitivity=%{x:.3f}<br>Specificity=%{y:.3f}<extra></extra>",
        name="Runs"
    ))

    fig.update_layout(
        title="Sensitivity vs Specificity for increasing training size",
        xaxis_title="Sensitivity (Recall of C)",
        yaxis_title="Specificity (Recall of N)",
    )

    fig.write_html(output_path)
    print(f"Interactive plot saved to {output_path}")


def plot_runtime_vs_train_size(train_sizes, runtimes,
                               output_path="runtime_vs_train_size06.html"):
    """
    Plot runtime (seconds) vs training set size (number of sequences).
    Each run is a point with its index as text.
    """
    runs = list(range(1, len(train_sizes) + 1))

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=train_sizes,
        y=runtimes,
        mode='markers+text',
        text=[f"{i}" for i in runs],
        textposition="top center",
        hovertemplate="Run %{text}<br>Train size=%{x}<br>Runtime=%{y:.3f} s<extra></extra>",
        name="Runs"
    ))

    coef = np.polyfit(train_sizes, runtimes, 1)
    poly = np.poly1d(coef)

    x_line = np.linspace(min(train_sizes), max(train_sizes), 100)
    y_line = poly(x_line)

    fig.add_trace(go.Scatter(
        x=x_line,
        y=y_line,
        mode='lines',
        name="Trend line",
        line=dict(color="red", width=2)
    ))

    fig.update_layout(
        title="Runtime vs Training Set Size",
        xaxis_title="training set size (number of sequences)",
        yaxis_title="runtime (seconds)",
    )

    fig.write_html(output_path)
    print(f"Interactive runtime plot saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Predict CpG islands in DNA sequences.")
    parser.add_argument("--fasta_path", type=str, help="Path to the input FASTA file containing DNA sequences.")
    parser.add_argument("--output_file", type=str, help="Path to the output FASTA file for saving predictions.")

    args = parser.parse_args()

    training_sequences_path = r"data/CpG-islands.2K.seq.fa"
    training_labels_path = r"data/CpG-islands.2K.lbl.fa"

    # Prepare training data and train model
    training_data = prepare_training_data(training_sequences_path, training_labels_path)

    if args.fasta_path is not None and args.output_file is not None:
        print("Running in PREDICTION mode ")
        # Train final model on full training data
        classifier = train_classifier(training_data)

        # Annotate sequences and save predictions
        annotate_fasta_file(classifier, args.fasta_path, args.output_file)
        print(f"Saved predictions to {args.output_file}")
        return

    print("Running in EVALUATION mode ")
    # Split into train pool (80%) and validation set (20%)
    train_pool, val_set = split_train_val(training_data, val_fraction=0.2, seed=9)

    # Run incremental training and collect sensitivity/specificity points
    sensitivities, specificities, train_sizes, runtimes = run_sensitivity_specificity_curve(
        train_pool, val_set, num_runs=10)

    # Plot sensitivity vs specificity
    plot_sens_spec(sensitivities, specificities)

    # Plot runtime vs training set size
    plot_runtime_vs_train_size(train_sizes, runtimes)

    # Write validation set sequences to FASTA (for inspection)
    eval_fasta_path = "data/eval_sequences06.fa"
    write_sequences_to_fasta(val_set, eval_fasta_path)

    # Train final model on full train_pool
    classifier = train_classifier(train_pool)

    # Annotate sequences and save predictions for the validation set
    output_gz_path = "data/test_predictions.fa.gz"
    annotate_fasta_file(classifier, eval_fasta_path, output_gz_path)
    print(f"Saved validation predictions to {output_gz_path}")



if __name__ == '__main__':
    main()