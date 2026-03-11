# Model C: Custom Profile HMM Implementation

This folder contains the self-developed Profile HMM engine written in Python.

### Key Features
1.  **Viterbi Algorithm:** Implemented via dynamic programming in log-space to find the most probable path and score for any query sequence.
2.  **Delta Scoring:** Instead of absolute likelihood, we calculate pathogenicity as:
    $Score = LL_{WildType} - LL_{Variant}$
    This normalizes results against the canonical human sequence.
3.  **Smoothing:** Uses `np.maximum(matrix, 1e-300)` to ensure numerical stability during log-transformation.

### Files
* `Final_model_C.py`: The main logic for building the HMM from an MSA and scoring sequences.
* `run_model_C.py`: Wrapper for batch inference.