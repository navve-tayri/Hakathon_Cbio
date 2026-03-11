#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
main_model.py

A practical (but still compact) Profile-HMM implementation for a protein MSA,
plus utilities to score TP53 missense variants (e.g., from ClinVar).

What this script provides (compared to the earlier "emission-only" version):
1) A real Profile-HMM topology with Match/Insert/Delete states per position:
      M1..Mk, I0..Ik, D1..Dk  (plus begin/end)
2) Estimated transition probabilities from the alignment (with pseudocounts),
   in log-space, suitable for Forward/Viterbi scoring of sequences.
3) Emissions for:
      - Match states: learned per match column from the MSA
      - Insert states: learned globally from residues occurring in insert regions
        (or fallback to background), shared across all insert states.
4) A TP53-oriented "delta score" for missense variants, computed as:
      Δ = log P(mutAA | M_pos) - log P(wtAA | M_pos)
   This is often a strong baseline for conservation-based pathogenicity scoring.

Note:
- This is not a full "HMMER replacement" (e.g., it does not do alignment of a
  query to the model). However, it *is* a genuine Profile-HMM with transitions
  and a proper likelihood scoring for a given sequence using Forward/Viterbi.
- For mutation scoring you typically already know the position (TP53 AA index),
  so the emission delta at Match state is appropriate and efficient.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from Bio import AlignIO

import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc

# -----------------------------
# Global constants / mappings
# -----------------------------

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
AA_TO_INDEX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
INDEX_TO_AA = {i: aa for aa, i in AA_TO_INDEX.items()}

AA3_TO_AA1 = {
    "Ala": "A", "Cys": "C", "Asp": "D", "Glu": "E", "Phe": "F",
    "Gly": "G", "His": "H", "Ile": "I", "Lys": "K", "Leu": "L",
    "Met": "M", "Asn": "N", "Pro": "P", "Gln": "Q", "Arg": "R",
    "Ser": "S", "Thr": "T", "Val": "V", "Trp": "W", "Tyr": "Y",
    # common alternates / special:
    "Ter": "*", "Stop": "*"
}


# -----------------------------
# Helper utilities
# -----------------------------

def logsumexp(log_values: np.ndarray) -> float:
    """Stable log-sum-exp for a 1D vector."""
    m = np.max(log_values)
    if np.isneginf(m):
        return -np.inf
    return float(m + np.log(np.sum(np.exp(log_values - m))))


def safe_log(x: float) -> float:
    """Log with safety for zero."""
    if x <= 0.0:
        return -np.inf
    return math.log(x)


def ungap(seq: str) -> str:
    """Remove gaps from an aligned sequence."""
    return seq.replace("-", "")


def parse_clinvar_pdot(pdot: str) -> Optional[Tuple[int, str, str]]:
    """
    Parse a ClinVar protein change like: 'p.Arg175His' or 'p.R175H'.

    Returns:
        (pos, wt_aa, mut_aa) with 1-based position and 1-letter AAs,
        or None if parsing fails / not a simple missense.
    """
    if not isinstance(pdot, str):
        return None

    # 1) 3-letter form: p.Arg175His
    m = re.match(r"^p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})$", pdot.strip())
    if m:
        wt3, pos_s, mut3 = m.group(1), m.group(2), m.group(3)
        if wt3 not in AA3_TO_AA1 or mut3 not in AA3_TO_AA1:
            return None
        wt1, mut1 = AA3_TO_AA1[wt3], AA3_TO_AA1[mut3]
        if wt1 == "*" or mut1 == "*":
            return None  # not a missense
        return int(pos_s), wt1, mut1

    # 2) 1-letter form: p.R175H
    m = re.match(r"^p\.([A-Z\*])(\d+)([A-Z\*])$", pdot.strip())
    if m:
        wt1, pos_s, mut1 = m.group(1), m.group(2), m.group(3)
        if wt1 == "*" or mut1 == "*":
            return None
        if wt1 not in AA_TO_INDEX or mut1 not in AA_TO_INDEX:
            return None
        return int(pos_s), wt1, mut1

    return None


# -----------------------------
# MSA processing
# -----------------------------

def load_msa(msa_file: str):
    """Load an MSA (FASTA) using Biopython AlignIO."""
    alignment = AlignIO.read(msa_file, "fasta")
    print(f"Loaded MSA with {len(alignment)} sequences and {alignment.get_alignment_length()} columns.")
    return alignment


def identify_match_columns(alignment, gap_threshold: float = 0.5) -> List[int]:
    """
    Decide which alignment columns are 'Match' columns.
    Rule: a column is Match if (gap_fraction < gap_threshold).

    gap_threshold=0.5 matches the common rule-of-thumb in many introductions.
    """
    n_seqs = len(alignment)
    n_cols = alignment.get_alignment_length()
    match_cols: List[int] = []

    for col_i in range(n_cols):
        col = alignment[:, col_i]
        gap_fraction = col.count("-") / n_seqs
        if gap_fraction < gap_threshold:
            match_cols.append(col_i)

    print(f"Identified {len(match_cols)} Match columns out of {n_cols} MSA columns.")
    return match_cols


# -----------------------------
# Profile-HMM implementation
# -----------------------------

@dataclass(frozen=True)
class State:
    """A named HMM state."""
    name: str
    emits: bool  # whether this state emits a residue (Match/Insert)


class ProfileHMM:
    """
    A compact Profile-HMM with states:
      Begin (B), End (E),
      Match Mi (i=1..k), Insert Ii (i=0..k), Delete Di (i=1..k)

    Allowed transitions (standard profile HMM):
      From B:  M1, I0, D1
      For i=1..k-1:
        Mi -> Mi+1, Ii, Di+1
        Ii -> Mi+1, Ii, Di+1
        Di -> Mi+1, Ii, Di+1
      For i=k:
        Mk -> E, Ik
        Ik -> E, Ik
        Dk -> E, Ik

    Emissions:
      - Match: position-specific
      - Insert: shared distribution (learned from insert residues or background)

    This is sufficient to:
      - compute a sequence log-likelihood via Forward algorithm
      - compute a most-likely state path via Viterbi
    """

    def __init__(self, pseudocount: float = 1.0):
        self.pseudocount = float(pseudocount)

        self.match_cols: List[int] = []
        self.k: int = 0

        # state index mapping
        self.states: List[State] = []
        self.state_to_idx: Dict[str, int] = {}

        # log-transition matrix: [n_states, n_states]
        self.logA: Optional[np.ndarray] = None

        # log-emissions:
        #   match_logE[i, aa] for Mi (i=1..k) stored at index i-1
        self.match_logE: Optional[np.ndarray] = None

        # insert_logE[aa] shared for all insert states
        self.insert_logE: Optional[np.ndarray] = None

        # background (for optional log-odds):
        self.bg_logE: Optional[np.ndarray] = None

        # Reference (wild-type) mapping of match positions to a sequence (optional)
        self.reference_ungapped: Optional[str] = None

    # ---------- building ----------
    def build_from_alignment(
        self,
        alignment,
        gap_threshold: float = 0.5,
        reference_seq_index: int = 0,
    ) -> "ProfileHMM":
        """
        Build the model from a protein MSA.
        """
        self.match_cols = identify_match_columns(alignment, gap_threshold=gap_threshold)
        self.k = len(self.match_cols)
        if self.k == 0:
            raise ValueError("No match columns identified; try a larger gap_threshold.")

        # Build state list in a stable order:
        # B, I0, (M1,D1,I1), (M2,D2,I2), ... , (Mk,Dk,Ik), E
        self.states = [State("B", emits=False), State("I0", emits=True)]
        for i in range(1, self.k + 1):
            self.states.append(State(f"M{i}", emits=True))
            self.states.append(State(f"D{i}", emits=False))
            self.states.append(State(f"I{i}", emits=True))
        self.states.append(State("E", emits=False))
        self.state_to_idx = {st.name: idx for idx, st in enumerate(self.states)}

        # Learn emissions
        self._learn_background(alignment)
        self._learn_match_emissions(alignment)
        self._learn_insert_emissions(alignment)

        # Learn transitions
        self._learn_transitions(alignment)

        # Store a reference (ungapped) sequence (useful for sanity checks / variant scoring)
        ref_aln = str(alignment[reference_seq_index].seq)
        self.reference_ungapped = ungap(ref_aln)

        return self

    def _learn_background(self, alignment) -> None:
        """Compute a global amino-acid background from all non-gap chars in the MSA."""
        counts = np.zeros(len(AMINO_ACIDS), dtype=float)
        for rec in alignment:
            for ch in str(rec.seq).upper():
                if ch in AA_TO_INDEX:
                    counts[AA_TO_INDEX[ch]] += 1.0
        counts += self.pseudocount
        probs = counts / counts.sum()
        self.bg_logE = np.log(probs)

    def _learn_match_emissions(self, alignment) -> None:
        """Position-specific match emissions (Mi) from match columns."""
        n_aa = len(AMINO_ACIDS)
        counts = np.zeros((self.k, n_aa), dtype=float)

        for mi, col_idx in enumerate(self.match_cols):
            col = alignment[:, col_idx]
            for ch in col:
                ch = ch.upper()
                if ch in AA_TO_INDEX:
                    counts[mi, AA_TO_INDEX[ch]] += 1.0

        counts += self.pseudocount
        probs = counts / counts.sum(axis=1, keepdims=True)
        self.match_logE = np.log(probs)

    def _learn_insert_emissions(self, alignment) -> None:
        """
        Estimate a shared insert emission distribution by collecting residues that appear
        in insert columns (i.e., columns that are not in match_cols).
        Fallback to background if no insert residues exist.
        """
        match_set = set(self.match_cols)
        counts = np.zeros(len(AMINO_ACIDS), dtype=float)

        n_cols = alignment.get_alignment_length()
        for col_idx in range(n_cols):
            if col_idx in match_set:
                continue
            col = alignment[:, col_idx]
            for ch in col:
                ch = ch.upper()
                if ch in AA_TO_INDEX:
                    counts[AA_TO_INDEX[ch]] += 1.0

        if counts.sum() == 0:
            # fallback to background
            self.insert_logE = self.bg_logE.copy()
            return

        counts += self.pseudocount
        probs = counts / counts.sum()
        self.insert_logE = np.log(probs)

    def _learn_transitions(self, alignment) -> None:
        """
        Learn transitions by converting each aligned sequence to a path over states and
        counting adjacent transitions. Add pseudocounts to allowed transitions only.
        """
        n_states = len(self.states)
        A_counts = np.zeros((n_states, n_states), dtype=float)

        # Allowed transitions mask
        allowed = np.zeros((n_states, n_states), dtype=bool)
        for frm, tos in self._allowed_transitions().items():
            i = self.state_to_idx[frm]
            for to in tos:
                j = self.state_to_idx[to]
                allowed[i, j] = True

        # Count transitions for each sequence
        for rec in alignment:
            path = self.aligned_sequence_to_state_path(str(rec.seq))
            for s_from, s_to in zip(path, path[1:]):
                i = self.state_to_idx[s_from]
                j = self.state_to_idx[s_to]
                if not allowed[i, j]:
                    # Should not happen if path builder respects topology
                    continue
                A_counts[i, j] += 1.0

        # Add pseudocounts only to allowed transitions
        A_counts = A_counts + (allowed.astype(float) * self.pseudocount)

        # Normalize rows to probabilities; disallowed transitions remain zero
        row_sums = A_counts.sum(axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            A_probs = np.divide(A_counts, row_sums, out=np.zeros_like(A_counts), where=row_sums > 0)

        # Convert to log-space; keep zeros as -inf
        self.logA = np.full_like(A_probs, fill_value=-np.inf, dtype=float)
        mask = A_probs > 0
        self.logA[mask] = np.log(A_probs[mask])

    def _allowed_transitions(self) -> Dict[str, List[str]]:
        """Return the allowed transition graph (standard profile-HMM)."""
        allowed: Dict[str, List[str]] = {"B": ["M1", "I0", "D1"]}

        for i in range(1, self.k):
            allowed[f"M{i}"] = [f"M{i+1}", f"I{i}", f"D{i+1}"]
            allowed[f"I{i}"] = [f"M{i+1}", f"I{i}", f"D{i+1}"]
            allowed[f"D{i}"] = [f"M{i+1}", f"I{i}", f"D{i+1}"]

        # i = k
        allowed[f"M{self.k}"] = ["E", f"I{self.k}"]
        allowed[f"I{self.k}"] = ["E", f"I{self.k}"]
        allowed[f"D{self.k}"] = ["E", f"I{self.k}"]

        return allowed

    # ---------- alignment -> path ----------
    def aligned_sequence_to_state_path(self, aligned_seq: str) -> List[str]:
        """
        Convert ONE aligned sequence (with gaps) into a state path that is consistent
        with the profile-HMM topology.

        Strategy:
        - We scan match columns in order and decide for each match position i:
            - if aligned char is aa: emit via Mi
            - if aligned char is gap: go via Di (delete)
          Between match columns, any inserted residues in non-match columns are emitted
          by Ii-1 (or I0 before the first match), with self-loops for multiple residues.

        The resulting path always starts with "B" and ends with "E".
        """
        aligned_seq = aligned_seq.upper()
        path: List[str] = ["B"]

        # helper: emit a run of insert residues at insert state Ij
        def emit_inserts(insert_state: str, residues: List[str]) -> None:
            if not residues:
                return
            # Enter insert state once, then self-loop as needed.
            for idx, r in enumerate(residues):
                path.append(insert_state)
                if idx > 0:
                    # self-loop implied by consecutive same state
                    pass

        # We'll iterate over the alignment columns and collect residues between match cols.
        match_set = set(self.match_cols)
        match_positions = {col_idx: mi for mi, col_idx in enumerate(self.match_cols, start=1)}

        current_insert_residues: List[str] = []
        current_insert_state = "I0"

        for col_idx, ch in enumerate(aligned_seq):
            if col_idx not in match_set:
                if ch in AA_TO_INDEX:
                    current_insert_residues.append(ch)
                continue

            # This is a match column. First, flush inserts collected so far.
            emit_inserts(current_insert_state, current_insert_residues)
            current_insert_residues = []

            i = match_positions[col_idx]  # 1..k
            if ch in AA_TO_INDEX:
                path.append(f"M{i}")
            else:
                # gap or unknown -> treat as delete in match position
                path.append(f"D{i}")

            # After handling match i, inserts before next match are I_i
            current_insert_state = f"I{i}"

        # flush trailing inserts after last match
        emit_inserts(current_insert_state, current_insert_residues)

        path.append("E")
        return path

    # ---------- emissions ----------
    def _emit_logp(self, state_name: str, residue: str) -> float:
        """Return log P(residue | state)."""
        residue = residue.upper()
        if residue not in AA_TO_INDEX:
            return -np.inf
        aa_idx = AA_TO_INDEX[residue]

        if state_name.startswith("M"):
            i = int(state_name[1:])  # 1..k
            return float(self.match_logE[i - 1, aa_idx])
        if state_name.startswith("I"):
            return float(self.insert_logE[aa_idx])
        # Delete/Begin/End do not emit
        return -np.inf

    # ---------- scoring ----------
    def forward_log_likelihood(self, seq: str) -> float:
        """
        Forward algorithm in log-space.
        Input seq is an *ungapped* amino-acid sequence.
        Returns log P(seq | model).

        Implementation notes:
        - This DP is implemented over all states, at each emitted symbol position t.
        - Non-emitting states (B, D*, E) need care; we handle by expanding them with
          epsilon transitions using a simple relaxation (topology is acyclic across
          delete chain within each layer, but insert self-loops exist).
        - To keep the code compact, we implement a "two-level" DP:
            - Emitting step updates for M*/I* states consuming one residue
            - Then we propagate epsilon transitions through non-emitting states
          This works because deletes do not consume residues.
        """
        if self.logA is None:
            raise RuntimeError("Model not built yet.")
        seq = seq.upper()
        # Filter to valid AAs only; if you prefer strict behavior, raise instead.
        seq = "".join([c for c in seq if c in AA_TO_INDEX])
        T = len(seq)
        n_states = len(self.states)

        # alpha[state] at current time t (how many residues consumed)
        alpha = np.full(n_states, -np.inf, dtype=float)
        alpha[self.state_to_idx["B"]] = 0.0

        # Propagate epsilon transitions from B before consuming anything
        alpha = self._epsilon_closure(alpha)

        for t in range(T):
            aa = seq[t]

            next_alpha = np.full(n_states, -np.inf, dtype=float)

            # For each emitting state j, consider transitions from any i.
            for j, st in enumerate(self.states):
                if not st.emits:
                    continue
                emit = self._emit_logp(st.name, aa)
                if np.isneginf(emit):
                    continue

                # incoming from all i
                incoming = alpha + self.logA[:, j]
                next_alpha[j] = logsumexp(incoming) + emit

            # After consuming aa, propagate epsilon transitions via deletes/end etc.
            alpha = self._epsilon_closure(next_alpha)

        # Finally, probability is alpha at End after epsilon closure
        end_idx = self.state_to_idx["E"]
        return float(alpha[end_idx])

    def viterbi(self, seq: str) -> Tuple[float, List[str]]:
        """
        Viterbi algorithm in log-space.
        Returns (best_logp, best_state_path_names).

        Same caveats about epsilon transitions as forward().
        """
        if self.logA is None:
            raise RuntimeError("Model not built yet.")
        seq = seq.upper()
        seq = "".join([c for c in seq if c in AA_TO_INDEX])
        T = len(seq)
        n_states = len(self.states)

        # dp[t, j] = best logp ending in state j after consuming t residues
        dp = np.full((T + 1, n_states), -np.inf, dtype=float)
        back = np.full((T + 1, n_states), -1, dtype=int)

        dp[0, self.state_to_idx["B"]] = 0.0
        dp[0] = self._epsilon_closure(dp[0], back_row=back[0], t=0)

        for t in range(1, T + 1):
            aa = seq[t - 1]
            # first compute emitting transitions that consume aa
            for j, st in enumerate(self.states):
                if not st.emits:
                    continue
                emit = self._emit_logp(st.name, aa)
                if np.isneginf(emit):
                    continue

                candidates = dp[t - 1] + self.logA[:, j]
                i_best = int(np.argmax(candidates))
                best = float(candidates[i_best])
                if np.isneginf(best):
                    continue
                dp[t, j] = best + emit
                back[t, j] = i_best

            # then epsilon-closure at time t
            dp[t] = self._epsilon_closure(dp[t], back_row=back[t], t=t)

        end_idx = self.state_to_idx["E"]
        best_logp = float(dp[T, end_idx])

        # Reconstruct path (approximate, because epsilon-closure can skip explicit backpointers).
        # For typical use (mutation scoring) you won't need the full path, but we provide it.
        if np.isneginf(best_logp):
            return best_logp, []

        # naive traceback using recorded back pointers among emitting states only
        path_idx = [end_idx]
        cur_state = end_idx
        cur_t = T

        while cur_t >= 0 and cur_state != self.state_to_idx["B"]:
            prev_state = back[cur_t, cur_state]
            if prev_state == -1:
                break
            path_idx.append(prev_state)
            # if current state emits, we consumed a residue
            if self.states[cur_state].emits:
                cur_t -= 1
            cur_state = prev_state

        path_idx.reverse()
        path_names = [self.states[i].name for i in path_idx]
        return best_logp, path_names

    def _epsilon_closure(
        self,
        vec: np.ndarray,
        back_row: Optional[np.ndarray] = None,
        t: Optional[int] = None
    ) -> np.ndarray:
        """
        Propagate probabilities through non-emitting states (Delete + End) without consuming symbols.

        We do a relaxation over edges i->j where j is non-emitting (or generally, any edge that does not
        require emissions at j), until convergence. Because the profile topology is almost acyclic for
        deletes, and insert self-loops are emitting (handled elsewhere), this converges quickly.

        If back_row is provided (Viterbi), we update it when we improve a state score.
        """
        n_states = len(self.states)
        changed = True
        out = vec.copy()

        # Precompute which states are non-emitting
        non_emit = np.array([not st.emits for st in self.states], dtype=bool)

        while changed:
            changed = False
            for j in range(n_states):
                if not non_emit[j]:
                    continue
                candidates = out + self.logA[:, j]
                best_i = int(np.argmax(candidates))
                best_val = float(candidates[best_i])
                if best_val > out[j]:
                    out[j] = best_val
                    if back_row is not None and t is not None:
                        back_row[j] = best_i
                    changed = True

        return out

    # ---------- mutation scoring ----------
    def delta_match_emission_score(self, pos_1based: int, wt_aa: str, mut_aa: str) -> float:
        """
        Compute a simple conservation-based delta:
            Δ = log P(mut | M_pos) - log P(wt | M_pos)
        """
        if self.match_logE is None:
            raise RuntimeError("Model not built yet.")
        if not (1 <= pos_1based <= self.k):
            raise ValueError(f"pos_1based must be within 1..{self.k}, got {pos_1based}")

        wt_aa = wt_aa.upper()
        mut_aa = mut_aa.upper()
        if wt_aa not in AA_TO_INDEX or mut_aa not in AA_TO_INDEX:
            raise ValueError("wt_aa and mut_aa must be standard amino acids (A,C,...,Y).")

        i = pos_1based - 1
        wt_logp = float(self.match_logE[i, AA_TO_INDEX[wt_aa]])
        mut_logp = float(self.match_logE[i, AA_TO_INDEX[mut_aa]])
        return mut_logp - wt_logp

    def log_odds_match_score(self, pos_1based: int, aa: str) -> float:
        """
        Optional: log-odds against background at a match state:
            log P(aa | M_pos) - log P(aa | background)
        """
        if self.bg_logE is None:
            raise RuntimeError("Background not built.")
        aa = aa.upper()
        if aa not in AA_TO_INDEX:
            return -np.inf
        i = pos_1based - 1
        return float(self.match_logE[i, AA_TO_INDEX[aa]] - self.bg_logE[AA_TO_INDEX[aa]])


# -----------------------------
# TP53 pipeline wrapper
# -----------------------------

class TP53VariantScorer:
    """
    Convenience wrapper to:
      - build a ProfileHMM from an MSA
      - score missense variants from a ClinVar-like table
    """

    def __init__(self, msa_file: str, gap_threshold: float = 0.5, pseudocount: float = 1.0):
        self.msa_file = msa_file
        self.gap_threshold = gap_threshold
        self.pseudocount = pseudocount

        self.aln = None
        self.hmm = ProfileHMM(pseudocount=pseudocount)

    def fit(self) -> "TP53VariantScorer":
        self.aln = load_msa(self.msa_file)
        self.hmm.build_from_alignment(self.aln, gap_threshold=self.gap_threshold)
        print(f"Profile-HMM built. k(match states)={self.hmm.k}")
        print(f"Matrix shape (match emissions): {self.hmm.match_logE.shape}  # (k, 20)")
        print(f"Total states: {len(self.hmm.states)}")
        return self

    def score_variants_table(
        self,
        df: pd.DataFrame,
        protein_change_col: str = "ProteinChange",
        label_col: str = "ClinicalSignificance",
        benign_labels: Tuple[str, ...] = ("Benign", "Likely benign"),
        pathogenic_labels: Tuple[str, ...] = ("Pathogenic", "Likely pathogenic"),
    ) -> pd.DataFrame:
        """
        Adds columns:
          - pos, wt, mut
          - delta_score (Δ emission)
          - label_bin (0 benign, 1 pathogenic, NaN otherwise)
        """
        if self.hmm.match_logE is None:
            raise RuntimeError("Run fit() first.")

        rows = []
        for _, row in df.iterrows():
            pdot = row.get(protein_change_col, None)
            parsed = parse_clinvar_pdot(pdot)
            if parsed is None:
                rows.append((np.nan, None, None, np.nan))
                continue
            pos, wt, mut = parsed

            # Assumes pos maps to match state index.
            try:
                delta = self.hmm.delta_match_emission_score(pos, wt, mut)
            except Exception:
                delta = np.nan
            rows.append((pos, wt, mut, delta))

        out = df.copy()
        out[["pos", "wt", "mut", "delta_score"]] = pd.DataFrame(rows, index=df.index)

        def label_to_bin(x):
            if not isinstance(x, str):
                return np.nan
            x = x.strip()
            if x in benign_labels:
                return 0
            if x in pathogenic_labels:
                return 1
            return np.nan

        out["label_bin"] = out[label_col].apply(label_to_bin)
        return out


# -----------------------------
# Main (example usage)
# -----------------------------
def calculate_and_plot_roc(scored_df):
    """
    Calculates AUC and plots ROC curve.
    Assumes 'label_bin' is 0 for Benign, 1 for Pathogenic.
    Assumes 'delta_score' is lower for Pathogenic (so we invert it for ROC).
    """
    # סינון שורות ללא סיווג
    df = scored_df.dropna(subset=["delta_score", "label_bin"])

    if df.empty:
        print("Cannot calculate ROC: No valid labeled data.")
        return

    y_true = df["label_bin"]

    # טריק חשוב: ב-ROC אנחנו רוצים שהערך הגבוה ינבא 1 (פתוגני).
    # במודל שלנו, ציון *נמוך* (שלילי) אומר פתוגני.
    # לכן אנחנו הופכים את הסימן (מינוס) כדי שציון "שלילי מאוד" יהפוך ל"חיובי גבוה".
    y_scores = -df["delta_score"]

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    print(f"\n>>> ROC AUC Score: {roc_auc:.3f} <<<")

    # ציור הגרף
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (TP53)')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.savefig("roc_curve.png", dpi=200)
    plt.close()
    print("Saved ROC plot: roc_curve.png")



def main():
    # Update these paths as needed
    MSA_FILE = "tp53_30_msa.fasta"
    # Example: ClinVar-like CSV with columns: ProteinChange, ClinicalSignificance
    CLINVAR_CSV = "clinvar_tp53.txt"

    # 1. בניית המודל
    scorer = TP53VariantScorer(MSA_FILE, gap_threshold=0.5, pseudocount=1.0).fit()

    # Example sanity check: score likelihood for the reference ungapped sequence
    if scorer.hmm.reference_ungapped:
        ll = scorer.hmm.forward_log_likelihood(scorer.hmm.reference_ungapped)
        print(f"Forward log-likelihood of reference sequence: {ll:.2f}")

    # 2. טעינת ועיבוד ClinVar
    if CLINVAR_CSV:
        print(f"Loading ClinVar from {CLINVAR_CSV}...")
        try:
            # טעינה עם מפריד טאב (\t) כי זה הפורמט בקובץ שהעלית
            df = pd.read_csv(CLINVAR_CSV, sep='\t')
            print(f"Successfully loaded {len(df)} rows.")

            # ניקוי שמות עמודות (להסרת רווחים אם יש)
            df.columns = df.columns.str.strip()

            # --- תיקון 1: חילוץ המוטציה מתוך עמודת Name ---
            def extract_hgvs_from_name(name_val):
                if not isinstance(name_val, str): return None
                # מחפש ביטוי שמתחיל ב-p. ואחריו אותיות ומספרים, אופציונלית בתוך סוגריים
                match = re.search(r"\(?(p\.[A-Za-z]+\d+[A-Za-z]+)\)?", name_val)
                if match:
                    return match.group(1)  # מחזיר למשל p.Asp393Tyr
                return None

            if "Name" in df.columns:
                print("Extracting protein changes from 'Name' column...")
                df["extracted_pdot"] = df["Name"].apply(extract_hgvs_from_name)
            else:
                print("Error: 'Name' column not found.")
                return

            # --- תיקון 2: הגדרת עמודת הסיווג והערכים ---
            label_col = "Germline classification"
            if label_col not in df.columns:
                print(f"Error: Column '{label_col}' not found. Available: {list(df.columns)}")
                return

            # עדכון רשימת הערכים הפתוגניים
            pathogenic_labels = (
                "Pathogenic",
                "Likely pathogenic",
                "Pathogenic/Likely pathogenic",
                "Pathogenic/Likely pathogenic/Pathogenic, low penetrance"
            )

            # הרצת הסקור - קריאה אחת בלבד עם הפרמטרים הנכונים
            scored = scorer.score_variants_table(
                df,
                protein_change_col="extracted_pdot",
                label_col=label_col,
                pathogenic_labels=pathogenic_labels
            )

            # 3. יצירת הגרף ושמירה (בתוך ה-try כדי לוודא ש-scored קיים)
            plot_df = scored.dropna(subset=["delta_score", "label_bin"])

            if plot_df.empty:
                print("Warning: No valid data points to plot.")
            else:
                plt.figure(figsize=(8, 4))
                plt.hist(plot_df.loc[plot_df["label_bin"] == 0, "delta_score"], bins=50, alpha=0.7, label="Benign",
                         color='green')
                plt.hist(plot_df.loc[plot_df["label_bin"] == 1, "delta_score"], bins=50, alpha=0.7, label="Pathogenic",
                         color='red')
                plt.title("Δ emission score distribution (mut vs wt at Match position)")
                plt.xlabel("Δ score")
                plt.ylabel("Count")
                plt.legend()
                plt.tight_layout()
                plt.savefig("delta_score_hist.png", dpi=200)
                plt.close()
                print("Saved plot: delta_score_hist.png")

            calculate_and_plot_roc(scored)

            scored.to_csv("clinvar_scored.csv", index=False)
            print("Saved table: clinvar_scored.csv")

        except Exception as e:
            print(f"Error processing ClinVar file: {e}")
            import traceback
            traceback.print_exc()
            return


if __name__ == "__main__":
    main()