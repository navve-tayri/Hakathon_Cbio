import pandas as pd
import numpy as np
from Bio import AlignIO


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

if __name__ == "__main__":
    aln = AlignIO.read("tp53_msa.fasta", "fasta")
    alignment = [str(rec.seq).upper() for rec in aln]  # list[str]

    AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
    alphabet = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
    theta = 0.35

    states, transition_matrix, emission_matrix = profileHMM(alignment, alphabet, theta)

    eps = 1e-300  # למנוע log(0)
    T_log = np.log(np.maximum(transition_matrix, eps))
    E_log = np.log(np.maximum(emission_matrix, eps))

    print("\nStates:\n", states)
    print("\nTransition Matrix:\n", pd.DataFrame(transition_matrix, index=states, columns=states))
    print("\nEmission Matrix:\n", pd.DataFrame(emission_matrix, index=states, columns=list(alphabet.keys())))