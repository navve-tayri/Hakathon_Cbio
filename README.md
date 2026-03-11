# Patho-HMM: Distinguishing Pathogenic from Benign TP53 Mutations 

**Navve Tayri, Julia Kestenbaum, Avital Schwartz, Tamar Goldfarb, Hadar Koren** *Algorithms in Computational Biology | Hebrew University*

---

## Project Overview
TP53, known as the "guardian of the genome," is the most frequently mutated gene in human cancer. This project utilizes **Profile Hidden Markov Models (Profile HMMs)** to distinguish between pathogenic driver mutations and benign passenger variants based on evolutionary conservation. 

Our core hypothesis is that pathogenicity correlates with statistical improbability: mutations in strictly conserved regions disrupt protein function and yield lower log-likelihood scores compared to the "healthy" evolutionary profile.

## Methodology & Architecture
We compared three distinct models to evaluate performance:
* **Model A (PFAM):** A curated profile of the p53 DNA-binding domain (PF00870).
* **Model B (HMMER):** A locally built model using HMMER v3.3 trained on vertebrate p53 orthologs.
* **Model C (Custom):** A self-developed Python implementation featuring a unique **Delta Score** ($LL_{WT} - LL_{variant}$) strategy.

### Technical Specifications (Model C)
* **Gap Threshold ($\theta$):** Columns with < 35% gaps are treated as consensus Match/Delete states.
* **Numerical Stability:** Calculations performed in log-space with $\epsilon$-smoothing ($1 \times 10^{-300}$) to handle unseen amino acids.
* **Optimization:** Classification thresholds derived using **Youden’s Index** to maximize sensitivity and specificity.

---

## Repository Structure
* `C_model/`: Core Python implementation of the HMM and Viterbi algorithm.
* `A_B_models/`: Benchmarking scripts and industry-standard `.hmm` files.
* `Data/`: Training alignments (vertebrate orthologs) and labeled test sets (ClinVar/TP53_PROF).
* `MSA_conservation/`: Entropy-based conservation analysis.

---

## Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Running the Custom Model
To train the model and evaluate it against the labeled dataset:
```bash
python C_model/Final_model_C.py
```
### Running HMMER Benchmarks
To visualize Bit Score distributions for PFAM:
```bash
python A_B_models/boxplot_tblout.py --tbl A_B_models/scores_pfam.tbl --score neg_bitscore
```

### Performance Summary
Our custom Model C functions as a high-sensitivity filter, identifying nearly all pathogenic variants (Recall > 94% on ClinVar) by aggressively penalizing evolutionary deviations through the Delta Score.


### References
* **UniProtKB:** P53_HUMAN (P04637) sequence for calibration.
* **ClinVar:** Clinical significance annotations for human TP53 variants. 
* **HMMER3 User Guide:** Profile HMM training and structure. 
