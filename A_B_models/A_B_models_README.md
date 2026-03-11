# Models A & B: Industry Benchmarks

This folder manages comparison with the HMMER3 software suite.

### Profiles Used
* `PF00870.hmm`: Pre-trained DNA-binding domain profile.
* `tp53_from_orthologs.hmm`: Custom-built profile using HMMER's `hmmbuild` on our training alignment.

### Analysis Scripts
* `boxplot_tblout.py`: Parses HMMER's tabular output (`--tblout`). It can binarize ClinVar labels and negate bitscores so that higher values represent worse evolutionary fit (pathogenicity).