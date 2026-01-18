from Bio import SeqIO
import os

input_file = "../Data/tp53_orthologs_raw.fasta"
output_file = "../Data/tp53_orthologs_clean.fasta"




# To track unique sequences
seen_sequences = set()
clean_records = []

print(f"Reading {input_file}...")

# Process each record in the input FASTA file
for record in SeqIO.parse(input_file, "fasta"):
    seq_str = str(record.seq).upper()

    # Filter 1: Remove sequences with non-standard amino acids
    if seq_str.count("X") > (len(seq_str) * 0.01):
        continue

    # Filter 2: Remove sequences with length deviations > 5%
    if seq_str not in seen_sequences:
        seen_sequences.add(seq_str)
        clean_records.append(record)

# Write the cleaned records to the output FASTA file
SeqIO.write(clean_records, output_file, "fasta")

print("-" * 30)
print(f"Original sequences: {296}")
print(f"Clean sequences:    {len(clean_records)}")
print(f"Saved to: {output_file}")
print("-" * 30)
print("Ready for Alignment!")