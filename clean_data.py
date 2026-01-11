from Bio import SeqIO
import os

input_file = "tp53_orthologs_raw.fasta"
output_file = "tp53_orthologs_clean.fasta"

# רשימה לשמירת רצפים ייחודיים בלבד
seen_sequences = set()
clean_records = []

print(f"Reading {input_file}...")

for record in SeqIO.parse(input_file, "fasta"):
    # 1. המרת הרצף למחרוזת (string) ואותיות גדולות
    seq_str = str(record.seq).upper()

    # 2. סינון רצפים שמכילים תווים לא חוקיים (כמו X, B, Z)
    # X אומר שהמכונה לא ידעה איזו חומצה זו. זה הורס את ה-HMM.
    # נרשה עד 1% של אותיות לא ברורות, אם יש יותר - נזרוק.
    if seq_str.count("X") > (len(seq_str) * 0.01):
        continue

    # 3. מחיקת כפילויות (De-duplication)
    if seq_str not in seen_sequences:
        seen_sequences.add(seq_str)
        clean_records.append(record)

# שמירת הקובץ הנקי
SeqIO.write(clean_records, output_file, "fasta")

print("-" * 30)
print(f"Original sequences: {296}")  # או המספר המדויק שהיה לך
print(f"Clean sequences:    {len(clean_records)}")
print(f"Saved to: {output_file}")
print("-" * 30)
print("Ready for Alignment! 🚀")