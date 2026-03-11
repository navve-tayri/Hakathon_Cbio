import re
import pandas as pd


def load_fasta(filename):
    """
    קורא קובץ FASTA ומחזיר רשימה של רצפים (מחרוזות).
    """
    sequences = []
    current_seq = []

    try:
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith(">"):
                    if current_seq:
                        sequences.append("".join(current_seq))
                        current_seq = []
                else:
                    current_seq.append(line)
            if current_seq:
                sequences.append("".join(current_seq))
    except FileNotFoundError:
        print(f"Error: File {filename} not found.")
        return []

    return sequences


def parse_mutation_string(mut_str):
    """
    מפרק מחרוזת כמו 'P4L' או 'R175H' לרכיבים: ('P', 3, 'L')
    """
    # אם יש רשימה (למשל "D234Y, D261Y"), ניקח רק את הראשון
    if "," in mut_str:
        mut_str = mut_str.split(",")[0].strip()

    # חיפוש תבנית של אות-מספר-אות (למשל P4L)
    match = re.match(r"([A-Z])(\d+)([A-Z])", mut_str)
    if match:
        orig, pos, new = match.groups()
        return orig, int(pos) - 1, new  # המרה לאינדקס 0
    return None, None, None


def load_clinvar(filename):
    """
    טוען את קובץ ה-ClinVar בפורמט TSV (מופרד בטאבים).
    """
    try:
        # קריאה עם מפריד טאב (\t)
        df = pd.read_csv(filename, sep='\t', engine='python')

        # הדפסת העמודות כדי לוודא שהכל תקין
        print(f"DEBUG: Columns found: {list(df.columns)}")

        # חיפוש העמודות לפי השמות המדויקים שראינו בקובץ שלך
        mut_col = 'Protein change'
        sig_col = 'Germline classification'

        if mut_col not in df.columns or sig_col not in df.columns:
            print(f"Error: Missing columns. Looking for '{mut_col}' and '{sig_col}'")
            return []

        mutations = []
        for index, row in df.iterrows():
            mut_str = str(row[mut_col]).strip()
            label = str(row[sig_col]).strip()

            # דילוג על שורות ריקות או חסרות משמעות
            if mut_str == 'nan' or not mut_str:
                continue

            orig, pos, new_aa = parse_mutation_string(mut_str)

            if orig and pos is not None:
                mutations.append({
                    'name': mut_str,
                    'pos': pos,
                    'orig': orig,
                    'new': new_aa,
                    'label': label
                })

        print(f"Successfully loaded {len(mutations)} mutations.")
        return mutations

    except FileNotFoundError:
        print(f"Error: File {filename} not found.")
        return []
    except Exception as e:
        print(f"Error reading ClinVar file: {e}")
        return []