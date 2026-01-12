from Bio import SeqIO
import copy


def process_clinvar_data(input_file):
    # הגדרת ה-IDs של 8 הרצפים (2 מכל קטגוריה)
    example_targets = [
        "TP53_R136H", "TP53_R209Q",  # שמור - פתוגני
        "TP53_V235I", "TP53_N196S",  # שמור - לא פתוגני
        "TP53_E12D", "TP53_R26L",  # לא שמור - פתוגני
        "TP53_P4L", "TP53_M345T"  # לא שמור - לא פתוגני
    ]

    # קריאת כל הרצפים מהקובץ המקורי
    all_records = list(SeqIO.parse(input_file, "fasta"))

    example_clean_records = []  # 8 רצפים בלי לייבלים
    example_labeled_records = []  # 8 רצפים עם לייבלים
    remaining_records = []  # שאר הרצפים (עם לייבלים)

    # 1. שליפה ומיון של 8 הדוגמאות
    for target in example_targets:
        for rec in all_records:
            current_id_clean = rec.id.split('|')[0]
            if current_id_clean == target:
                # יצירת עותק עם לייבל (שומר על ה-ID המקורי מהקובץ)
                labeled_rec = copy.deepcopy(rec)
                example_labeled_records.append(labeled_rec)

                # יצירת עותק נקי (הסרת ה-label)
                clean_rec = copy.deepcopy(rec)
                clean_rec.id = target
                clean_rec.name = target
                clean_rec.description = target
                example_clean_records.append(clean_rec)
                break

    # 2. איסוף כל שאר הרצפים (סינון ה-8 שהוצאנו)
    for rec in all_records:
        current_id_clean = rec.id.split('|')[0]
        if current_id_clean not in example_targets:
            remaining_records.append(rec)

    # שמירת הקבצים
    # קובץ 1: 8 דוגמאות ללא לייבלים (להרצה במודל)
    if example_clean_records:
        SeqIO.write(example_clean_records, "tp53_clinvar_8_examples_clean.fasta", "fasta")
        print(f"✅ נוצר קובץ: 'tp53_clinvar_8_examples_clean.fasta' (8 רצפים ללא לייבל)")

    # קובץ 2: 8 דוגמאות עם לייבלים (לצורך המצגת והשוואה)
    if example_labeled_records:
        SeqIO.write(example_labeled_records, "tp53_clinvar_8_examples_with_labels.fasta", "fasta")
        print(f"✅ נוצר קובץ: 'tp53_clinvar_8_examples_with_labels.fasta' (8 רצפים עם לייבל)")

    # קובץ 3: שאר הנתונים למודל (עם לייבלים)
    if remaining_records:
        SeqIO.write(remaining_records, "tp53_clinvar_data_for_model_without_8.fasta", "fasta")
        print(f"✅ נוצר קובץ: 'tp53_clinvar_data_for_model_without_8.fasta' ({len(remaining_records)} רצפים)")


# הרצה על הקובץ שלך
process_clinvar_data("tp53_clinvar_labeled.fasta")