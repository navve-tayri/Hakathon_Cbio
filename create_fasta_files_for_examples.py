from Bio import SeqIO

def process_clinvar_data(input_file):
    # הגדרת ה-IDs של 8 הרצפים (2 מכל קטגוריה)
    # סדר: 2 שמור-פתוגני, 2 שמור-לא פתוגני, 2 לא שמור-פתוגני, 2 לא שמור-לא פתוגני
    example_targets = [
        "TP53_R136H", "TP53_R209Q",  # שמור - פתוגני
        "TP53_V235I", "TP53_N196S",  # שמור - לא פתוגני
        "TP53_E12D",  "TP53_R26L",   # לא שמור - פתוגני
        "TP53_P4L",   "TP53_M345T"   # לא שמור - לא פתוגני
    ]

    # קריאת כל הרצפים מהקובץ המקורי
    all_records = list(SeqIO.parse(input_file, "fasta"))

    example_records = []
    remaining_records = []

    # 1. שליפה וניקוי של 8 הדוגמאות לפי הסדר המוגדר
    for target in example_targets:
        for rec in all_records:
            # פיצול ה-ID כדי לבדוק התאמה לפני סימן ה-|
            current_id_clean = rec.id.split('|')[0]
            if current_id_clean == target:
                # יצירת עותק חדש וניקוי השם (הסרת ה-label מה-header)
                new_rec = rec
                new_rec.id = target
                new_rec.name = target
                new_rec.description = target
                example_records.append(new_rec)
                break

    # 2. איסוף כל שאר הרצפים לקובץ המודל (סינון ה-8 שהוצאנו)
    for rec in all_records:
        current_id_clean = rec.id.split('|')[0]
        if current_id_clean not in example_targets:
            remaining_records.append(rec)

    # שמירת הקובץ הראשון: 8 הדוגמאות
    if example_records:
        SeqIO.write(example_records, "tp53_clinvar_8_examples.fasta", "fasta")
        print(f"✅ נוצר קובץ דוגמאות: 'tp53_clinvar_8_examples.fasta' ({len(example_records)} רצפים)")

    # שמירת הקובץ השני: שאר הנתונים למודל
    if remaining_records:
        SeqIO.write(remaining_records, "tp53_clinvar_data_for_model_without_8.fasta", "fasta")
        print(f"✅ נוצר קובץ למודל: 'tp53_clinvar_data_for_model_without_8.fasta' ({len(remaining_records)} רצפים)")

# הרצה על הקובץ שלך
process_clinvar_data("tp53_clinvar_labeled.fasta")