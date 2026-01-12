from Bio import SeqIO

input_file = "tp53_clinvar_labeled.fasta"
def process_clinvar_data(input_file):
    # הגדרת ה-IDs של 6 הרצפים כפי שהם מופיעים בקובץ המקורי
    conserved_ids = ["TP53_R136H", "TP53_R209Q", "TP53_G206S"]
    non_conserved_ids = ["TP53_P33R", "TP53_P4L", "TP53_D354Y"]

    all_target_ids = conserved_ids + non_conserved_ids

    # קריאת כל הרצפים מהקובץ המקורי
    all_records = list(SeqIO.parse(input_file, "fasta"))

    example_records = []
    remaining_records = []

    # מילוי רשימת ה-examples לפי הסדר המדויק שביקשת
    for target in all_target_ids:
        for rec in all_records:
            if rec.id.split('|')[0] == target:
                # ניקוי ה-ID וה-Description (הסרת ה-label)
                clean_name = target
                rec.id = clean_name
                rec.name = clean_name
                rec.description = clean_name
                example_records.append(rec)
                break

    # מילוי רשימת היתרה (כל מה שלא ב-6 הרצפים)
    for rec in all_records:
        current_id = rec.id.split('|')[0]
        if current_id not in all_target_ids:
            remaining_records.append(rec)

    # שמירת הקובץ הראשון: 6 הדוגמאות
    if example_records:
        SeqIO.write(example_records, "tp53_clinvar_examples.fasta", "fasta")
        print(f"✅ נוצר קובץ דוגמאות: 'tp53_clinvar_examples.fasta' ({len(example_records)} רצפים)")

    # שמירת הקובץ השני: שאר הנתונים למודל
    if remaining_records:
        SeqIO.write(remaining_records, "tp53_clinvar_data_for_model.fasta", "fasta")
        print(f"✅ נוצר קובץ למודל: 'tp53_clinvar_data_for_model.fasta' ({len(remaining_records)} רצפים)")


# הרצה על הקובץ שלך
process_clinvar_data("tp53_clinvar_labeled.fasta")