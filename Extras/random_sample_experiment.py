import requests
import random
from Bio import SeqIO
from io import StringIO

# 1. הורדת כל הרצפים האיכותיים (Reviewed Vertebrates)
query = "gene:tp53 AND taxonomy_id:7742 AND reviewed:true"
url = "https://rest.uniprot.org/uniprotkb/stream"
params = {
    "format": "fasta",
    "query": query,
    "includeIsoform": "false"
}

print("Fetching ALL reviewed vertebrate sequences from UniProt...")
response = requests.get(url, params=params)

if response.status_code != 200:
    print("Error downloading.")
    exit()

# טעינת הרצפים לזיכרון
all_sequences = list(SeqIO.parse(StringIO(response.text), "fasta"))
total_count = len(all_sequences)
print(f"Total reviewed sequences found: {total_count}")

# 2. בחירה רנדומלית של 30 רצפים
print("Selecting 30 random sequences...")
random_30 = random.sample(all_sequences, 30)

# 3. שמירה לקובץ
output_file = "tp53_random_30.fasta"
with open(output_file, "w") as f:
    SeqIO.write(random_30, f, "fasta")

print(f"✅ Saved 30 random sequences to '{output_file}'")
print("⚠️ DON'T FORGET: You must ALIGN this file before running the model!")