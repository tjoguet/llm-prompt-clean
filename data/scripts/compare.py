import json
import os
import time
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tqdm import tqdm
import statistics

# Charger client OpenAI
client = OpenAI(api_key="api")

# Fichiers d'entrée
file_nano = "/Users/tim/Desktop/Code/llm-prompt-clean/data/OpenAssistant_oasst1/responses_4_1_nano.jsonl"
file_nano_filtered = "/Users/tim/Desktop/Code/llm-prompt-clean/data/OpenAssistant_oasst1/filtered_large.jsonl"
file_big = "/Users/tim/Desktop/Code/llm-prompt-clean/data/OpenAssistant_oasst1/responses_4_1_big.jsonl"
output_file = "similarities_large.jsonl"

# Charger les réponses dans des dicts {id: response}
def load_responses(filepath):
    responses = {}
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            responses[data["id"]] = data["response"]
    return responses

# Étape de filtrage : ne garder que les réponses du gros fichier dont l'id est dans le petit fichier
filtered_large_file = "filtered_large.jsonl"
def filter_large_file(larger_file, smaller_file):
    small_ids = set()
    with open(smaller_file, "r", encoding="utf-8") as f_small:
        for line in f_small:
            data = json.loads(line)
            small_ids.add(data["id"])
    with open(larger_file, "r", encoding="utf-8") as f_large, \
         open(filtered_large_file, "w", encoding="utf-8") as f_out:
        for line in f_large:
            data = json.loads(line)
            if data["id"] in small_ids:
                f_out.write(json.dumps(data, ensure_ascii=False) + "\n")

# Obtenir embedding via l'API
def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-large",
        input=text
    )
    return response.data[0].embedding

# Calculer similarité cosinus
def cosine_sim(vec1, vec2):
    return cosine_similarity([vec1], [vec2])[0][0]

# Traitement principal
def main():
    # Étape de filtrage
    # filter_large_file(file_nano, file_big)

    # Charger les réponses
    small_responses = load_responses(file_nano_filtered)
    large_responses = load_responses(file_big)
    ids = set(small_responses.keys()) & set(large_responses.keys())
    

    # Charger les IDs déjà traités
    already_done_ids = set()
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    already_done_ids.add(data["id"])
                except:
                    pass  # Ligne corrompue

    # Reste des IDs à traiter
    remaining_ids = sorted(ids - already_done_ids)

    print(f"{len(already_done_ids)} IDs déjà traités.")
    print(f"{len(remaining_ids)} IDs à traiter.")

    similarities = []
    with open(output_file, "a", encoding="utf-8") as f_out:
        for id_ in tqdm(remaining_ids):
            try:
                emb_small = get_embedding(small_responses[id_])
                emb_large = get_embedding(large_responses[id_])
                similarity = cosine_sim(emb_small, emb_large)
                similarities.append(similarity)
                f_out.write(json.dumps({
                    "id": id_,
                    "similarity": similarity
                }, ensure_ascii=False) + "\n")
                f_out.flush()
                time.sleep(0.2)  # éviter les limites API
            except Exception as e:
                print(f"[ERROR] {id_}: {e}")

    # Calculer et afficher les stats globales
    if similarities:
        mean_similarity = np.mean(similarities)
        median_similarity = np.median(similarities)
        print(f"\nStatistiques des nouvelles similarités:")
        print(f"Moyenne: {mean_similarity:.4f}")
        print(f"Médiane: {median_similarity:.4f}")
        print(f"Nombre de paires comparées: {len(similarities)}")
    else:
        print("Aucune similarité calculée.")

if __name__ == "__main__":
    main()