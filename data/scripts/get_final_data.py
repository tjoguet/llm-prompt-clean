import json
import numpy as np
import statistics
from sklearn.preprocessing import StandardScaler

def main():
    similarities = []
    similarity_data = []
    
    # Lire le fichier similarities.jsonl
    with open("/Users/tim/Desktop/Code/llm-prompt-clean/data/OpenAssistant_oasst1/similarities_large.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            similarities.append(data["similarity"])
            similarity_data.append(data)
    
    # Normaliser les similarités avec StandardScaler
    sim_scores = np.array(similarities).reshape(-1, 1)
    scaler = StandardScaler()
    scaled_scores = scaler.fit_transform(sim_scores)
    
    # Charger les réponses du gros fichier pour récupérer les prompts
    responses_big = {}
    with open("/Users/tim/Desktop/Code/llm-prompt-clean/data/OpenAssistant_oasst1/responses_4_1_big.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            responses_big[data["id"]] = data["prompt"]
    
    # Créer le fichier final_data.jsonl avec id, prompt original, et similarités normalisées
    with open("final_data_large.jsonl", "w", encoding="utf-8") as f_out:
        for i, sim_data in enumerate(similarity_data):
            id_ = sim_data["id"]
            original_similarity = sim_data["similarity"]
            scaled_similarity = scaled_scores[i][0]
            original_prompt = responses_big.get(id_, "Prompt non trouvé")
            
            final_data = {
                "id": id_,
                "prompt": original_prompt,
                "original_similarity": original_similarity,
                "scaled_similarity": scaled_similarity
            }
            f_out.write(json.dumps(final_data, ensure_ascii=False) + "\n")
    
    # Calculer et afficher les statistiques
    if similarities:
        mean_similarity = np.mean(similarities)
        median_similarity = np.median(similarities)
        mean_scaled = np.mean(scaled_scores)
        median_scaled = np.median(scaled_scores)
        
        print(f"\nStatistiques des similarités:")
        print(f"Originales - Moyenne: {mean_similarity:.4f}, Médiane: {median_similarity:.4f}")
        print(f"Normalisées - Moyenne: {mean_scaled:.4f}, Médiane: {median_scaled:.4f}")
        print(f"Nombre de paires comparées: {len(similarities)}")
        print(f"Fichier final_data.jsonl créé avec succès!")
    else:
        print("Aucune similarité trouvée dans le fichier.")

if __name__ == "__main__":
    main()