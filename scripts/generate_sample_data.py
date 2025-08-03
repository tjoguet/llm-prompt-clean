"""Script pour générer des données d'exemple pour tester le système."""

import json
import os
from typing import List, Dict

def create_sample_data() -> List[Dict]:
    """Crée des données d'exemple avec différents niveaux de complexité."""
    
    sample_data = [
        # Requêtes très simples (score ~0.1)
        {
            "query": "Hello",
            "complexity_score": 0.05,
            "confidence": 0.95,
            "metadata": {"category": "greeting", "length": 5}
        },
        {
            "query": "Hi there!",
            "complexity_score": 0.08,
            "confidence": 0.92,
            "metadata": {"category": "greeting", "length": 9}
        },
        {
            "query": "Yes",
            "complexity_score": 0.03,
            "confidence": 0.98,
            "metadata": {"category": "confirmation", "length": 3}
        },
        {
            "query": "Thanks",
            "complexity_score": 0.04,
            "confidence": 0.96,
            "metadata": {"category": "gratitude", "length": 6}
        },
        
        # Requêtes simples (score ~0.2-0.3)
        {
            "query": "What time is it?",
            "complexity_score": 0.15,
            "confidence": 0.88,
            "metadata": {"category": "factual_simple", "length": 16}
        },
        {
            "query": "How are you?",
            "complexity_score": 0.12,
            "confidence": 0.90,
            "metadata": {"category": "social", "length": 12}
        },
        {
            "query": "What's the weather like?",
            "complexity_score": 0.22,
            "confidence": 0.85,
            "metadata": {"category": "information", "length": 24}
        },
        {
            "query": "Can you help me?",
            "complexity_score": 0.18,
            "confidence": 0.87,
            "metadata": {"category": "request_help", "length": 16}
        },
        
        # Requêtes moyennes (score ~0.4-0.6)
        {
            "query": "Explain how photosynthesis works",
            "complexity_score": 0.45,
            "confidence": 0.82,
            "metadata": {"category": "science_explanation", "length": 34}
        },
        {
            "query": "What are the main causes of climate change?",
            "complexity_score": 0.52,
            "confidence": 0.78,
            "metadata": {"category": "science_complex", "length": 44}
        },
        {
            "query": "How do I cook pasta properly?",
            "complexity_score": 0.35,
            "confidence": 0.83,
            "metadata": {"category": "instruction", "length": 29}
        },
        {
            "query": "Compare Python and JavaScript for web development",
            "complexity_score": 0.58,
            "confidence": 0.75,
            "metadata": {"category": "technical_comparison", "length": 50}
        },
        
        # Requêtes complexes (score ~0.7-0.8)
        {
            "query": "Analyze the economic implications of artificial intelligence on labor markets in developing countries",
            "complexity_score": 0.75,
            "confidence": 0.72,
            "metadata": {"category": "analysis_complex", "length": 104}
        },
        {
            "query": "Explain quantum entanglement and its applications in quantum computing with mathematical formulations",
            "complexity_score": 0.82,
            "confidence": 0.68,
            "metadata": {"category": "science_advanced", "length": 108}
        },
        {
            "query": "Provide a detailed business plan for a sustainable energy startup including market analysis and financial projections",
            "complexity_score": 0.78,
            "confidence": 0.70,
            "metadata": {"category": "business_complex", "length": 122}
        },
        
        # Requêtes très complexes (score ~0.9+)
        {
            "query": "Develop a comprehensive framework for evaluating the ethical implications of AI decision-making systems in healthcare, considering cultural differences, legal frameworks, and philosophical perspectives across different societies",
            "complexity_score": 0.95,
            "confidence": 0.65,
            "metadata": {"category": "philosophical_complex", "length": 224}
        },
        {
            "query": "Design a multi-layered neural network architecture for real-time processing of multimodal data streams while ensuring privacy preservation and explaining the mathematical foundations",
            "complexity_score": 0.92,
            "confidence": 0.63,
            "metadata": {"category": "technical_advanced", "length": 180}
        },
        
        # Requêtes ambiguës (confiance faible)
        {
            "query": "Tell me about it",
            "complexity_score": 0.30,
            "confidence": 0.45,
            "metadata": {"category": "ambiguous", "length": 17}
        },
        {
            "query": "What do you think about that thing?",
            "complexity_score": 0.25,
            "confidence": 0.40,
            "metadata": {"category": "vague", "length": 35}
        },
        
        # Requêtes techniques spécialisées
        {
            "query": "Implement a distributed consensus algorithm using Raft protocol",
            "complexity_score": 0.85,
            "confidence": 0.80,
            "metadata": {"category": "technical_implementation", "length": 62}
        },
        {
            "query": "Debug this Python code",
            "complexity_score": 0.40,
            "confidence": 0.70,
            "metadata": {"category": "technical_help", "length": 20}
        }
    ]
    
    return sample_data

def save_data_splits(data: List[Dict], train_ratio: float = 0.7, val_ratio: float = 0.2):
    """Divise les données en train/val/test et les sauvegarde."""
    
    # Créer les dossiers
    os.makedirs("data/processed", exist_ok=True)
    
    # Mélanger les données
    import random
    random.seed(42)
    random.shuffle(data)
    
    # Calculer les indices de division
    n = len(data)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    # Diviser
    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]
    
    # Sauvegarder
    datasets = {
        "train": train_data,
        "val": val_data,
        "test": test_data
    }
    
    for split_name, split_data in datasets.items():
        file_path = f"data/processed/{split_name}.jsonl"
        with open(file_path, 'w', encoding='utf-8') as f:
            for item in split_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        print(f"✅ {split_name}.jsonl créé : {len(split_data)} échantillons")
    
    return len(train_data), len(val_data), len(test_data)

def main():
    """Génère et sauvegarde les données d'exemple."""
    print("🎯 Génération des données d'exemple...")
    
    # Créer les données
    sample_data = create_sample_data()
    
    # Dupliquer pour avoir plus de données (pour tester)
    extended_data = sample_data * 5  # 100 échantillons total
    
    # Ajouter un peu de variabilité
    import random
    random.seed(42)
    for item in extended_data:
        # Ajouter un peu de bruit aux scores
        noise = random.uniform(-0.05, 0.05)
        item['complexity_score'] = max(0.0, min(1.0, item['complexity_score'] + noise))
        
        # Ajuster la confiance si nécessaire
        if item['complexity_score'] < 0.1 or item['complexity_score'] > 0.9:
            item['confidence'] = max(0.6, item['confidence'])
    
    # Sauvegarder
    train_size, val_size, test_size = save_data_splits(extended_data)
    
    print(f"📊 Données générées :")
    print(f"   Train: {train_size} échantillons")
    print(f"   Val: {val_size} échantillons") 
    print(f"   Test: {test_size} échantillons")
    print(f"   Total: {len(extended_data)} échantillons")
    
    # Afficher quelques exemples
    print(f"\n📝 Exemples générés :")
    for i, item in enumerate(sample_data[:3]):
        print(f"{i+1}. '{item['query']}' -> Score: {item['complexity_score']:.2f}, Conf: {item['confidence']:.2f}")

if __name__ == "__main__":
    main()