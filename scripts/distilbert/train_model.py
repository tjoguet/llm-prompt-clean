import sys
import os

# Ajouter le répertoire racine au Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    """Script principal d'entraînement."""
    from src.distilbert.complexity_scorer import ComplexityScorer
    from src.distilbert.trainer import ComplexityDataset  # Import manquant
    from src.distilbert.trainer import ComplexityTrainer  # Import manquant
    from transformers import AutoTokenizer
    import json
    import random
    
    # Configuration
    MODEL_NAME = "distilbert-base-uncased"
    FINAL_DATA_PATH = "final_data.jsonl"
    BATCH_SIZE = 32
    LEARNING_RATE = 2e-5
    NUM_EPOCHS = 4
    TRAIN_RATIO = 0.8  # 80% pour train, 20% pour val
    
    # Vérifier que le fichier de données existe
    if not os.path.exists(FINAL_DATA_PATH):
        print(f"❌ Fichier {FINAL_DATA_PATH} introuvable!")
        print("💡 Lancez d'abord: python3 data/scripts/get_final_data.py")
        return
    
    # Charger et diviser les données
    print("📝 Chargement et division des données...")
    data = []
    with open(FINAL_DATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    
    # Mélanger les données
    random.shuffle(data)
    
    # Diviser en train/val
    split_idx = int(len(data) * TRAIN_RATIO)
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    
    # Créer les fichiers temporaires
    train_path = "temp_train.jsonl"
    val_path = "temp_val.jsonl"
    
    # Écrire les données d'entraînement
    with open(train_path, "w", encoding="utf-8") as f:
        for item in train_data:
            # Adapter le format pour le trainer
            train_item = {
                "query": item["prompt"],
                "complexity_score": item["scaled_similarity"]  # Utiliser scaled_similarity
            }
            f.write(json.dumps(train_item, ensure_ascii=False) + "\n")
    
    # Écrire les données de validation
    with open(val_path, "w", encoding="utf-8") as f:
        for item in val_data:
            # Adapter le format pour le trainer
            val_item = {
                "query": item["prompt"],
                "complexity_score": item["scaled_similarity"]  # Utiliser scaled_similarity
            }
            f.write(json.dumps(val_item, ensure_ascii=False) + "\n")
    
    # Initialisation
    print("🔧 Initialisation...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = ComplexityScorer(MODEL_NAME)
    
    # Datasets
    print("📝 Chargement des datasets...")
    train_dataset = ComplexityDataset(train_path, tokenizer)
    val_dataset = ComplexityDataset(val_path, tokenizer)
    
    print(f"📊 {len(train_dataset)} échantillons train, {len(val_dataset)} échantillons val")
    
    # Trainer
    trainer = ComplexityTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        num_epochs=NUM_EPOCHS
    )
    
    # Entraînement
    trainer.train()
    
    # Nettoyer les fichiers temporaires
    os.remove(train_path)
    os.remove(val_path)
    print("🧹 Fichiers temporaires supprimés")

if __name__ == "__main__":
    main()