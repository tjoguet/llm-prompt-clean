from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import json
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from xgboost import XGBRegressor, callback
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm
import warnings
import os
import pickle

def load_data_from_final_data():
    prompts = []
    scores = []
    
    with open("/Users/tim/Desktop/Code/llm-prompt-clean/data/OpenAssistant_oasst1/final_data_large.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            prompts.append(data["prompt"])
            scores.append(data["original_similarity"])
    
    return prompts, np.array(scores)

def get_or_create_embeddings(prompts, encoder):
    embeddings_file = "/Users/tim/Desktop/Code/llm-prompt-clean/data/OpenAssistant_oasst1/embeddings_e5.pkl"
    
    if os.path.exists(embeddings_file):
        print("📂 Chargement des embeddings existants...")
        with open(embeddings_file, "rb") as f:
            embeddings = pickle.load(f)
        print(f"   Embeddings chargés: {embeddings.shape}")
        return embeddings
    else:
        print("📝 Calcul des embeddings...")
        formatted_prompts = [f"query: {p}" for p in prompts]
        embeddings = encoder.encode(formatted_prompts, show_progress_bar=True)
        
        os.makedirs("models", exist_ok=True)
        with open(embeddings_file, "wb") as f:
            pickle.dump(embeddings, f)
        print(f"   Embeddings sauvegardés: {embeddings.shape}")
        return embeddings

def bin_scores(scores, n_bins=3):
    bins = np.linspace(0, 1, n_bins + 1)
    labels = np.digitize(scores, bins) - 1  # -1 pour que ça commence à 0
    return labels

def label_scores(scores):
    labels = np.zeros_like(scores, dtype=int)
    labels[scores > 0.88] = 0       # facile
    labels[scores <= 0.88] = 1     # difficile
    return labels

def train_xgboost_classifier():
    print("📊 Chargement des données...")
    prompts, scores = load_data_from_final_data()
    print(f"   {len(prompts)} échantillons chargés")
    print(f"   Scores entre {scores.min():.3f} et {scores.max():.3f}")
    
    # Convertir en labels
    # labels = bin_scores(scores, n_bins=3)  # [0, 1, 2]
    labels = label_scores(scores)
    print(f"   Labels uniques: {np.unique(labels)}")
    print(f"nombre de labels faciles: {np.sum(labels == 0)}")
    print(f"nombre de labels moyens: {np.sum(labels == 1)}")
    print(f"nombre de labels difficiles: {np.sum(labels == 2)}")

    
    print("\n🔧 Chargement du modèle E5-small-v2...")
    encoder = SentenceTransformer("intfloat/e5-small-v2")
    X = get_or_create_embeddings(prompts, encoder)
    
    print("\n🔄 Division train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print("\n🚀 Entraînement XGBoost (classification)...")
    n_estimators = 300
    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=4,
        learning_rate=0.05,
        reg_lambda=1.0,
        objective="multi:softprob",
        num_class=2,
        eval_metric="mlogloss",  # <-- ici maintenant
        random_state=42,
        verbosity=1
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=50
    )
    
    print("\n📈 Évaluation du modèle (classification)...")

    print("Format y_test:", y_test.shape, "Unique:", np.unique(y_test))

    # y_pred est ici un tableau (n_samples, 2) avec des probabilités => convertis en labels
    y_pred_proba = model.predict(X_test)
    print("Format y_pred (proba):", y_pred_proba.shape)

    y_pred = np.argmax(y_pred_proba, axis=1)  # prendre la classe avec la probabilité max
    print("Format y_pred (labels):", y_pred.shape, "Unique:", np.unique(y_pred))

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    print(f"   Accuracy: {acc:.4f}")
    print(f"   F1-score: {f1:.4f}")
    print("\nClassification report:\n")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    print("Matrice de confusion:")
    print(cm)

    print("\n💾 Sauvegarde du modèle...")
    os.makedirs("models", exist_ok=True)
    model.save_model("models/xgboost_complexity_classifier.json")
    print("   Modèle sauvegardé dans models/xgboost_complexity_classifier.json")
    
    
    return model, (acc, f1)

if __name__ == "__main__":
    model, metrics = train_xgboost_classifier()
    print(f"\n🎉 Entraînement terminé !")
    print(f"   Accuracy: {metrics[0]:.4f}")
    print(f"   F1-score: {metrics[1]:.4f}")