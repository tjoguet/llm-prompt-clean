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
warnings.filterwarnings('ignore')

def load_data_from_final_data():
    prompts = []
    scores = []
    
    with open("/Users/tim/Desktop/Code/llm-prompt/data/OpenAssistant_oasst1/final_data.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            prompts.append(data["prompt"])
            scores.append(data["scaled_similarity"])
    
    return prompts, np.array(scores)

def get_or_create_embeddings(prompts, encoder):
    embeddings_file = "/Users/tim/Desktop/Code/llm-prompt/data/OpenAssistant_oasst1/embeddings_e5.pkl"
    
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

def train_xgboost_model():
    print("📊 Chargement des données...")
    prompts, scores = load_data_from_final_data()
    print(f"   {len(prompts)} échantillons chargés")
    print(f"   Scores entre {scores.min():.3f} et {scores.max():.3f}")
    
    print("\n🔧 Chargement du modèle E5-small-v2...")
    encoder = SentenceTransformer("intfloat/e5-small-v2")
    
    X = get_or_create_embeddings(prompts, encoder)
    print(f"   Embeddings shape: {X.shape}")
    
    print("\n🔄 Division train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, scores, test_size=0.2, random_state=42
    )
    print(f"   Train: {X_train.shape[0]} échantillons")
    print(f"   Test: {X_test.shape[0]} échantillons")
    
    print("\n🚀 Entraînement XGBoost...")
    n_estimators = 300
    model = XGBRegressor(
        n_estimators=n_estimators,
        max_depth=4,
        learning_rate=0.05,
        reg_lambda=1.0,
        objective="reg:squarederror",
        random_state=42,
        verbosity=0
    )
    
    pbar = tqdm(total=n_estimators, desc="Training XGBoost", unit="tree")

    class TQDMCallback(callback.TrainingCallback):
        def after_iteration(self, model, epoch, evals_log):
            pbar.update(1)
            return False  # continue training

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[TQDMCallback()],
    )

    pbar.close()
    
    print("\n📈 Évaluation du modèle...")
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"   MAE: {mae:.4f}")
    print(f"   MSE: {mse:.4f}")
    print(f"   RMSE: {rmse:.4f}")
    print(f"   R²: {r2:.4f}")
    
    print("\n💾 Sauvegarde du modèle...")
    os.makedirs("models", exist_ok=True)
    model.save_model("models/xgboost_complexity_model.json")
    print("   Modèle sauvegardé dans models/xgboost_complexity_model.json")
    
    print("\n📊 Création des graphiques...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    ax1.scatter(y_test, y_pred, alpha=0.6, s=20)
    ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
              color='red', linestyle='--', linewidth=2)
    ax1.set_xlabel("Score réel")
    ax1.set_ylabel("Score prédit")
    ax1.set_title("Prédictions vs Réalité")
    ax1.grid(True, alpha=0.3)
    
    errors = y_test - y_pred
    ax2.hist(errors, bins=30, alpha=0.7, edgecolor='black')
    ax2.set_xlabel("Erreur (réel - prédit)")
    ax2.set_ylabel("Fréquence")
    ax2.set_title("Distribution des erreurs")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("models/xgboost_results.png", dpi=300, bbox_inches='tight')
    print("   Graphiques sauvegardés dans models/xgboost_results.png")
    
    return model, (mae, mse, rmse, r2)

if __name__ == "__main__":
    model, metrics = train_xgboost_model()
    print(f"\n🎉 Entraînement terminé !")
    print(f"   MAE: {metrics[0]:.4f}")
    print(f"   R²: {metrics[3]:.4f}")