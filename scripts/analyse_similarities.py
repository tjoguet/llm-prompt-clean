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


def train_xgboost_model():
    
    # Faire un graphique de la répartition des scores
    prompts, scores = load_data_from_final_data()
    plt.hist(scores, bins=30, alpha=0.7, edgecolor='black')
    plt.xlabel("Score")
    plt.ylabel("Fréquence")
    plt.title("Répartition des scores")
    plt.grid(True, alpha=0.3)
    plt.savefig("models/scores_distribution.png", dpi=300, bbox_inches='tight')
    

if __name__ == "__main__":
    train_xgboost_model()
