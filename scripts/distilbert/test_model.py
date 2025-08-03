import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.distilbert.complexity_scorer import ComplexityScorer

def main():
    scorer = ComplexityScorer(model_path="models/checkpoints/best_model.pt")
    prompt = "What are the main causes of climate change?"
    score, confidence = scorer.score(prompt)
    
    print(f"Requête: {prompt}")
    print(f"➡️ Score de complexité: {score:.4f}")
    print(f"➡️ Confiance: {confidence:.4f}")

if __name__ == "__main__":
    main()