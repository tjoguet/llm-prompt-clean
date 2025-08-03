from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn

class ComplexityScorer(nn.Module):
    def __init__(self, model_name="distilbert-base-uncased", model_path=None, device='cpu'):
        super(ComplexityScorer, self).__init__()
        self.device = device
        self.distilbert = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.complexity_head = nn.Sequential(
            nn.Linear(self.distilbert.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)  # Une seule sortie au lieu de 2
        )

        self.to(self.device)

        # Charger les poids si fournis
        if model_path:
            self.load_model(model_path)
        
    def forward(self, input_ids, attention_mask=None):
        # Passer par DistilBERT
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Extraire l'embedding du token [CLS]
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # [batch_size, 768]
        
        # Passer par notre tête personnalisée
        raw_output = self.complexity_head(cls_embedding)  # [batch_size, 1]
        
        # Appliquer tanh pour avoir des valeurs entre -1 et 1
        complexity_score = torch.tanh(raw_output).squeeze(-1)  # Score entre -1 et 1
        

        return complexity_score
    
    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            self.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.load_state_dict(checkpoint)
        self.eval()
    
    def score(self, prompt):
        tokens = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        tokens = {k: v.to(self.device) for k, v in tokens.items()}
        with torch.no_grad():
            score = self(**tokens)
        return score.item()

# Initialisation
# tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
# model = ComplexityScorer()

# # User input prompt
# input_text = "What is the capital of France?"

# # Tokenize input
# inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)

# # Forward pass
# with torch.no_grad():
#     complexity_score = model(
#         input_ids=inputs['input_ids'],
#         attention_mask=inputs['attention_mask']
#     )

# print(f"Input: {input_text}")
# print(f"Complexity Score: {complexity_score.item():.4f}")

# # Test avec d'autres exemples
# test_examples = [
#     "Hello!",
#     "Explain quantum mechanics and its applications in modern computing",
#     "What time is it?",
#     "Can you provide a detailed analysis of the socioeconomic factors influencing climate change policies?"
# ]

# print("\n" + "="*60)
# print("Tests avec différentes requêtes:")
# print("="*60)

# for text in test_examples:
#     inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
#     with torch.no_grad():
#         score = model(
#             input_ids=inputs['input_ids'],
#             attention_mask=inputs['attention_mask']
#         )
#     print(f"Requête: {text}")
#     print(f"Score: {score.item():.4f}")
#     print("-" * 40)