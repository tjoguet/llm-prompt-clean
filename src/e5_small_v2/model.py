from sentence_transformers import SentenceTransformer

# Télécharger et charger le modèle
model = SentenceTransformer("intfloat/e5-small-v2")

# ⚠️ Important : pour e5, il faut ajouter un prompt prefix
prompt = "Salut, comment vas-tu aujourd'hui ?"
embedding = model.encode("query: " + prompt, normalize_embeddings=True)

print(f"Embedding shape: {embedding.shape}")  # (384,)
print(f"Type: {type(embedding)}") 