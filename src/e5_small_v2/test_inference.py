import time
import numpy as np
from sentence_transformers import SentenceTransformer
import torch

def test_model_performance():
    """Teste les performances du modèle E5-small-v2."""
    
    print("🚀 Chargement du modèle E5-small-v2...")
    start_time = time.time()
    
    # Charger le modèle (depuis le cache ou local)
    model = SentenceTransformer("intfloat/e5-small-v2")
    
    load_time = time.time() - start_time
    print(f"✅ Modèle chargé en {load_time:.2f} secondes")
    
    # Test prompts
    test_prompts = [
        "Salut, comment vas-tu ?",
        "Explique-moi la théorie de la relativité",
        "Quel est le temps qu'il fait aujourd'hui ?",
        "Peux-tu analyser les facteurs économiques qui influencent le marché ?",
        "Bonjour, j'ai une question simple",
        "Pouvez-vous fournir une analyse détaillée des implications sociopolitiques ?",
        "Quelle heure est-il ?",
        "Expliquez les mécanismes moléculaires complexes",
        "Hello world",
        "Analysez les corrélations entre les variables climatiques et économiques"
    ]
    
    print(f"\n📊 Test d'inférence avec {len(test_prompts)} prompts...")
    
    # Test 1: Inférence simple
    print("\n🔍 Test 1: Inférence simple")
    start_time = time.time()
    
    for i, prompt in enumerate(test_prompts):
        embedding = model.encode("query: " + prompt, normalize_embeddings=True)
        if i == 0:
            print(f"   Premier embedding: shape={embedding.shape}, type={type(embedding)}")
    
    simple_time = time.time() - start_time
    print(f"   ⏱️  Temps total: {simple_time:.2f}s")
    print(f"   ⏱️  Temps moyen par prompt: {simple_time/len(test_prompts):.3f}s")
    
    # Test 2: Batch processing
    print("\n🔍 Test 2: Traitement par batch")
    start_time = time.time()
    
    # Préparer les prompts avec le prefix
    batch_prompts = ["query: " + prompt for prompt in test_prompts]
    embeddings = model.encode(batch_prompts, normalize_embeddings=True)
    
    batch_time = time.time() - start_time
    print(f"   ⏱️  Temps total: {batch_time:.2f}s")
    print(f"   ⏱️  Temps moyen par prompt: {batch_time/len(test_prompts):.3f}s")
    print(f"   📈 Gain de performance: {simple_time/batch_time:.1f}x plus rapide")
    
    # Test 3: Multiple runs pour moyenne
    print("\n🔍 Test 3: Moyenne sur 10 runs")
    times = []
    
    for run in range(10):
        start_time = time.time()
        embeddings = model.encode(batch_prompts, normalize_embeddings=True)
        run_time = time.time() - start_time
        times.append(run_time)
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    print(f"   ⏱️  Temps moyen: {avg_time:.3f}s ± {std_time:.3f}s")
    print(f"   ⏱️  Temps moyen par prompt: {avg_time/len(test_prompts):.3f}s")
    
    # Test 4: Mémoire utilisée
    print("\n🔍 Test 4: Utilisation mémoire")
    if torch.cuda.is_available():
        print(f"   🖥️  GPU disponible: {torch.cuda.get_device_name()}")
        print(f"   💾 Mémoire GPU: {torch.cuda.memory_allocated()/1024**2:.1f}MB")
    else:
        print(f"   🖥️  CPU uniquement")
    
    # Résumé
    print(f"\n📋 Résumé des performances:")
    print(f"   • Modèle: E5-small-v2")
    print(f"   • Dimensions: {embeddings.shape[1]}")
    print(f"   • Temps moyen par prompt: {avg_time/len(test_prompts):.3f}s")
    print(f"   • Throughput: {len(test_prompts)/avg_time:.1f} prompts/seconde")

if __name__ == "__main__":
    test_model_performance()
