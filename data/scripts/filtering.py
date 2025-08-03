import json
import re
import html
import unicodedata

def clean_text(text):
    # Décodage des entités Unicode (ex: \u2019 → ’)
    text = text.encode("utf-8", "surrogatepass").decode("unicode_escape", "ignore")

    # Décodage HTML éventuel (ex: &amp; → &)
    text = html.unescape(text)

    # Supprimer les emojis et caractères non imprimables
    text = ''.join(c for c in text if c.isprintable() and not is_emoji(c))

    # Normaliser les apostrophes, guillemets, etc.
    text = unicodedata.normalize("NFKC", text)

    # Supprimer les tabulations et espaces inutiles
    text = text.replace("\t", " ").strip()

    # Réduire les retours à la ligne multiples
    text = re.sub(r"\n{2,}", "\n", text)

    return text

def is_emoji(char):
    return unicodedata.category(char) in {"So", "Cs"}  # Symbol, Other or Surrogate

def count_words(text):
    return len(re.findall(r'\w+', text))

filtered_data = []

with open("/Users/tim/Desktop/Code/llm-prompt/data/OpenAssistant_oasst1/original_data.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        text = item.get("text", "").strip()
        lang = item.get("lang", "unknown")


        if lang == "en":
            text = clean_text(text)
            word_count = count_words(text)
            if 5 < word_count <= 300:
                filtered_data.append({
                    "id": item["message_id"],
                    "prompt": text
                })

# Sauvegarde
with open("filtered_prompts.jsonl", "w", encoding="utf-8") as f_out:
    for item in filtered_data:
        f_out.write(json.dumps(item, ensure_ascii=False) + "\n")
        
        
   