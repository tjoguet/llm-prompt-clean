import json
import re
import html
import unicodedata

def clean_text(text):

    text = text.encode("utf-8", "surrogatepass").decode("unicode_escape", "ignore")
    text = html.unescape(text)
    text = ''.join(c for c in text if c.isprintable() and not is_emoji(c))
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\t", " ").strip()
    text = re.sub(r"\n{2,}", "\n", text)

    return text

def is_emoji(char):
    return unicodedata.category(char) in {"So", "Cs"} 

def count_words(text):
    return len(re.findall(r'\w+', text))

filtered_data = []

with open("/Users/tim/Desktop/Code/llm-prompt-clean/data/OpenAssistant_oasst1/original_data.jsonl", "r", encoding="utf-8") as f:
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
        
        
   