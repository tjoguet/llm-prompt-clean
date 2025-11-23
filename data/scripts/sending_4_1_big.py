import json
from openai import OpenAI
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="api", 
)

output_file = "responses_4_1_big.jsonl"
input_file = "/Users/tim/Desktop/Code/llm-prompt-clean/data/OpenAssistant_oasst1/filtered_prompts.jsonl"
max_lines = 5000  
num_threads = 20  

lock = threading.Lock()
error_count = 0

def process_prompt(item):
    global error_counts
    prompt = item["prompt"]
    start = time.time()
    try:
        response = client.chat.completions.create(
            model="openai/gpt-4.1",
            messages=[{"role": "user", "content": prompt}],
            extra_headers={
                "HTTP-Referer": "https://yourproject.com",
                "X-Title": "MyLLMJudgeApp",
            }
        )
        output = {
            "id": item["id"],
            "prompt": prompt,
            "response": response.choices[0].message.content
        }
        iter_time = time.time() - start
        return (output, None, iter_time)
    except Exception as e:
        with lock:
            global error_count
            error_count += 1
        iter_time = time.time() - start
        return (None, f"[ERROR] {item['id']} — {e}", iter_time)

def main():
    items = []
    with open(input_file, "r", encoding="utf-8") as f_in:
        for i, line in enumerate(f_in):
            if i >= max_lines:
                break
            items.append(json.loads(line))

    total_time = 0
    count = 0
    start_global = time.time()

    with open(output_file, "w", encoding="utf-8") as f_out, ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {executor.submit(process_prompt, item): item for item in items}
        for idx, future in enumerate(as_completed(futures)):
            output, error, iter_time = future.result()
            total_time += iter_time
            count += 1
            avg_progress_time = (time.time() - start_global) / count if count > 0 else 0
            if output:
                with lock:
                    f_out.write(json.dumps(output, ensure_ascii=False) + "\n")
                print(f"[OK] {output['id']} (line {idx+1}/{len(items)}) | avg progress: {avg_progress_time:.2f}s")
            else:
                print(error)

    print(f"Total errors encountered: {error_count}")

if __name__ == "__main__":
    main()