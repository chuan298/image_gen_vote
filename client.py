import httpx
import asyncio
import base64
import os
from PIL import Image
import io
import time
import re
import json
import pandas as pd
from typing import List, Optional, Tuple

# --- Configuration ---
SERVER_URLS: List[str] = [
    "http://localhost:8000/generate",
    # "http://localhost:8001/generate",
    # "http://localhost:8002/generate",
]
REQUEST_TIMEOUT: float = 700.0
DEFAULT_OUTPUT_BASE_DIR: str = "generated_images"
PROMPT_CSV_FILE: str = "prompts.csv" # CSV with 'prompt' column

NUM_IMAGES_TO_GENERATE: int = 5
TOP_K_TO_SELECT: int = 2
# --- End Configuration ---

output_base_dir_actual = DEFAULT_OUTPUT_BASE_DIR # Will be updated by model type

def sanitize_filename(name: str) -> str:
    name = re.sub(r'[<>:"/\\|?*]', '_', name)
    name = re.sub(r'\s+', '_', name)
    return name[:100]

async def send_request_and_save(
    client: httpx.AsyncClient,
    worker_url: str,
    original_prompt: str,
    num_images: int,
    top_k: int
) -> str:
    global output_base_dir_actual
    payload = {"prompt": original_prompt, "num_images": num_images, "top_k": top_k}
    log_prompt = original_prompt[:50] + "..." if len(original_prompt) > 50 else original_prompt
    
    try:
        # print(f"Sending to {worker_url}: {log_prompt}") # Verbose
        response = await client.post(worker_url, json=payload, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        data = response.json()

        model_type = data.get("model_type", "UNKNOWN_MODEL").upper()
        used_seed = data.get("used_seed", -1)

        if output_base_dir_actual == DEFAULT_OUTPUT_BASE_DIR and model_type != "UNKNOWN_MODEL":
            output_base_dir_actual = f"{model_type.lower()}_response"
            os.makedirs(output_base_dir_actual, exist_ok=True)
            print(f"Output directory determined: {output_base_dir_actual}")
        elif output_base_dir_actual == DEFAULT_OUTPUT_BASE_DIR: # Ensure default exists if not updated
             os.makedirs(DEFAULT_OUTPUT_BASE_DIR, exist_ok=True)


        folder_name = f"{sanitize_filename(original_prompt)}_s{used_seed}_n{num_images}_k{top_k}"
        output_folder_path = os.path.join(output_base_dir_actual, folder_name)
        os.makedirs(output_folder_path, exist_ok=True)

        prompt_details = {
            "original_prompt": original_prompt, "refined_prompt": data["refined_prompt"],
            "model_type": model_type, "num_images_requested": num_images,
            "top_k_selected": top_k, "client_provided_seed": None,
            "server_used_seed": used_seed, "num_images_generated": len(data["images"]),
            "processed_by_worker": worker_url
        }
        with open(os.path.join(output_folder_path, "prompt_details.json"), "w", encoding="utf-8") as f:
            json.dump(prompt_details, f, indent=4)

        for i, img_data in enumerate(data["images"]):
            img_bytes = base64.b64decode(img_data["base64_image"])
            Image.open(io.BytesIO(img_bytes)).save(os.path.join(output_folder_path, f"image_{i+1}.png"))
        
        return f"OK: {log_prompt} (Worker: {worker_url.split('/')[-2]})" # Show port or part of URL
    
    except httpx.HTTPStatusError as e:
        status = e.response.status_code
        err_detail = e.response.text[:100] # Truncate long error messages
        msg = f"HTTP_ERR {status}: {log_prompt} ({err_detail})"
        print(msg)
        return msg
    except Exception as e:
        msg = f"FAIL: {log_prompt} ({type(e).__name__})"
        print(msg)
        return msg


def load_prompts_from_csv_pandas(filepath: str) -> List[str]:
    try:
        df = pd.read_csv(filepath, encoding='utf-8-sig')
        if 'prompt' not in df.columns:
            print(f"Error: CSV '{filepath}' needs 'prompt' column."); return []
        prompts = df['prompt'].dropna().astype(str).str.strip().tolist()
        return [p for p in prompts if p]
    except FileNotFoundError: print(f"Error: File not found '{filepath}'"); return []
    except Exception as e: print(f"Error reading CSV '{filepath}': {e}"); return []

async def main():
    global output_base_dir_actual
    raw_prompts = load_prompts_from_csv_pandas(PROMPT_CSV_FILE)
    if not raw_prompts: print(f"No prompts from '{PROMPT_CSV_FILE}'. Exiting."); return
    if not SERVER_URLS: print("Error: SERVER_URLS is empty."); return

    # Ensure the initial default directory exists if no model-specific one is made
    if not os.path.exists(DEFAULT_OUTPUT_BASE_DIR) and output_base_dir_actual == DEFAULT_OUTPUT_BASE_DIR:
        os.makedirs(DEFAULT_OUTPUT_BASE_DIR, exist_ok=True)

    num_workers = len(SERVER_URLS)
    print(f"Client starting. {num_workers} workers. {len(raw_prompts)} prompts.")
    print(f"Hardcoded: num_images={NUM_IMAGES_TO_GENERATE}, top_k={TOP_K_TO_SELECT}")

    all_results = []
    
    async with httpx.AsyncClient() as client:
        for i in range(0, len(raw_prompts), num_workers):
            batch_prompts = raw_prompts[i:i + num_workers]
            tasks = []
            
            print(f"\nProcessing batch {i//num_workers + 1}/{ (len(raw_prompts) + num_workers -1) // num_workers }...")

            for j, prompt_text in enumerate(batch_prompts):
                worker_url_index = j % num_workers # Assigns prompts to workers in order for this batch
                worker_url = SERVER_URLS[worker_url_index]
                tasks.append(
                    send_request_and_save(
                        client,
                        worker_url,
                        prompt_text,
                        NUM_IMAGES_TO_GENERATE,
                        TOP_K_TO_SELECT
                    )
                )
            
            batch_results = await asyncio.gather(*tasks, return_exceptions=False)
            all_results.extend(batch_results)
            for res in batch_results: print(f"  {res}") # Print result of each item in batch

    print("\n--- All batches processed ---")
    success_count = sum(1 for res in all_results if res and res.startswith("OK:"))
    print(f"Total successful: {success_count} / {len(all_results)}")
    print(f"Output saved in subdirectories under: '{output_base_dir_actual}'")


if __name__ == "__main__":
    start_time = time.time()
    asyncio.run(main())
    end_time = time.time()
    print(f"Client finished in {end_time - start_time:.2f} seconds.")