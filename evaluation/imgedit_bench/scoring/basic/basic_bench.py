import base64
import os
import json
import argparse
import re
from openai import OpenAI
from tqdm import tqdm
from tenacity import retry, wait_exponential, stop_after_attempt
from concurrent.futures import ThreadPoolExecutor, as_completed

def load_prompts(prompts_json_path):
    with open(prompts_json_path, 'r') as f:
        return json.load(f)

def image_to_base64(image_path):
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        print(f"File {image_path} not found.")
        return None

# @retry(wait=wait_exponential(multiplier=1, min=2, max=2), stop=stop_after_attempt(100))
def call_gpt(original_image_path, result_image_path, edit_prompt, edit_type, prompts):
    try:
        original_image_base64 = image_to_base64(original_image_path)
        result_image_base64 = image_to_base64(result_image_path)

        if not original_image_base64 or not result_image_base64:
            return {"error": "Image conversion failed"}

        client = OpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
            base_url=os.environ.get("OPENAI_BASE_URL", "https://api.nuwaapi.com/v1"),
        )

        prompt = prompts[edit_type]
        full_prompt = prompt.replace('<edit_prompt>', edit_prompt)

        response = client.chat.completions.create(
            model=os.environ.get("OPENAI_SCORING_MODEL", "gpt-4o"),
            stream=False,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": full_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{original_image_base64}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{result_image_base64}"}}
                ]
            }]
        )

        return response
    except Exception as e:
        print(f"Error in calling GPT API: {e}")
        raise

def extract_scores_and_average(entry) -> float:
    if not isinstance(entry, str):
        return None
    scores = []
    for line in entry.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        clean = stripped.replace('*', '')
        if re.match(r'(?i)^\s*brief reasoning\b', clean):
            continue
        match = re.search(r':\s*([1-5])\b', clean)
        if match:
            scores.append(int(match.group(1)))
    if scores:
        return round(sum(scores) / len(scores), 2)
    match = re.search(r'(?i)\bscore\s*:\s*\*{0,2}\s*([1-5])\b', entry)
    if match:
        return float(match.group(1))
    return None


def is_valid_result(value) -> bool:
    return extract_scores_and_average(value) is not None


def process_single_item(key, item, result_img_folder, origin_img_root, prompts):
    result_img_path = os.path.join(result_img_folder, f"{key}.png")
    origin_img_path = os.path.join(origin_img_root, item["id"])
    edit_prompt = item["prompt"]
    edit_type = item["edit_type"]

    missing = []
    if not os.path.isfile(origin_img_path):
        missing.append(f"origin image missing: {origin_img_path}")
    if not os.path.isfile(result_img_path):
        missing.append(f"result image missing: {result_img_path}")
    if missing:
        return key, {"error": "; ".join(missing)}

    try:
        response = call_gpt(
            origin_img_path, result_img_path, edit_prompt, edit_type, prompts
        )
    except Exception as e:
        return key, {"error": str(e)}

    if isinstance(response, dict) and "error" in response:
        return key, response
    return key, response.choices[0].message.content


def process_json(
    edit_json,
    result_img_folder,
    origin_img_root,
    num_threads,
    prompts,
    output_json=None,
    skip_existing=False,
):
    with open(edit_json, "r") as f:
        edit_infos = json.load(f)

    results_path = output_json or os.path.join(result_img_folder, "result.json")
    results = {}
    if skip_existing and os.path.isfile(results_path):
        with open(results_path, "r") as f:
            results = json.load(f)

    pending = {}
    for key, item in edit_infos.items():
        if skip_existing and is_valid_result(results.get(key)):
            continue
        pending[key] = item

    skipped = len(edit_infos) - len(pending)
    if skipped:
        print(f"Skipping {skipped} already-scored items")

    if not pending:
        print("No pending Basic items to score")
    else:
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            future_to_key = {
                executor.submit(
                    process_single_item,
                    key,
                    item,
                    result_img_folder,
                    origin_img_root,
                    prompts,
                ): key
                for key, item in pending.items()
            }

            for future in tqdm(
                as_completed(future_to_key),
                total=len(future_to_key),
                desc="Processing edits",
            ):
                key = future_to_key[future]
                try:
                    k, result = future.result()
                    results[k] = result
                    if isinstance(result, dict) and "error" in result:
                        print(f"Skipped key {key}: {result['error']}")
                except Exception as e:
                    print(f"Error processing key {key}: {e}")
                    results[key] = {"error": str(e)}

    os.makedirs(os.path.dirname(results_path) or ".", exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)

def main():
    parser = argparse.ArgumentParser(description="Evaluate image edits using GPT")
    parser.add_argument('--result_img_folder', type=str, required=True, help="Folder with subfolders of edited images")
    parser.add_argument('--edit_json', '--basic_edit', type=str, required=True, dest='edit_json',
                        help="Path to JSON file mapping keys to metadata")
    parser.add_argument('--output_json', type=str, default=None, help="Where to save GPT raw results")
    parser.add_argument('--origin_img_root', type=str, required=True, help="Root path where original images are stored")
    parser.add_argument('--num_processes', type=int, default=32, help="Number of parallel threads")
    parser.add_argument('--prompts_json', type=str, required=True, help="JSON file containing prompts")
    parser.add_argument(
        '--skip_existing',
        action='store_true',
        help='Skip keys that already have valid GPT text in output_json',
    )
    args = parser.parse_args()

    prompts = load_prompts(args.prompts_json)

    process_json(
        args.edit_json,
        args.result_img_folder,
        args.origin_img_root,
        args.num_processes,
        prompts,
        output_json=args.output_json,
        skip_existing=args.skip_existing,
    )

if __name__ == "__main__":
    main()
