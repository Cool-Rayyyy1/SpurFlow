import json
import argparse
import re

def extract_scores_and_average(entry) -> float:
    if not isinstance(entry, str):
        return None
    lines = entry.splitlines()
    scores = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        clean = stripped.replace('*', '')
        if re.match(r'(?i)^\s*brief reasoning\b', clean):
            continue
        match = re.search(r':\s*([1-5])\b', clean)
        if match:
            scores.append(int(match.group(1)))
            continue
        match = re.match(r'^\s*([1-5])\s*(?:\(|:|\b)', clean)
        if match:
            scores.append(int(match.group(1)))
    if scores:
        return round(sum(scores) / len(scores), 2)
    match = re.search(r'(?i)\bscore\s*:\s*\*{0,2}\s*([1-5])\b', entry)
    if match:
        return float(match.group(1))
    return None

def compute_final_average(result_json_dict):
    total_scores = []
    
    for value in result_json_dict.values():
        avg = extract_scores_and_average(value)
        if avg is not None:
            total_scores.append(avg)
    
    if total_scores:
        return round(sum(total_scores) / len(total_scores), 2)
    return None

def main():
    parser = argparse.ArgumentParser(description="Calculate the average score for all keys and print the final average")
    parser.add_argument('--result_json', type=str, required=True, help='Path of result json')
    parser.add_argument('--average_score_json', type=str, default=None,
                        help='Optional path to save per-key averages plus final average')

    args = parser.parse_args()

    with open(args.result_json, 'r', encoding='utf-8') as f:
        data = json.load(f)

    final_average = compute_final_average(data)

    if args.average_score_json:
        out = {}
        for key, value in data.items():
            avg = extract_scores_and_average(value)
            if avg is not None:
                out[key] = avg
        out["__final_average__"] = final_average
        with open(args.average_score_json, 'w', encoding='utf-8') as f:
            json.dump(out, f, indent=2)

    if final_average is not None:
        print(final_average)
    else:
        print("No valid scores found.")

if __name__ == '__main__':
    main()
