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
    if scores:
        return round(sum(scores) / len(scores), 2)
    match = re.search(r'(?i)\bscore\s*:\s*\*{0,2}\s*([1-5])\b', entry)
    if match:
        return float(match.group(1))
    return None

def compute_averages(result_json_dict):
    result = {}
    for key, value in result_json_dict.items():
        avg = extract_scores_and_average(value)
        if avg is not None:
            result[key] = avg
    return result

def main():
    parser = argparse.ArgumentParser(description="Calculate the average score for each key and save it as a new JSON file")
    parser.add_argument('--result_json', type=str, required=True, help='Path of result_json json')
    parser.add_argument('--average_score_json', type=str, required=True, help='Path of average_score_json json')

    args = parser.parse_args()

    with open(args.result_json, 'r', encoding='utf-8') as f:
        data = json.load(f)

    averaged_data = compute_averages(data)

    with open(args.average_score_json, 'w', encoding='utf-8') as f:
        json.dump(averaged_data, f, indent=2)


if __name__ == '__main__':
    main()
