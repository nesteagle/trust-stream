import json
from pathlib import Path


def load_and_parse_eval(input_path: str) -> dict:
    with open(input_path, "r", encoding="utf-8") as file:
        results = json.load(file)

    explanation_lookup = {}

    for r in results:
        msg_id = r.get("custom_id")
        if not msg_id:
            continue

        message_metrics = {}

        for metric, score in r["final_scores"].items():
            audit_key = f"{metric}_analysis"
            explanation_text = r["reasoning_audit"].get(audit_key)
            message_metrics[metric] = (explanation_text, score)

        explanation_lookup[msg_id] = message_metrics
    return explanation_lookup


def save_json(data: list, path: str) -> None:
    Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"Saved {len(data)} objects -> {path}")
