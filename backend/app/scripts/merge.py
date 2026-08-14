import json
import statistics


def merge_by_median(file_paths, output_path):
    data_by_id = {}
    for path in file_paths:
        with open(path, "r", encoding="utf-8") as f:
            for obj in json.load(f):
                data_by_id.setdefault(obj["custom_id"], []).append(obj)

    final_output = []

    for custom_id, records in data_by_id.items():
        score_keys = records[0]["final_scores"].keys()

        merged_scores = {}
        merged_audit = {}

        for key in score_keys:
            audit_key = f"{key}_analysis"
            pairs = [
                (r["final_scores"][key], r["reasoning_audit"][audit_key])
                for r in records
            ]
            median_score = statistics.median([score for score, _ in pairs])

            best_pair = min(pairs, key=lambda p: abs(p[0] - median_score))

            merged_scores[key] = best_pair[0]
            merged_audit[audit_key] = best_pair[1]

        final_output.append(
            {
                "custom_id": custom_id,
                "reasoning_audit": merged_audit,
                "final_scores": merged_scores,
            }
        )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=4, ensure_ascii=False)


def main():
    # Include your JSON output paths from eval.py: we use n=3 (1,2,3)
    paths = [
        f"output/latest/eval_results_luna{iteration}.json" for iteration in range(1, 4)
    ]

    merge_by_median(
        file_paths=paths, output_path="output/latest/eval_results_luna_median.json"
    )


if __name__ == "__main__":
    main()
