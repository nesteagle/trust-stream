from dotenv import load_dotenv
from eval_judge import (
    build_requests_batch,
    load_and_parse_results,
    retrieve_batch,
    submit_batch,
)

from app.evaluator.config import CONFIG
from app.evaluator.io import save_json
from app.evaluator.preprocessing import build_request_contents
from app.pipeline.parser import load_comms_dataframe, load_rounds_dataframe
from app.pipeline.preprocessing import resolve_and_tidy


def main():
    load_dotenv()

    comms_df = load_comms_dataframe("data/raw/data.json")
    rounds_df = load_rounds_dataframe("data/raw/data.json")
    df_messages, _ = resolve_and_tidy(comms_df)

    requests = build_request_contents(df_messages=df_messages, df_rounds=rounds_df)

    jsonl_batch_path = "output/latest/eval_batch.jsonl"
    build_requests_batch(
        messages=requests,
        output_path=jsonl_batch_path,
        config=CONFIG,
    )

    batch_id = submit_batch(jsonl_path=jsonl_batch_path)

    batch_output_path = "output/latest/batch_output.jsonl"

    retrieve_batch(batch_id=batch_id, output_path=batch_output_path)

    results = load_and_parse_results(input_path=batch_output_path)

    save_json(results, "output/latest/eval_results.json")


if __name__ == "__main__":
    main()
