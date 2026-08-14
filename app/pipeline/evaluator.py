import pandas as pd

from app.evaluator.io import load_and_parse_eval, load_scores
from app.evaluator.scoring import (
    get_explanation_scores,
    get_external_score,
    get_internal_score,
)


def _get_eval_ids(message_id: str) -> tuple[str, str]:
    """IDs of content and reasoning formatted for LLM eval lookup"""
    return [message_id + "-content", message_id + "-reasoning"]


def score_dataframe(df: pd.DataFrame, eval_path: str) -> pd.DataFrame:
    """Add alignment score explanations to dataframe"""
    explanations = load_and_parse_eval(eval_path)

    df["score_explanations_external"] = df["message_id"].apply(
        lambda m_id: get_explanation_scores(explanations, _get_eval_ids(m_id)[0])
    )
    df["score_explanations_internal"] = df["message_id"].apply(
        lambda m_id: get_explanation_scores(explanations, _get_eval_ids(m_id)[1])
    )
    return df
