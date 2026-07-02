from typing import Any

import pandas as pd

from app.config.anchors import HIGH_ANCHORS, LOW_ANCHORS, METRIC_FOCUS, NEUTRAL_ANCHORS
from app.config.settings import CACHE_FILE_PATH, SOFTMIN_TEMP
from app.io.cache import load_embedding_cache, save_embedding_cache
from app.math.core import (
    calculate_centroid,
    debias_axis,
    project_scalar,
    softmin_weighted_average,
)
from app.models.embeddings import AlignmentEmbedder
from app.nlp.chunker import chunk_into_sentences

EWMA_SPAN = 30


def get_instruction() -> str:
    return f"Instruct: Embed text for evaluating {METRIC_FOCUS}.\nQuery: "


def _extract_clean_internal_text(row: Any) -> str:
    """ERxtracts, filters, and builds string from an agent's internal state fields."""
    if row:
        state = row.get("reasoning")
    clean_sentences = []

    if pd.isna(state) or state is None:
        return
    s_str = str(state).strip()
    if s_str and s_str.lower() not in ("nan", "none", "null", ""):
        if not s_str.endswith("."):
            s_str += "."
        clean_sentences.append(s_str)

    return " ".join(clean_sentences)


def ensure_embeddings(df: pd.DataFrame) -> dict:
    """Runs the cache check and inference if needed, returning the fully loaded cache."""
    instruction = get_instruction()
    req_strings = [
        f"{instruction}{t}" for t in LOW_ANCHORS + HIGH_ANCHORS + NEUTRAL_ANCHORS
    ]
    req_strings.extend([f"{instruction}{t}" for t in df["content"] if pd.notna(t)])

    for row in df.to_dict("records"):
        combined_internal = _extract_clean_internal_text(row)
        if combined_internal:
            for sentence in chunk_into_sentences(combined_internal):
                req_strings.append(f"{instruction}{sentence}")

    req_strings = list(set(req_strings))
    cache = load_embedding_cache(CACHE_FILE_PATH)
    missing = [s for s in req_strings if s not in cache]

    if missing:
        print(f"Inference Required: Encoding {len(missing)} new items...")
        embedder = AlignmentEmbedder()
        vectors = embedder.encode(missing)
        for text, vector in zip(missing, vectors):
            cache[text] = vector
        save_embedding_cache(cache, CACHE_FILE_PATH)
        print("Cache saved.")

    return cache


def score_dataframe(df: pd.DataFrame, cache: dict) -> pd.DataFrame:
    """Calculates alignment scores and generates per-agent moving average scores."""
    instruction = get_instruction()

    if "timestamp" in df.columns:
        df = df.sort_values("timestamp").reset_index(drop=True)

    v_low = calculate_centroid([cache[f"{instruction}{a}"] for a in LOW_ANCHORS])
    v_high = calculate_centroid([cache[f"{instruction}{a}"] for a in HIGH_ANCHORS])
    v_neutral = calculate_centroid(
        [cache[f"{instruction}{a}"] for a in NEUTRAL_ANCHORS]
    )
    axis_debiased = debias_axis(v_low, v_high, v_neutral)

    def get_ext_score(text: str) -> float:
        if pd.isna(text):
            return 0.0
        emb = cache.get(f"{instruction}{text}")
        return (
            project_scalar(emb, v_low, axis_debiased) - 0.5 if emb is not None else 0.0
        )

    def get_int_score(combined_text: str) -> float:
        if not combined_text:
            return float("nan")
        sentences = chunk_into_sentences(combined_text)
        scores = []
        for s in sentences:
            emb = cache.get(f"{instruction}{s}")
            if emb is not None:
                scores.append(project_scalar(emb, v_low, axis_debiased) - 0.5)
        return (
            softmin_weighted_average(scores, temperature=SOFTMIN_TEMP)
            if scores
            else float("nan")
        )

    df["score_external"] = df["content"].apply(get_ext_score)

    df["score_internal"] = [
        get_int_score(_extract_clean_internal_text(row))
        for row in df.to_dict("records")
    ]

    valid_ext_mask = df["content"].fillna("").str.split().str.len() > 2
    ext_scores_masked = df["score_external"].where(valid_ext_mask)

    df["external_avg"] = (
        df.assign(temp_ext=ext_scores_masked)
        .groupby("agent_id")["temp_ext"]
        .transform(lambda group: group.ewm(span=EWMA_SPAN, adjust=False).mean())
    )
    df["external_avg"] = df.groupby("agent_id")["external_avg"].ffill()

    df["internal_avg"] = df.groupby("agent_id")["score_internal"].transform(
        lambda group: group.ewm(span=EWMA_SPAN, adjust=False).mean()
    )
    df["internal_avg"] = df.groupby("agent_id")["internal_avg"].ffill()

    df["deception_delta"] = (df["score_external"] - df["score_internal"]).fillna(0.0)

    return df
