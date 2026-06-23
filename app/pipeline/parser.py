import json
from pathlib import Path
from typing import Any, Dict, List, Union

import pandas as pd


def _load_json(file_path: Union[str, Path]) -> Dict[str, Any]:
    path = Path(file_path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _extract_rounds(rounds_data: List[Dict[str, Any]]) -> pd.DataFrame:
    """Extracts rounds data JSON to a DataFrame"""
    records = []

    for round_idx, rnd in enumerate(rounds_data):
        ctx = rnd.get("environment_context") or {}
        market = ctx.get("market_snapshot") or {}

        records.append(
            {
                "round_index": round_idx,
                "hour": rnd.get("hour"),
                "event_narrative": ctx.get("event_narrative"),
                "event_headline": ctx.get("event_headline"),
                "stock_price_raw": market.get("stock_price"),
                "percent_change_raw": market.get("percent_change"),
                "sentiment": market.get("sentiment"),
                "trending_hashtags": market.get("trending_hashtags", []),
                "media_events": ctx.get("media_events", []),
                "social_state": ctx.get("social_state"),
                "external_actor_actions": ctx.get("external_actor_actions", []),
                "social_manager_alerts": ctx.get("social_manager_alerts", []),
                "agents_unavailable": ctx.get("agents_unavailable", []),
                "critical_deadlines": ctx.get("critical_deadlines", []),
                "news": ctx.get("news", []),
            }
        )

    return pd.DataFrame(records)


def _extract_participants(rounds_data: List[Dict[str, Any]]) -> pd.DataFrame:
    """Extracts participants data JSON to a DataFrame"""
    records = []

    for round_idx, rnd in enumerate(rounds_data):
        for part in rnd.get("participants", []):
            meta = part.get("agent_round_metadata") or {}

            records.append(
                {
                    "round_index": round_idx,
                    "agent_id": part.get("agent_id"),
                    "agent_role": part.get("agent_role"),
                    "agent_label": part.get("agent_label"),
                    "declared_action": part.get("declared_action"),
                    "sentiment_at_turn": meta.get("sentiment_at_turn"),
                    "action_classification": meta.get("action_classification"),
                }
            )

    return pd.DataFrame(records)


def _extract_comms(rounds_data: List[Dict[str, Any]]) -> pd.DataFrame:
    """Extracts communications data JSON to a DataFrame"""
    records = []

    for round_idx, rnd in enumerate(rounds_data):
        for comm in rnd.get("communications", []):
            istate = comm.get("internal_state") or {}

            records.append(
                {
                    "round_index": round_idx,
                    "message_id": comm.get("message_id"),
                    "agent_id": comm.get("agent_id"),
                    "agent_role": comm.get("agent_role"),
                    "agent_label": comm.get("agent_label"),
                    "channel": comm.get("channel"),
                    "recipients": comm.get("recipients", []),
                    "message_type": comm.get("message_type"),
                    "responding_to": comm.get("responding_to"),
                    "content": comm.get("content"),
                    "timestamp": comm.get("timestamp"),
                    "reacting": istate.get("reacting", None),
                    "rationalizing": istate.get("rationalizing", None),
                    "deliberating": istate.get("deliberating", None),
                }
            )

    return pd.DataFrame(records)


def load_comms_dataframe(file_path: Union[str, Path]) -> pd.DataFrame:
    """External API function for loading communications"""
    raw_data = _load_json(file_path)
    return _extract_comms(raw_data.get("rounds", []))


def load_rounds_dataframe(file_path: Union[str, Path]) -> pd.DataFrame:
    """External API function for loading rounds"""
    raw_data = _load_json(file_path)
    return _extract_rounds(raw_data.get("rounds", []))


def load_participants_dataframe(file_path: Union[str, Path]) -> pd.DataFrame:
    """External API function for loading participants"""
    raw_data = _load_json(file_path)
    return _extract_participants(raw_data.get("rounds", []))
