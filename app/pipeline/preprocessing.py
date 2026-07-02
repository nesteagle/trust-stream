from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Dict, Tuple

import pandas as pd

# convert to true role found in data
CORRECTION_ROLE_MAP = {"social_manager": "social_media"}

# message ping origins are searched in this order
CHANNEL_HIERARCHY = [
    "one_on_one_chat",
    "side_huddle",
    "comms_huddle",
    "official_post",
]


@dataclass(frozen=True, slots=True)
class MessageThread:
    id: str
    timestamp: pd.Timestamp


def _normalize_role(role: str) -> str:
    """Normalizes string roles to true roles found in data."""
    return CORRECTION_ROLE_MAP.get(role, role)


def resolve_and_tidy(
    df: pd.DataFrame, max_hour_difference=1
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Corrects dataset and standardizes time to pd.Timestamp; derives edges from direct replies + implicit mentions"""
    df_working = df.copy()
    df_working["timestamp"] = pd.to_datetime(df_working["timestamp"])
    sort_cols = [c for c in ["round_index", "timestamp"] if c in df_working.columns]
    records = (
        df_working.sort_values(by=sort_cols).reset_index(drop=True).to_dict("records")
    )

    channel_to_index = {ch.strip().lower(): i for i, ch in enumerate(CHANNEL_HIERARCHY)}

    thread_registry: Dict[Any, MessageThread] = {}
    role_by_msg_id: dict = {}
    members_by_msg_id: dict = {}

    edges = []

    for record in records:
        sender_role = _normalize_role(str(record["agent_role"]).strip().lower())
        msg_id = record["message_id"]
        resp_to = record["responding_to"]
        channel = str(record["channel"]).strip().lower()
        current_ts = record.get("timestamp")

        # frozenset so we can use as key in registry
        member_set = frozenset([sender_role])

        if pd.notna(resp_to):
            resp_to_str = str(resp_to).strip()

            # handling @role responding_to format in data
            if resp_to_str.startswith("@"):
                mentioned_roles = [
                    _normalize_role(r.lstrip("@").lower()) for r in resp_to_str.split()
                ]
                start_idx = channel_to_index.get(channel)
                search_path = (
                    CHANNEL_HIERARCHY[start_idx:]
                    if start_idx is not None
                    else [channel]
                )

                member_set = frozenset([sender_role] + mentioned_roles)

                prior_msg_id = None
                for candidate in search_path:
                    for target_role in mentioned_roles:
                        entry = thread_registry.get(
                            (candidate, member_set, target_role)
                        )

                        if entry and current_ts - entry.timestamp <= timedelta(
                            hours=max_hour_difference
                        ):
                            prior_msg_id = entry.id
                            break

                if prior_msg_id:
                    edges.append(
                        {
                            "message_id": msg_id,
                            "target_id": prior_msg_id,
                            "interaction_type": "mention_reply",
                        }
                    )
                record["responding_to"] = prior_msg_id

            # regular formatted message_id
            elif resp_to_str:
                edge_label = "direct_reply"
                parts = resp_to_str.split("_")

                # as some IDs in VAST data are hallucinated future times
                if int(parts[1]) > record["round_index"]:
                    parts[1] = str(record["round_index"]).zfill(2)
                    resp_to_str = "_".join(parts)
                    record["responding_to"] = resp_to_str
                    edge_label = "direct_reply_corrected"

                if resp_to_str in members_by_msg_id:
                    member_set = members_by_msg_id[resp_to_str]
                else:
                    parent_role = role_by_msg_id.get(resp_to_str, "all")
                    member_set = frozenset([sender_role, parent_role])

                edges.append(
                    {
                        "message_id": msg_id,
                        "target_id": resp_to_str,
                        "interaction_type": edge_label,
                    }
                )

        role_by_msg_id[msg_id] = sender_role
        members_by_msg_id[msg_id] = member_set

        thread_registry[(channel, member_set, sender_role)] = MessageThread(
            id=msg_id, timestamp=current_ts
        )

    df_messages = pd.DataFrame(records)
    df_messages["reasoning"] = (
        df_messages[["reacting", "rationalizing", "deliberating"]]
        .bfill(axis=1)
        .iloc[:, 0]
    ).drop(columns=["responding_to", "reacting", "rationalizing", "deliberating"])
    df_edges = (
        pd.DataFrame(edges)
        if edges
        else pd.DataFrame(columns=["message_id", "target_id", "interaction_type"])
    )

    return df_messages, df_edges
