import re

import pandas as pd
from eval_judge import ContextMessage


def _format_section(title: str, items: list) -> str:
    """Helper to format list sections cleanly, ignoring empty/NaN cases."""
    if not isinstance(items, list) or not items:
        return ""
    bullets = "\n".join(f"- {item}" for item in items if pd.notna(item))
    return f"{title}:\n{bullets}\n\n" if bullets else ""


def build_request_contents(
    df_messages: pd.DataFrame, df_rounds: pd.DataFrame
) -> list[ContextMessage]:
    """Consumes message and rounds data, constructs context payloads, and yields evaluation messages."""
    # VAST Challenge specific regex for `action` type messages
    ACTION_MESSAGE_PATTERN = re.compile(r"^[A-Z:_]+\s*")

    rounds_dict = df_rounds.set_index("round_index").to_dict(orient="index")
    requests = []

    for round_idx, round_msg_df in df_messages.groupby("round_index"):
        round_info = rounds_dict.get(round_idx)
        if not round_info:
            continue

        SIMPLE_FIELDS = [("Sentiment", "social_state")]
        LIST_FIELDS = [
            ("Deadlines", "critical_deadlines"),
            ("Media activity", "media_events"),
            ("External activity", "external_actor_actions"),
            ("News", "news"),
        ]
        chunks = [f"Situation: {round_info['event_narrative']}"]
        chunks += [
            f"{label}: {round_info[k]}"
            for label, k in SIMPLE_FIELDS
            if pd.notna(round_info.get(k))
        ]
        chunks += [
            _format_section(label, round_info.get(k)) for label, k in LIST_FIELDS
        ]

        context = "\n\n".join(c for c in chunks if c).strip()

        for message in round_msg_df.itertuples():
            message_text = getattr(message, "content", "")

            if getattr(message, "message_type", None) == "action":
                message_text = ACTION_MESSAGE_PATTERN.sub("", message_text)
                if not message_text.strip():
                    continue

            requests.append(
                ContextMessage(
                    message_id=message.message_id,
                    context=context,
                    content=message_text,
                    reasoning=False,
                )
            )

            reasoning = getattr(message, "reasoning", None)
            if pd.notna(reasoning) and str(reasoning).strip():
                requests.append(
                    ContextMessage(
                        message_id=message.message_id,
                        content=reasoning,
                        reasoning=True,
                    )
                )

    return requests
