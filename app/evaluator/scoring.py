def _get_score(scores: dict, custom_id: str) -> float | None:
    return scores.get(custom_id)


def get_external_score(scores: dict, message_id: str) -> float | None:
    return _get_score(scores, message_id)


def get_internal_score(scores: dict, message_id: str) -> float | None:
    if message_id.split("-")[1] != "reasoning":
        return None
    return _get_score(scores, message_id)


def get_explanation_scores(explanation: dict, message_id: str) -> dict:
    return explanation.get(message_id)
