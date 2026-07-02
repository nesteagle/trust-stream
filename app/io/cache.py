import os
import pickle


def load_embedding_cache(file_path: str) -> dict:
    """Loads the embedding cache from disk. Returns empty dict if missing."""
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            return pickle.load(f)
    return {}


def save_embedding_cache(cache: dict, file_path: str) -> None:
    """Writes the embedding dictionary to disk."""
    temp_path = f"{file_path}.tmp"
    with open(temp_path, "wb") as f:
        pickle.dump(cache, f)
    os.replace(temp_path, file_path)
