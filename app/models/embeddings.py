from sentence_transformers import SentenceTransformer

class AlignmentEmbedder:
    def __init__(
        self, model_name: str = "microsoft/harrier-oss-v1-0.6b", max_seq_length=768
    ):
        self.model = SentenceTransformer(model_name)
        self.model.max_seq_length = max_seq_length

    def encode(self, texts: list[str], batch_size: int = 8):
        """Encodes text using embeddings model."""
        if not texts:
            return []
        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
