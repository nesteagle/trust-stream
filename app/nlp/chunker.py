import spacy

MIN_TOKEN_LENGTH = 3

# we only need sentencizer
nlp = spacy.load(
    "en_core_web_sm",
    exclude=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer", "ner"],
)
nlp.add_pipe("sentencizer")


def chunk_into_sentences(text: str) -> list[str]:
    """Splits text into sentences using spaCy sentencizer."""
    if not text:
        return []

    doc = nlp(text.strip())
    return [sent.text.strip() for sent in doc.sents if len(sent) > MIN_TOKEN_LENGTH]
