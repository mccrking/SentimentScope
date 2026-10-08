"""Sentiment inference shared by the web interface and CLI."""
import re
from functools import lru_cache

MODEL_ID = "lxyuan/distilbert-base-multilingual-cased-sentiments-student"
LABELS = ("positif", "neutre", "négatif")
LABEL_MAP = {"positive": "positif", "neutral": "neutre", "negative": "négatif"}


def clean_text(text: str) -> str:
    """Preserve accents, emojis and negation; remove URLs and user handles."""
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"(?<!\w)@\w+", " ", text)
    return " ".join(text.split())


@lru_cache(maxsize=2)
def load_engine(backend: str):
    if backend == "transformer":
        import torch
        torch.set_num_threads(min(4, torch.get_num_threads()))
        from transformers import pipeline
        return pipeline("text-classification", model=MODEL_ID, device=-1)
    if backend == "vader":
        import nltk
        from nltk.sentiment import SentimentIntensityAnalyzer
        try:
            return SentimentIntensityAnalyzer()
        except LookupError:
            if not nltk.download("vader_lexicon", quiet=True):
                raise RuntimeError("Téléchargement du lexique VADER impossible.")
            return SentimentIntensityAnalyzer()
    raise ValueError(f"Moteur inconnu : {backend}")


def analyze_texts(texts, backend="transformer", batch_size=16, progress=None):
    """Return one result per input, including an explicit status for empty rows."""
    if backend not in {"transformer", "vader"}:
        raise ValueError(f"Moteur inconnu : {backend}")
    if batch_size < 1:
        raise ValueError("batch_size doit être positif.")
    cleaned = [clean_text(t) if isinstance(t, str) else "" for t in texts]
    results = [{"texte_nettoye": t, "sentiment": None, "score": None,
                "statut": "texte_vide", "moteur": backend} for t in cleaned]
    indices = [i for i, t in enumerate(cleaned) if t]
    if not indices:
        return results
    engine = load_engine(backend)
    if backend == "transformer":
        predictions = []
        for start in range(0, len(indices), batch_size):
            batch = indices[start:start + batch_size]
            predictions.extend(engine([cleaned[i] for i in batch], batch_size=batch_size,
                                      truncation=True, max_length=512, top_k=None))
            if progress:
                progress(min(start + batch_size, len(indices)), len(indices))
        if len(predictions) != len(indices):
            raise RuntimeError("Nombre de prédictions inattendu.")
        for i, scores in zip(indices, predictions):
            translated = {LABEL_MAP[s["label"].lower()]: float(s["score"]) for s in scores}
            if set(translated) != set(LABELS):
                raise RuntimeError("Le modèle doit fournir les trois classes attendues.")
            label = max(translated, key=translated.get)
            results[i].update(sentiment=label, score=translated[label], statut="ok",
                              **{f"score_{k}": v for k, v in translated.items()})
    else:
        for i in indices:
            scores = engine.polarity_scores(cleaned[i])
            compound = scores["compound"]
            label = "positif" if compound >= .05 else "négatif" if compound <= -.05 else "neutre"
            # Compound is a polarity, not a probability or confidence.
            results[i].update(sentiment=label, statut="ok", polarite=compound)
        if progress:
            progress(len(indices), len(indices))
    return results
