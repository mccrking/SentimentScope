"""Shared CSV loading: preserve identifiers and detect text columns."""
import io
from pathlib import Path
import pandas as pd


def read_dataset(source, sep=None, encoding="utf-8-sig"):
    raw = Path(source).read_bytes() if isinstance(source, (str, Path)) else source.getvalue()
    text = raw.decode(encoding)
    # Strings preserve tweet IDs and labels, including numeric-looking values.
    frame = pd.read_csv(io.StringIO(text), sep=sep, engine="python", dtype=str,
                        keep_default_na=False)
    if frame.empty:
        raise ValueError("Le CSV ne contient aucune ligne.")
    return frame


def detect_text_column(frame):
    candidates = {str(c).strip().lower(): c for c in frame.columns}
    for name in ("text", "content", "tweet", "texte", "review", "avis"):
        if name in candidates:
            return candidates[name]
    raise ValueError("Colonne texte non reconnue : indiquez-la explicitement.")
