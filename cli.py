"""CSV batch inference: python cli.py data/demo_fr.csv --output outputs/results.csv"""
import argparse
from pathlib import Path
import pandas as pd
from sentiment import analyze_texts
from dataset import read_dataset, detect_text_column


def analyze_frame(frame, column, backend, progress=None):
    if column not in frame.columns:
        raise ValueError(f"Colonne '{column}' absente. Colonnes : {', '.join(frame.columns)}")
    reserved = {"texte_nettoye", "sentiment", "score", "statut", "moteur", "polarite",
                "score_positif", "score_neutre", "score_négatif"}
    conflicts = set(frame.columns) & reserved
    if conflicts:
        raise ValueError(f"Colonnes réservées déjà présentes : {', '.join(sorted(conflicts))}")
    results = pd.DataFrame(analyze_texts(frame[column].tolist(), backend, progress=progress), index=frame.index)
    return pd.concat([frame, results], axis=1)


def main():
    parser = argparse.ArgumentParser(description="Classification de sentiment sur CSV")
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/results.csv"))
    parser.add_argument("--column", default=None)
    parser.add_argument("--backend", choices=["transformer", "vader"], default="transformer")
    parser.add_argument("--sep", default=None, help="Détection automatique par défaut")
    parser.add_argument("--encoding", default="utf-8-sig")
    args = parser.parse_args()
    try:
        frame = read_dataset(args.input, sep=args.sep, encoding=args.encoding)
        column = args.column or detect_text_column(frame)
        print(f"Analyse de {len(frame)} lignes, colonne {column}, moteur {args.backend}", flush=True)
        result = analyze_frame(frame, column, args.backend)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(args.output, index=False, encoding="utf-8-sig")
    except (ValueError, OSError, RuntimeError, ImportError) as exc:
        parser.exit(1, f"Erreur : {exc}\n")
    print(f"{len(result)} lignes exportées vers {args.output}")
    print(result["sentiment"].value_counts().to_string())


if __name__ == "__main__":
    main()
