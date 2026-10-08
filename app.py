from pathlib import Path
import pandas as pd
import streamlit as st
from cli import analyze_frame
from sentiment import LABELS, analyze_texts
from dataset import read_dataset, detect_text_column

st.set_page_config(page_title="Sentiment Studio", page_icon="💬", layout="wide")
st.title("💬 Sentiment Studio")
st.caption("Analyse de tweets et d’avis clients · Positif, neutre, négatif")
with st.sidebar:
    st.header("Paramètres")
    backend = st.selectbox("Moteur", ["transformer", "vader"],
                           format_func=lambda x: "DistilBERT · multilingue" if x == "transformer" else "NLTK VADER · anglais")
    if backend == "transformer":
        st.info("Premier lancement : téléchargement du modèle. Inférence locale sur CPU. Les textes longs sont tronqués à 512 tokens.")
    else:
        st.warning("VADER est destiné aux textes anglais. Son score de polarité n’est pas une probabilité.")
    st.caption("Les scores du modèle ne garantissent pas la justesse des prédictions. Ironie et contexte peuvent modifier le sentiment.")

single, dataset = st.tabs(["Un texte", "Dataset CSV"])
with single:
    text = st.text_area("Texte à analyser", placeholder="J’adore ce produit, la livraison était rapide !")
    if st.button("Analyser le texte", type="primary"):
        try:
            with st.spinner("Analyse en cours…"):
                result = analyze_texts([text], backend)[0]
            if result["statut"] != "ok":
                st.warning("Saisissez un texte contenant autre chose qu’un lien ou une mention.")
            else:
                st.success(f"Sentiment : {result['sentiment']}")
                if backend == "transformer":
                    st.bar_chart(pd.DataFrame({"Score": [result[f"score_{label}"] for label in LABELS]}, index=LABELS))
                else:
                    st.metric("Polarité (−1 à +1)", f"{result['polarite']:.3f}")
        except Exception as exc:
            st.error(f"Analyse impossible : {exc}")

with dataset:
    st.write("Importez un CSV contenant une colonne de tweets ou d’avis.")
    upload = st.file_uploader("Dataset", type="csv")
    source = st.selectbox("Dataset intégré", ["Vos tweets", "Exemples synthétiques"])
    separator = st.selectbox("Séparateur", [None, ",", ";", "\t"], format_func=lambda s: "Automatique" if s is None else "Tabulation" if s == "\t" else s)
    encoding = st.selectbox("Encodage", ["utf-8-sig", "cp1252", "latin-1"])
    try:
        demo = Path(__file__).parent / "data" / ("demo_fr.csv" if backend == "transformer" else "demo_en.csv")
        integrated = Path(__file__).parent / "data" / "tweets.csv"
        file = integrated if source == "Vos tweets" else demo
        frame = read_dataset(upload if upload is not None else file, sep=separator, encoding=encoding)
    except Exception as exc:
        st.error(f"Lecture du CSV impossible : {exc}")
        frame = pd.DataFrame()
    if upload is None and source == "Exemples synthétiques":
        st.caption("Exemples synthétiques : ils illustrent l’application et ne mesurent pas les performances du modèle.")
    elif upload is None:
        st.caption("Fichier tweets.csv fourni par vous. Sa provenance n’a pas été vérifiée. Les pseudo-étiquettes ne sont pas des annotations de référence certifiées.")
    if not frame.empty:
        try:
            suggested = detect_text_column(frame)
        except ValueError:
            suggested = frame.columns[0]
        column = st.selectbox("Colonne contenant le texte", frame.columns, index=list(frame.columns).index(suggested))
        st.dataframe(frame.head(20), hide_index=True, width="stretch")
        # Key ties stored results to both source and settings, preventing stale exports.
        signature = (frame.to_csv(index=False), column, backend)
        if st.button(f"Analyser les {len(frame)} lignes", type="primary"):
            try:
                with st.spinner("Chargement du moteur et classification…"):
                    bar = st.progress(0, text="Chargement du modèle…")
                    output = analyze_frame(frame, column, backend,
                        progress=lambda done, total: bar.progress(done / total, text=f"{done} / {total} textes analysés"))
                    bar.empty()
                st.session_state["analysis"] = (signature, output)
            except Exception as exc:
                st.error(f"Analyse impossible : {exc}")
        stored = st.session_state.get("analysis")
        if stored and stored[0] == signature:
            output = stored[1]
            valid = output[output["statut"] == "ok"]
            cols = st.columns(3)
            for col, label in zip(cols, LABELS):
                count = int((valid["sentiment"] == label).sum())
                col.metric(label.capitalize(), count)
            st.bar_chart(valid["sentiment"].value_counts().reindex(LABELS, fill_value=0))
            skipped = len(output) - len(valid)
            if skipped:
                st.info(f"{skipped} ligne(s) vide(s) ou sans texte conservée(s), sans prédiction.")
            filters = st.multiselect("Filtrer les sentiments affichés", LABELS, default=list(LABELS))
            st.dataframe(output[output["sentiment"].isin(filters)], hide_index=True, width="stretch")
            st.download_button("Télécharger tous les résultats CSV", output.to_csv(index=False).encode("utf-8-sig"), "sentiments.csv", "text/csv")
    elif not frame.columns.empty:
        st.warning("Ce CSV ne contient aucune ligne.")
