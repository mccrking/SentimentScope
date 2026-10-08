import unittest
from unittest.mock import patch
import pandas as pd
from sentiment import analyze_texts, clean_text
from cli import analyze_frame


class SentimentTests(unittest.TestCase):
    def test_cleaning_preserves_sentiment_signals(self):
        self.assertEqual(clean_text("@shop Je n’aime pas ça 😡 ! https://exemple.fr #déçu"),
                         "Je n’aime pas ça 😡 ! #déçu")

    def test_empty_rows_do_not_load_model(self):
        with patch("sentiment.load_engine") as loader:
            output = analyze_texts([None, float("nan"), "  ", "@shop https://example.com"])
            loader.assert_not_called()
        self.assertTrue(all(r["statut"] == "texte_vide" for r in output))

    def test_batch_mapping_and_alignment(self):
        predictions = [
            [{"label": "negative", "score": .8}, {"label": "positive", "score": .1}, {"label": "neutral", "score": .1}],
            [{"label": "neutral", "score": .2}, {"label": "negative", "score": .1}, {"label": "positive", "score": .7}],
        ]
        with patch("sentiment.load_engine") as loader:
            loader.return_value.return_value = predictions
            result = analyze_frame(pd.DataFrame({"text": ["bad", None, "good"], "id": [1, 2, 3]}), "text", "transformer")
            self.assertEqual(loader.return_value.call_args.args[0], ["bad", "good"])
        self.assertEqual(result["sentiment"].tolist(), ["négatif", None, "positif"])
        self.assertEqual(result["id"].tolist(), [1, 2, 3])
        self.assertAlmostEqual(result.iloc[2]["score"], .7)

    def test_vader_real_negation_and_neutral(self):
        output = analyze_texts(["I am happy", "I am not happy", "The store opens at nine"], "vader")
        self.assertEqual([r["sentiment"] for r in output], ["positif", "négatif", "neutre"])
        self.assertTrue(all(r["score"] is None for r in output))

    def test_missing_column(self):
        with self.assertRaisesRegex(ValueError, "absente"):
            analyze_frame(pd.DataFrame({"review": ["test"]}), "text", "vader")


if __name__ == "__main__":
    unittest.main()
