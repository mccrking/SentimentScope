import io
import unittest
from dataset import read_dataset, detect_text_column


class DatasetTests(unittest.TestCase):
    def test_semicolon_and_identifier_preservation(self):
        frame = read_dataset(io.BytesIO(b'id;content\n001;NA\n002;hello\n'))
        self.assertEqual(frame.id.tolist(), ["001", "002"])
        self.assertEqual(frame.content.tolist(), ["NA", "hello"])
        self.assertEqual(detect_text_column(frame), "content")

    def test_supplied_tweets(self):
        frame = read_dataset("data/tweets.csv")
        self.assertEqual(len(frame), 2243)
        self.assertEqual(detect_text_column(frame), "content")
        self.assertTrue(frame.content.str.len().gt(0).all())

    def test_empty_csv(self):
        with self.assertRaisesRegex(ValueError, "aucune ligne"):
            read_dataset(io.BytesIO(b'text\n'))
