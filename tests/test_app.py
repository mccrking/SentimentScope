import unittest
from pathlib import Path
from streamlit.testing.v1 import AppTest


class AppTests(unittest.TestCase):
    def test_supplied_dataset_full_interface(self):
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"))
        app.run(timeout=30)
        self.assertEqual(app.selectbox[3].value, "content")
        app.sidebar.selectbox[0].select("vader").run()
        app.button[1].click().run(timeout=30)
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.error), 0)
        self.assertEqual(sum(int(m.value) for m in app.metric), 2243)

    def test_vader_dataset_flow_and_stale_results(self):
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"))
        app.run(timeout=30)
        self.assertEqual(len(app.exception), 0)
        app.sidebar.selectbox[0].select("vader").run()
        app.selectbox[0].select("Exemples synthétiques").run()
        app.button[1].click().run(timeout=30)
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.error), 0)
        self.assertEqual(len(app.metric), 3)
        self.assertEqual(sum(int(m.value) for m in app.metric), 8)
        app.sidebar.selectbox[0].select("transformer").run()
        self.assertEqual(len(app.metric), 0)


if __name__ == "__main__":
    unittest.main()
