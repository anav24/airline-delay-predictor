from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class TestProjectStructure(unittest.TestCase):
    def test_required_project_files_exist(self):
        required_paths = [
            "README.md",
            "requirements.txt",
            ".gitignore",
            "src/train_final_departure_model.py",
            "src/predict_departure_delay.py",
            "docs/prediction_input_schema.md",
            "docs/model_card.md",
            "docs/project_checklist.md",
        ]

        for relative_path in required_paths:
            with self.subTest(relative_path=relative_path):
                self.assertTrue((PROJECT_ROOT / relative_path).exists())

    def test_local_artifact_rules_are_ignored(self):
        gitignore = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8")

        expected_rules = [
            "data/",
            "models/",
            "visuals/",
            "*.csv",
            "*.parquet",
            "*.joblib",
            "*.pkl",
        ]

        for rule in expected_rules:
            with self.subTest(rule=rule):
                self.assertIn(rule, gitignore)

    def test_readme_links_documentation(self):
        readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")

        expected_links = [
            "docs/prediction_input_schema.md",
            "docs/model_card.md",
            "docs/project_checklist.md",
            "src/train_final_departure_model.py",
            "src/predict_departure_delay.py",
        ]

        for link in expected_links:
            with self.subTest(link=link):
                self.assertIn(link, readme)


if __name__ == "__main__":
    unittest.main()
