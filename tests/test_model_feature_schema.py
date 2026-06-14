from pathlib import Path
import unittest
import importlib.util


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = PROJECT_ROOT / "src" / "train_final_departure_model.py"


def load_train_module():
    spec = importlib.util.spec_from_file_location(
        "train_final_departure_model",
        TRAIN_SCRIPT,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestModelFeatureSchema(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train_module = load_train_module()

    def test_target_is_not_used_as_feature(self):
        all_features = (
            self.train_module.NUMERIC_FEATURES
            + self.train_module.CATEGORICAL_FEATURES
        )

        self.assertNotIn(self.train_module.TARGET, all_features)

    def test_departure_delay_features_are_included(self):
        required_departure_features = [
            "DEP_DELAY",
            "DEP_DELAY_NEW",
            "DEP_DEL15",
            "DEP_DELAY_GROUP",
            "TAXI_OUT",
            "WHEELS_OFF",
        ]

        all_features = (
            self.train_module.NUMERIC_FEATURES
            + self.train_module.CATEGORICAL_FEATURES
        )

        for feature in required_departure_features:
            with self.subTest(feature=feature):
                self.assertIn(feature, all_features)

    def test_schema_doc_mentions_all_model_features(self):
        schema = (
            PROJECT_ROOT / "docs" / "prediction_input_schema.md"
        ).read_text(encoding="utf-8")

        all_features = (
            self.train_module.NUMERIC_FEATURES
            + self.train_module.CATEGORICAL_FEATURES
        )

        for feature in all_features:
            with self.subTest(feature=feature):
                self.assertIn(feature, schema)


if __name__ == "__main__":
    unittest.main()
