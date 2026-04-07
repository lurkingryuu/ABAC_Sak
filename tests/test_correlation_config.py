import configparser
import importlib.util
import json
import os
import tempfile
import unittest

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
INPUT_PATH = os.path.join(ROOT_DIR, "access_control", "input.py")
SPEC = importlib.util.spec_from_file_location("input_module", INPUT_PATH)
INPUT_MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(INPUT_MODULE)
validate_input_semantics = INPUT_MODULE.validate_input_semantics
write_config_file = INPUT_MODULE.write_config_file


class CorrelationConfigTests(unittest.TestCase):
    def test_validate_input_semantics_accepts_valid_pairs(self):
        payload = {
            "subject_attributes_count": 2,
            "object_attributes_count": 1,
            "environment_attributes_count": 1,
            "subject_attributes_values": [2, 2],
            "object_attributes_values": [2],
            "environment_attributes_values": [2],
            "subject_distributions": [{"distribution": "U"}, {"distribution": "U"}],
            "object_distributions": [{"distribution": "U"}],
            "environment_distributions": [{"distribution": "U"}],
            "correlations": {
                "subject": {
                    "pairs": [
                        {
                            "attr_a": 1,
                            "attr_b": 2,
                            "target": {"joint_table": [[0.3, 0.2], [0.1, 0.4]]},
                        }
                    ]
                }
            },
        }
        validate_input_semantics(payload)

    def test_validate_input_semantics_rejects_bad_dimensions(self):
        payload = {
            "subject_attributes_count": 2,
            "object_attributes_count": 1,
            "environment_attributes_count": 1,
            "subject_attributes_values": [2, 2],
            "object_attributes_values": [2],
            "environment_attributes_values": [2],
            "subject_distributions": [{"distribution": "U"}, {"distribution": "U"}],
            "object_distributions": [{"distribution": "U"}],
            "environment_distributions": [{"distribution": "U"}],
            "correlations": {
                "subject": {
                    "pairs": [
                        {
                            "attr_a": 1,
                            "attr_b": 2,
                            "target": {"joint_table": [[1.0], [0.0]]},
                        }
                    ]
                }
            },
        }
        with self.assertRaises(ValueError):
            validate_input_semantics(payload)

    def test_write_config_file_persists_sampling_and_correlations(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = os.path.join(tmp, "config.ini")
            os.environ["ABAC_CONFIG_INI"] = config_path
            write_config_file(
                4,
                4,
                1,
                2,
                1,
                1,
                {
                    "values": [3, 3],
                    "distributions": [{"distribution": "U"}, {"distribution": "U"}],
                    "correlations": {"pairs": [{"attr_a": 1, "attr_b": 2, "target": {"cramers_v": 0.7}}]},
                },
                {"values": [2], "distributions": [{"distribution": "U"}], "correlations": {}},
                {"values": [2], "distributions": [{"distribution": "U"}], "correlations": {}},
                0,
                0,
                seed=13,
                sampling_config={"alpha": 0.8, "beta": 0.2},
            )
            cfg = configparser.ConfigParser()
            cfg.read(config_path)
            self.assertEqual(cfg["NUMBERS"]["seed"], "13")
            self.assertEqual(json.loads(cfg["SAMPLING"]["config"])["alpha"], 0.8)
            self.assertIn("pairs", json.loads(cfg["SUBJECT_ATTRIBUTES"]["correlations"]))


if __name__ == "__main__":
    unittest.main()
