import json
import unittest

import app as app_module


def make_valid_payload():
    return {
        "subject_size": 2,
        "object_size": 2,
        "environment_size": 2,
        "subject_attributes_count": 2,
        "object_attributes_count": 1,
        "environment_attributes_count": 1,
        "subject_attributes_values": [3, 2],
        "object_attributes_values": [2],
        "environment_attributes_values": [2],
        "subject_distributions": [{"distribution": "U"}, {"distribution": "U"}],
        "object_distributions": [{"distribution": "U"}],
        "environment_distributions": [{"distribution": "U"}],
        "permit_rules_count": 1,
        "deny_rules_count": 0,
    }


class _NoOpThread:
    def __init__(self, target=None, args=(), daemon=None):
        self.target = target
        self.args = args
        self.daemon = daemon

    def start(self):
        return None


class StarIntegrationTests(unittest.TestCase):
    def setUp(self):
        self._orig_verify = app_module.verify_recaptcha
        self._orig_thread = app_module.threading.Thread
        app_module.verify_recaptcha = lambda _resp: True
        app_module.threading.Thread = _NoOpThread

    def tearDown(self):
        app_module.verify_recaptcha = self._orig_verify
        app_module.threading.Thread = self._orig_thread

    def test_validate_abac_payload_accepts_legacy_numeric_values(self):
        payload = make_valid_payload()
        app_module.validate_abac_payload(payload)

    def test_validate_abac_payload_accepts_star_pairs(self):
        payload = make_valid_payload()
        payload["global_stars"] = 1
        payload["subject_attributes_values"] = [[3, 2], 2]
        payload["object_attributes_values"] = [[2, 1]]
        payload["environment_attributes_values"] = [[2, 0]]
        app_module.validate_abac_payload(payload)

    def test_validate_abac_payload_rejects_invalid_star_tuple(self):
        payload = make_valid_payload()
        payload["subject_attributes_values"] = [[3, -1], 2]
        with self.assertRaises(ValueError):
            app_module.validate_abac_payload(payload)

    def test_validate_abac_payload_rejects_count_mismatch(self):
        payload = make_valid_payload()
        payload["subject_attributes_values"] = [3]
        with self.assertRaises(ValueError):
            app_module.validate_abac_payload(payload)

    def test_upload_json_rejects_invalid_star_shape(self):
        payload = make_valid_payload()
        payload["subject_attributes_values"] = [[3], 2]
        payload["g-recaptcha-response"] = "ok"
        with app_module.app.test_request_context("/upload-json", method="POST", json=payload):
            response, status_code = app_module.upload_json()
        self.assertEqual(status_code, 400)
        body = response.get_json()
        self.assertIn("subject_attributes_values", body["error"])

    def test_merge_multimodal_preserves_user_star_values(self):
        min_config = make_valid_payload()
        min_config["global_stars"] = 2
        min_config["subject_attributes_values"] = [[3, 2], 2]
        extracted_config = {
            "subject_attributes_values": [9, 9],
            "subject_distributions": [{"distribution": "U"}, {"distribution": "U"}],
        }
        merged = app_module.merge_multimodal_configs(min_config, extracted_config)
        self.assertEqual(merged["subject_attributes_values"], [[3, 2], 2])
        self.assertEqual(merged["global_stars"], 2)

    def test_example_endpoint_returns_star_enabled_sample(self):
        body, status_code, _headers = app_module.get_example_json()
        self.assertEqual(status_code, 200)
        payload = json.loads(body)
        self.assertIn("global_stars", payload)
        self.assertIsInstance(payload["subject_attributes_values"][0], list)


if __name__ == "__main__":
    unittest.main()
