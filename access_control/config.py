import configparser
import os

ENV_CONFIG_INI = "ABAC_CONFIG_INI"

config = configparser.ConfigParser()

# Prefer an explicit config path (per-job) via env, otherwise fall back to access_control/config.ini.
_default_config_path = os.path.join(os.path.dirname(__file__), "config.ini")
_config_path = os.environ.get(ENV_CONFIG_INI) or _default_config_path
config.read(_config_path)

n1 = int(config["NUMBERS"]["n1"])
n2 = int(config["NUMBERS"]["n2"])
n3 = int(config["NUMBERS"]["n3"])
n4 = int(config["NUMBERS"]["n4"])
n5 = int(config["NUMBERS"]["n5"])
n6 = int(config["NUMBERS"]["n6"])

subject_attributes = list(map(int, config["SUBJECT_ATTRIBUTES"]["values"].split(",")))

object_attributes = list(map(int, config["OBJECT_ATTRIBUTES"]["values"].split(",")))

environment_attributes = list(map(int, config["ENVIRONMENT_ATTRIBUTES"]["values"].split(",")))

n = int(config["RULES"]["n"])
