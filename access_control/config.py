import configparser

config = configparser.ConfigParser()
config.read(r"config.ini")

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
