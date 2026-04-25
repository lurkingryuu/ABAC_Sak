
import configparser
import json
import os

ENV_INPUT_JSON = "ABAC_INPUT_JSON"
ENV_CONFIG_INI = "ABAC_CONFIG_INI"

def parse_attribute_values_with_stars(raw_values, global_stars=0):
    """
    Parse attribute values that can be:
    - A single number: e.g., 5 (means 5 values, use global_stars for stars)
    - A pair: e.g., [5, 2] (means 5 values, 2 stars)
    
    Returns:
    - num_values_list: list of number of values for each attribute
    - num_stars_list: list of number of stars for each attribute
    """
    num_values_list = []
    num_stars_list = []
    
    for item in raw_values:
        if isinstance(item, list):
            # Pair: [num_values, num_stars]
            num_values_list.append(item[0])
            num_stars_list.append(item[1])
        else:
            # Single number: use global_stars as default
            num_values_list.append(item)
            num_stars_list.append(global_stars)
    
    return num_values_list, num_stars_list

def write_config_file(n1, n2, n3, n4, n5, n6, subject_attributes, object_attributes, environment_attributes, accepted_rules_count, denied_rules_count):
    """
    Write the configuration to config.ini.
    
    subject_attributes, object_attributes, environment_attributes each should have:
    - "values": list of number of values per attribute
    - "stars": list of number of stars per attribute (optional)
    - "distributions": list of distribution dicts
    """
    config = configparser.ConfigParser()
    
    # Write numbers
    config['NUMBERS'] = {
        'n1': str(n1),
        'n2': str(n2),
        'n3': str(n3),
        'n4': str(n4),
        'n5': str(n5),
        'n6': str(n6)
    }
    
    # Write subject attributes
    config['SUBJECT_ATTRIBUTES'] = {
        'count': str(n4),
        'values': ','.join(map(str, subject_attributes["values"])),
        'stars': ','.join(map(str, subject_attributes.get("stars", [0] * len(subject_attributes["values"])))),
        'distributions': json.dumps(subject_attributes["distributions"])
    }
    
    # Write object attributes
    config['OBJECT_ATTRIBUTES'] = {
        'count': str(n5),
        'values': ','.join(map(str, object_attributes["values"])),
        'stars': ','.join(map(str, object_attributes.get("stars", [0] * len(object_attributes["values"])))),
        'distributions': json.dumps(object_attributes["distributions"])
    }
    
    # Write environment attributes
    config['ENVIRONMENT_ATTRIBUTES'] = {
        'count': str(n6),
        'values': ','.join(map(str, environment_attributes["values"])),
        'stars': ','.join(map(str, environment_attributes.get("stars", [0] * len(environment_attributes["values"])))),
        'distributions': json.dumps(environment_attributes["distributions"])
    }
    
    # Write rules counts (accepted and denied)
    config['RULES'] = {
        'accepted_rules_count': str(accepted_rules_count),
        'denied_rules_count': str(denied_rules_count)
    }
    
    # Write config.ini: default is access_control/config.ini, but can be overridden per-job.
    config_path = os.environ.get(ENV_CONFIG_INI) or os.path.join(
        os.path.dirname(__file__), 'config.ini'
    )
    os.makedirs(os.path.dirname(os.path.abspath(config_path)), exist_ok=True)
    with open(config_path, 'w') as configfile:
        config.write(configfile)

def read_input_json(file_path):
    """
    Read input.json and return the parsed data.
    """
    with open(file_path, "r") as file:
        data = json.load(file)
        return data

if __name__ == "__main__":
    # Read input.json: default is ../uploads/input.json, but can be overridden per-job.
    input_path = os.environ.get(ENV_INPUT_JSON) or os.path.join(
        os.path.dirname(__file__), '../uploads/input.json'
    )
    input_data = read_input_json(input_path)
    
    # Extract values from JSON
    n1 = input_data["subject_size"]
    n2 = input_data["object_size"]
    n3 = input_data["environment_size"]
    n4 = input_data["subject_attributes_count"]
    n5 = input_data["object_attributes_count"]
    n6 = input_data["environment_attributes_count"]
    
    # Get global stars (optional, defaults to 0)
    global_stars = input_data.get("global_stars", 0)
    
    # Parse attribute values with stars
    subject_values, subject_stars = parse_attribute_values_with_stars(
        input_data["subject_attributes_values"], global_stars
    )
    object_values, object_stars = parse_attribute_values_with_stars(
        input_data["object_attributes_values"], global_stars
    )
    environment_values, environment_stars = parse_attribute_values_with_stars(
        input_data["environment_attributes_values"], global_stars
    )
    
    subject_attributes = {
        "values": subject_values,
        "stars": subject_stars,
        "distributions": input_data["subject_distributions"]
    }
    object_attributes = {
        "values": object_values,
        "stars": object_stars,
        "distributions": input_data["object_distributions"]
    }
    environment_attributes = {
        "values": environment_values,
        "stars": environment_stars,
        "distributions": input_data["environment_distributions"]
    }
    accepted_rules_count = input_data["accepted_rules_count"]
    denied_rules_count = input_data["denied_rules_count"]

    # Write to config.ini
    write_config_file(n1, n2, n3, n4, n5, n6, subject_attributes, object_attributes, environment_attributes, accepted_rules_count, denied_rules_count)    }
    
    # Write environment attributes
    config['ENVIRONMENT_ATTRIBUTES'] = {
        'count': str(n6),
        'values': ','.join(map(str, environment_attributes["values"])),
        'distributions': json.dumps(environment_attributes["distributions"])
    }
    
    # Write rules count
    config['RULES'] = {'rules_count': str(N)}
    
    # Write config.ini: default is access_control/config.ini, but can be overridden per-job.
    config_path = os.environ.get(ENV_CONFIG_INI) or os.path.join(
        os.path.dirname(__file__), 'config.ini'
    )
    os.makedirs(os.path.dirname(os.path.abspath(config_path)), exist_ok=True)
    with open(config_path, 'w') as configfile:
        config.write(configfile)

def read_input_json(file_path):
    """
    Read input.json and return the parsed data.
    """
    with open(file_path, "r") as file:
        data = json.load(file)
        return data

if __name__ == "__main__":
    # Read input.json: default is ../uploads/input.json, but can be overridden per-job.
    input_path = os.environ.get(ENV_INPUT_JSON) or os.path.join(
        os.path.dirname(__file__), '../uploads/input.json'
    )
    input_data = read_input_json(input_path)
    
    # Extract values from JSON
    n1 = input_data["subject_size"]
    n2 = input_data["object_size"]
    n3 = input_data["environment_size"]
    n4 = input_data["subject_attributes_count"]
    n5 = input_data["object_attributes_count"]
    n6 = input_data["environment_attributes_count"]
    subject_attributes = {
        "values": input_data["subject_attributes_values"],
        "distributions": input_data["subject_distributions"]
    }
    object_attributes = {
        "values": input_data["object_attributes_values"],
        "distributions": input_data["object_distributions"]
    }
    environment_attributes = {
        "values": input_data["environment_attributes_values"],
        "distributions": input_data["environment_distributions"]
    }
    N = input_data["rules_count"]

    # Write to config.ini
    write_config_file(n1, n2, n3, n4, n5, n6, subject_attributes, object_attributes, environment_attributes, N)
