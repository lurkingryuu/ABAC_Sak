
import configparser
import json
import os

def write_config_file(n1, n2, n3, n4, n5, n6, subject_attributes, object_attributes, environment_attributes, N):
    """
    Write the configuration to config.ini.
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
        'distributions': json.dumps(subject_attributes["distributions"])
    }
    
    # Write object attributes
    config['OBJECT_ATTRIBUTES'] = {
        'count': str(n5),
        'values': ','.join(map(str, object_attributes["values"])),
        'distributions': json.dumps(object_attributes["distributions"])
    }
    
    # Write environment attributes
    config['ENVIRONMENT_ATTRIBUTES'] = {
        'count': str(n6),
        'values': ','.join(map(str, environment_attributes["values"])),
        'distributions': json.dumps(environment_attributes["distributions"])
    }
    
    # Write rules count
    config['RULES'] = {'rules_count': str(N)}
    
    # Write config.ini inside the access_control directory
    config_path = os.path.join(os.path.dirname(__file__), 'config.ini')
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
    # Read input.json from the uploads directory
    input_path = os.path.join(os.path.dirname(__file__), '../uploads/input.json')
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