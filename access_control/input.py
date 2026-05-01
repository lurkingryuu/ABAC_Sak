
import configparser
import json
import os

ENV_INPUT_JSON = "ABAC_INPUT_JSON"
ENV_CONFIG_INI = "ABAC_CONFIG_INI"

def _ensure_lengths(name, values, distributions, expected_count):
    if len(values) != expected_count:
        raise ValueError(
            f"{name}_attributes_values must have exactly {expected_count} entries"
        )
    if len(distributions) != expected_count:
        raise ValueError(
            f"{name}_distributions must have exactly {expected_count} entries"
        )


def _normalize_joint_table(table):
    total = 0.0
    normalized = []
    for row in table:
        row_out = []
        for value in row:
            v = float(value)
            if v < 0:
                raise ValueError("joint_table probabilities must be non-negative")
            row_out.append(v)
            total += v
        normalized.append(row_out)
    if total <= 0:
        raise ValueError("joint_table must have positive total mass")
    return [[v / total for v in row] for row in normalized]


def _attribute_counts_only(raw_values):
    """
    Accepts mixed format values:
    - int: 5
    - pair: [5, 2]  (values, stars)

    Returns list[int] containing only the value counts.
    """
    counts = []
    for item in raw_values:
        if isinstance(item, list):
            if len(item) != 2:
                raise ValueError("Attribute value pair must be [num_values, num_stars]")
            counts.append(int(item[0]))
        else:
            counts.append(int(item))
    return counts


def parse_attribute_values_with_stars(raw_values, global_stars=0):
    """
    Parse attribute values that can be:
    - A single number: e.g., 5 (means 5 values, uses global_stars for stars)
    - A pair: e.g., [5, 2] (means 5 values, 2 stars)

    Returns:
    - num_values_list: list[int]
    - num_stars_list: list[int]
    """
    num_values_list = []
    num_stars_list = []

    for item in raw_values:
        if isinstance(item, list):
            if len(item) != 2:
                raise ValueError("Attribute value pair must be [num_values, num_stars]")
            num_values_list.append(int(item[0]))
            num_stars_list.append(int(item[1]))
        else:
            num_values_list.append(int(item))
            num_stars_list.append(int(global_stars))

    return num_values_list, num_stars_list


def _validate_correlation_pairs(entity_name, pair_list, attribute_values):
    attribute_count = len(attribute_values)
    for idx, pair in enumerate(pair_list):
        pair_path = f"correlations.{entity_name}.pairs[{idx}]"
        attr_a = int(pair.get("attr_a", 0))
        attr_b = int(pair.get("attr_b", 0))
        if attr_a < 1 or attr_a > attribute_count:
            raise ValueError(f"{pair_path}.attr_a is out of range 1..{attribute_count}")
        if attr_b < 1 or attr_b > attribute_count:
            raise ValueError(f"{pair_path}.attr_b is out of range 1..{attribute_count}")
        if attr_a == attr_b:
            raise ValueError(f"{pair_path} cannot correlate an attribute with itself")
        target = pair.get("target", {})
        if "joint_table" in target:
            rows = target["joint_table"]
            if not isinstance(rows, list) or not rows:
                raise ValueError(f"{pair_path}.target.joint_table must be a non-empty 2D array")
            expected_rows = int(attribute_values[attr_a - 1])
            expected_cols = int(attribute_values[attr_b - 1])
            if len(rows) != expected_rows:
                raise ValueError(
                    f"{pair_path}.target.joint_table row count must equal attribute-{attr_a} domain ({expected_rows})"
                )
            for row in rows:
                if not isinstance(row, list) or len(row) != expected_cols:
                    raise ValueError(
                        f"{pair_path}.target.joint_table column count must equal attribute-{attr_b} domain ({expected_cols})"
                    )
            pair["target"]["joint_table"] = _normalize_joint_table(rows)
        elif "cramers_v" in target:
            value = float(target["cramers_v"])
            if value < 0 or value > 1:
                raise ValueError(f"{pair_path}.target.cramers_v must be in [0, 1]")
        else:
            raise ValueError(
                f"{pair_path}.target must define either joint_table or cramers_v"
            )


def validate_input_semantics(input_data):
    n4 = int(input_data["subject_attributes_count"])
    n5 = int(input_data["object_attributes_count"])
    n6 = int(input_data["environment_attributes_count"])

    subject_values_raw = input_data["subject_attributes_values"]
    object_values_raw = input_data["object_attributes_values"]
    environment_values_raw = input_data["environment_attributes_values"]
    subject_distributions = input_data["subject_distributions"]
    object_distributions = input_data["object_distributions"]
    environment_distributions = input_data["environment_distributions"]

    _ensure_lengths("subject", subject_values_raw, subject_distributions, n4)
    _ensure_lengths("object", object_values_raw, object_distributions, n5)
    _ensure_lengths("environment", environment_values_raw, environment_distributions, n6)

    correlations = input_data.get("correlations", {})
    # Correlation validation expects plain integer domains; if the user provided
    # star-pairs ([values, stars]) we validate using the extracted value counts.
    subject_values = _attribute_counts_only(subject_values_raw)
    object_values = _attribute_counts_only(object_values_raw)
    environment_values = _attribute_counts_only(environment_values_raw)
    for entity_name, values in (
        ("subject", subject_values),
        ("object", object_values),
        ("environment", environment_values),
    ):
        entity_block = correlations.get(entity_name, {})
        pair_list = entity_block.get("pairs", [])
        _validate_correlation_pairs(entity_name, pair_list, values)


def write_config_file(
    n1,
    n2,
    n3,
    n4,
    n5,
    n6,
    subject_attributes,
    object_attributes,
    environment_attributes,
    permit_rules,
    deny_rules,
    seed=None,
    sampling_config=None,
):
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
        'n6': str(n6),
        'seed': '' if seed is None else str(seed),
    }
    
    # Write subject attributes
    config['SUBJECT_ATTRIBUTES'] = {
        'count': str(n4),
        'values': ','.join(map(str, subject_attributes["values"])),
        'stars': ','.join(map(str, subject_attributes.get("stars", [0] * len(subject_attributes["values"])))),
        'distributions': json.dumps(subject_attributes["distributions"]),
        'correlations': json.dumps(subject_attributes.get("correlations", {})),
    }
    
    # Write object attributes
    config['OBJECT_ATTRIBUTES'] = {
        'count': str(n5),
        'values': ','.join(map(str, object_attributes["values"])),
        'stars': ','.join(map(str, object_attributes.get("stars", [0] * len(object_attributes["values"])))),
        'distributions': json.dumps(object_attributes["distributions"]),
        'correlations': json.dumps(object_attributes.get("correlations", {})),
    }
    
    # Write environment attributes
    config['ENVIRONMENT_ATTRIBUTES'] = {
        'count': str(n6),
        'values': ','.join(map(str, environment_attributes["values"])),
        'stars': ','.join(map(str, environment_attributes.get("stars", [0] * len(environment_attributes["values"])))),
        'distributions': json.dumps(environment_attributes["distributions"]),
        'correlations': json.dumps(environment_attributes.get("correlations", {})),
    }
    
    # Write rules count
    config['RULES'] = {
        'permit_rules_count': str(permit_rules),
        'deny_rules_count': str(deny_rules)
    }

    config['SAMPLING'] = {
        'config': json.dumps(sampling_config or {})
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
    validate_input_semantics(input_data)
    
    # Extract values from JSON
    n1 = input_data["subject_size"]
    n2 = input_data["object_size"]
    n3 = input_data["environment_size"]
    n4 = input_data["subject_attributes_count"]
    n5 = input_data["object_attributes_count"]
    n6 = input_data["environment_attributes_count"]

    global_stars = int(input_data.get("global_stars", 0))
    subject_values, subject_stars = parse_attribute_values_with_stars(
        input_data["subject_attributes_values"], global_stars=global_stars
    )
    object_values, object_stars = parse_attribute_values_with_stars(
        input_data["object_attributes_values"], global_stars=global_stars
    )
    environment_values, environment_stars = parse_attribute_values_with_stars(
        input_data["environment_attributes_values"], global_stars=global_stars
    )

    subject_attributes = {
        "values": subject_values,
        "stars": subject_stars,
        "distributions": input_data["subject_distributions"],
        "correlations": input_data.get("correlations", {}).get("subject", {}),
    }
    object_attributes = {
        "values": object_values,
        "stars": object_stars,
        "distributions": input_data["object_distributions"],
        "correlations": input_data.get("correlations", {}).get("object", {}),
    }
    environment_attributes = {
        "values": environment_values,
        "stars": environment_stars,
        "distributions": input_data["environment_distributions"],
        "correlations": input_data.get("correlations", {}).get("environment", {}),
    }

    permit_rules = input_data.get("permit_rules_count", 0)
    deny_rules = input_data.get("deny_rules_count", 0)
    seed = input_data.get("seed")
    sampling_config = input_data.get("sampling_config", {})

    # Write to config.ini
    write_config_file(
        n1, n2, n3, n4, n5, n6,
        subject_attributes, object_attributes, environment_attributes,
        permit_rules, deny_rules,
        seed=seed,
        sampling_config=sampling_config,
    )
