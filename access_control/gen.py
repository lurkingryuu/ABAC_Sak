
import configparser
import json
import os
import sys
import random
import numpy as np
from math import factorial,exp
import gen_rules
import healthcare_generator
import university_generator
try:
    from scipy.stats import truncnorm
except ImportError:
    # Fallback implementation of truncated normal using rejection sampling
    # if scipy is not available. This is useful in constrained environments.
    class TruncnormFallback:
        @staticmethod
        def rvs(a, b, loc=0, scale=1):
            # a and b are bounds in standard units: (low - loc) / scale, (high - loc) / scale
            low = loc + a * scale
            high = loc + b * scale
            while True:
                x = np.random.normal(loc, scale)
                if low <= x <= high:
                    return x
    truncnorm = TruncnormFallback()
import gen_rules

ENV_CONFIG_INI = "ABAC_CONFIG_INI"
ENV_OUTPUT_DIR = "ABAC_OUTPUT_DIR"
ENABLE_MEANINGFUL_NAMES = False

# config.ini: default is access_control/config.ini, but can be overridden per-job.
config_path = os.environ.get(ENV_CONFIG_INI) or os.path.join(
    os.path.dirname(__file__), 'config.ini'
)
if not os.path.exists(config_path):
    raise FileNotFoundError("config.ini not found. Ensure input.py runs successfully before gen.py.")

# Load the config file
config = configparser.ConfigParser()
config.read(config_path)

# Output directory: default is ../outputs, but can be overridden per-job.
OUTPUT_FOLDER = os.environ.get(ENV_OUTPUT_DIR) or os.path.join(
    os.path.dirname(__file__), '../outputs'
)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Parse attributes from config.ini
def parse_attributes(section):
    return {
        "count": int(config[section]["count"]),
        "values": list(map(int, config[section]["values"].split(','))),
        "distributions": json.loads(config[section]["distributions"])
    }

subject_attributes = parse_attributes("SUBJECT_ATTRIBUTES")
object_attributes = parse_attributes("OBJECT_ATTRIBUTES")
environment_attributes = parse_attributes("ENVIRONMENT_ATTRIBUTES")

# Generate attribute values (SAV, OAV, EAV)
def generate_attribute_values(attribute, prefix):
    n = attribute["values"]
    values = {}
    for i, num_values in enumerate(n, start=1):
        values[f"{prefix}_{i}"] = [f"{prefix}_{i}_{j}" for j in range(1, num_values + 1)]
    return values

SAV = generate_attribute_values(subject_attributes, "SA")
OAV = generate_attribute_values(object_attributes, "OA")
EAV = generate_attribute_values(environment_attributes, "EA")

# Sampling functions

# def sample_truncated_normal(mean=0.5, sigma=0.1, low=0.0, high=1.0):
#     """Sample a value from a truncated normal distribution."""
#     a, b = (low - mean) / sigma, (high - mean) / sigma  # bounds in standard units
#     return truncnorm.rvs(a, b, loc=mean, scale=sigma)

# def sample_poisson(lam, low=0.0, high=1.0):
#     """Sample a value from a Poisson distribution scaled to [low, high]."""
#     value = np.random.poisson(lam)
#     return min(max(low, value / lam), high)  # Scale to [low, high]

# # Assign attribute values to entities based on distributions
# def assign_values(attribute_values, distributions, entity_count, attribute_prefix, entity_prefix):
#     """
#     Assign values to entities based on distributions.
#     :param attribute_values: Dictionary of attribute values (e.g., SAV, OAV, EAV).
#     :param distributions: List of distributions for each attribute.
#     :param entity_count: Number of entities (e.g., subjects, objects, environments).
#     :param attribute_prefix: Prefix for attribute keys (e.g., "SA", "OA", "EA").
#     :param entity_prefix: Prefix for entity keys (e.g., "S", "O", "E").
#     :return: Dictionary of assigned values for each entity.
#     """
#     entity_values = {}
#     for entity in range(1, entity_count + 1):
#         entity_values[f"{entity_prefix}_{entity}"] = []
#         for i, dist in enumerate(distributions):
#             values = attribute_values[f"{attribute_prefix}_{i+1}"]
#             num_values = len(values)
#             ranges = np.linspace(0, 1, num_values + 1)  # Divide [0, 1] into ranges for each value

#             # Sample a number based on the distribution
#             if dist["distribution"] == "N":
#                 # mean = dist["mean"] / num_values  # Scale mean to [0, 1]
#                 mean = dist["mean"]   # Scale mean to [0, 1]
#                 # sigma = np.sqrt(dist["variance"]) / num_values  # Scale variance to [0, 1]
#                 sigma = np.sqrt(dist["variance"])  # Scale variance to [0, 1]
#                 sampled_value = sample_truncated_normal(mean=mean, sigma=sigma, low=0.0, high=1.0)
#             elif dist["distribution"] == "P":
#                 lam = dist["lambda"] / num_values  # Scale lambda to [0, 1]
#                 sampled_value = sample_poisson(lam=lam, low=0.0, high=1.0)
#             else:
#                 raise ValueError("Unsupported distribution type")

#             # Determine which range the sampled value falls into
#             for j in range(num_values):
#                 if ranges[j] <= sampled_value < ranges[j + 1]:
#                     entity_values[f"{entity_prefix}_{entity}"].append(values[j])
#                     break
#     return entity_values

def poisson_pmf(lam, k):
    """Return Poisson PMF P(K=k) for integer k >= 0."""
    return (lam**k) * exp(-lam) / factorial(k)

# def sample_truncated_normal(mean=0.5, sigma=0.1, low=0.0, high=1.0):
#     """Sample a value from a Normal(mean, sigma^2) truncated to [low, high]."""
#     a, b = (low - mean) / sigma, (high - mean) / sigma
#     return truncnorm.rvs(a, b, loc=mean, scale=sigma)

def sample_truncated_normal(mean, variance, low, high):
    """Sample a value from Normal(mean, variance) truncated to [low, high]."""
    sigma = np.sqrt(variance)
    a, b = (low - mean) / sigma, (high - mean) / sigma
    return truncnorm.rvs(a, b, loc=mean, scale=sigma)

# ---- main assignment routine ----
def assign_values(attribute_values, distributions, entity_count,
                  attribute_prefix, entity_prefix):
    """
    attribute_values: dict like {"SA_1": ["SA_1_1","SA_1_2",...], ...}
    distributions: list of dicts, one per attribute, e.g. [{"distribution":"N"}, {"distribution":"P"}, ...]
                   For Normal you may optionally include "mean" and "variance", but defaults are used if missing.
                   For Poisson we IGNORE any provided lambda and compute lambda=(1+n)/2 as requested.
    """
    entity_values = {}

    for entity in range(1, entity_count + 1):
        key = f"{entity_prefix}_{entity}"
        entity_values[key] = []

        for i, dist in enumerate(distributions):
            values = attribute_values[f"{attribute_prefix}_{i+1}"]
            n = len(values)

            if dist["distribution"] == "N":
                # use defaults unless user provided overrides
                mean = dist.get("mean", (n+1)/2.0)
                # mean=0.5
                # variance = dist.get("variance", 0.01)
                variance = dist.get("variance", (n/6.0)**2)

                # variance=0.01
                # sigma = np.sqrt(variance)
                x = sample_truncated_normal(mean=mean, variance=variance, low=0.0, high=float(n))

                # map continuous x in [0,1] to one of n equal bins [0,1/n),[1/n,2/n),...
                # map x ∈ [0,n] to bin k: choose a_k if x ∈ [k-1, k)
                # 0-based index: idx = floor(x), then clamp to [0, n-1]
                idx = int(x)
                if idx >= n:    # rare edge if x == 1.0
                    idx = n - 1
                entity_values[key].append(values[idx])

            elif dist["distribution"] == "P":
                # λ set to middle of 1..n 
                lam = dist["lambda"]

                # compute (truncated) weights for k = 1..n (attribute j corresponds to k=j+1)
                probs = np.array([poisson_pmf(lam, k=j+1) for j in range(n)], dtype=float)

                # normalize so they sum to 1 (this is the "scale up the prob sum to 1" step)
                probs /= probs.sum()

                # sample index according to these weights
                idx = np.random.choice(np.arange(n), p=probs)
                entity_values[key].append(values[idx])

            elif dist["distribution"] == "U":
                # Uniform distribution
                low_idx = dist.get("low", 0)
                high_idx = dist.get("high", n)
                
                # Ensure integer bounds and clamp
                low_idx = max(0, int(low_idx))
                high_idx = min(n, int(high_idx))
                
                if high_idx <= low_idx:
                    # Fallback to full range if invalid
                    low_idx = 0
                    high_idx = n
                
                idx = np.random.choice(np.arange(low_idx, high_idx))
                entity_values[key].append(values[idx])

            else:
                raise ValueError(f"Unsupported distribution type: {dist['distribution']}")

    return entity_values


# # Assign values to subjects, objects, and environments
# n1 = int(config["NUMBERS"]["n1"])
# n2 = int(config["NUMBERS"]["n2"])
# n3 = int(config["NUMBERS"]["n3"])
# SV = assign_values(SAV, subject_attributes["distributions"], n1, "SA", "S")
# OV = assign_values(OAV, object_attributes["distributions"], n2, "OA", "O")
# EV = assign_values(EAV, environment_attributes["distributions"], n3, "EA", "E")

# # Generate users, objects, and environments
# S = [f"S_{i}" for i in range(1, n1 + 1)]
# O = [f"O_{i}" for i in range(1, n2 + 1)]
# E = [f"E_{i}" for i in range(1, n3 + 1)]

# # Generate attribute names
# n4 = int(config["NUMBERS"]["n4"])
# n5 = int(config["NUMBERS"]["n5"])
# n6 = int(config["NUMBERS"]["n6"])
# SA = [f"SA_{i}" for i in range(1, n4 + 1)]
# OA = [f"OA_{i}" for i in range(1, n5 + 1)]
# EA = [f"EA_{i}" for i in range(1, n6 + 1)]

# # Create output.json
# output_data = {
#     "S": S,
#     "O": O,
#     "E": E,
#     "SA": SA,
#     "OA": OA,
#     "EA": EA,
#     "SAV": SAV,
#     "OAV": OAV,
#     "EAV": EAV,
#     "SV": SV,
#     "OV": OV,
#     "EV": EV
# }


# Assign values to subjects, objects, and environments
n1 = int(config["NUMBERS"]["n1"])
n2 = int(config["NUMBERS"]["n2"])
n3 = int(config["NUMBERS"]["n3"])
n4 = int(config["NUMBERS"]["n4"])
n5 = int(config["NUMBERS"]["n5"])
n6 = int(config["NUMBERS"]["n6"])

SV = assign_values(SAV, subject_attributes["distributions"], n1, "SA", "S")
OV = assign_values(OAV, object_attributes["distributions"], n2, "OA", "O")
EV = assign_values(EAV, environment_attributes["distributions"], n3, "EA", "E")

# Generate users, objects, and environments
S = [f"S_{i}" for i in range(1, n1 + 1)]
O = [f"O_{i}" for i in range(1, n2 + 1)]
E = [f"E_{i}" for i in range(1, n3 + 1)]

# Generate attribute names
SA = [f"SA_{i}" for i in range(1, n4 + 1)]
OA = [f"OA_{i}" for i in range(1, n5 + 1)]
EA = [f"EA_{i}" for i in range(1, n6 + 1)]

# ==================== CREATE OUTPUT.JSON ====================
# Build output_data in correct order:
# Step 1: S, O, E (entities)
# Step 2: SA, OA, EA (attribute names)
# Step 3: SAV, OAV, EAV (attribute values)
# Step 4: SV, OV, EV (entity-attribute assignments)

output_data = {}

# Step 1: Add base entities
output_data["S"] = S
output_data["O"] = O
output_data["E"] = E

# Step 2: Add attribute names
output_data["SA"] = SA
output_data["OA"] = OA
output_data["EA"] = EA

# Step 3: Add attribute values
output_data["SAV"] = SAV
output_data["OAV"] = OAV
output_data["EAV"] = EAV

# Step 4: Add entity-attribute assignments
output_data["SV"] = SV
output_data["OV"] = OV
output_data["EV"] = EV


#############################

# Generate accepted and denied rules
accepted_rules_count = int(config["RULES"]["accepted_rules_count"])
denied_rules_count = int(config["RULES"]["denied_rules_count"])

accepted_rules = gen_rules.generate_rules_2(accepted_rules_count, n4, n5, n6, SV, OV, EV) if accepted_rules_count > 0 else []
denied_rules = gen_rules.generate_rules_2(denied_rules_count, n4, n5, n6, SV, OV, EV) if denied_rules_count > 0 else []

# Integrate generated rules into output.json as well
output_data["accepted_rules"] = accepted_rules
output_data["denied_rules"] = denied_rules

with open(os.path.join(OUTPUT_FOLDER, 'output.json'), 'w') as f:
    json.dump(output_data, f, indent=4)

# with open(os.path.join(OUTPUT_FOLDER, "rules_temp.txt"), "w") as file:
#     for rule in rules:
#         file.write(rule + "\n")

print("Output generated successfully.")

if not ENABLE_MEANINGFUL_NAMES:
    # Generate only the indexed outputs and skip readable-name generation.
    A = [[[0] * n3 for _ in range(n2)] for _ in range(n1)]

    no_of_ones = 0

    def satisfies_rule(rule, SA1, OA1, EA1):
        rule_parts = rule.split(", ")
        for part in rule_parts:
            key, value = part.split(" = ")
            if key.startswith("SA_") and value not in SA1 and value != '*':
                return False
            if key.startswith("OA_") and value not in OA1 and value != '*':
                return False
            if key.startswith("EA_") and value not in EA1 and value != '*':
                return False
        return True

    def fill_matrix_with_rules(A, SV, OV, EV, accepted_rules, denied_rules, n1, n2, n3):
        global no_of_ones

        has_accepted = len(accepted_rules) > 0
        has_denied = len(denied_rules) > 0

        for i in range(n1):
            for j in range(n2):
                for k in range(n3):
                    SA1 = SV[f"S_{i + 1}"]
                    OA1 = OV[f"O_{j + 1}"]
                    EA1 = EV[f"E_{k + 1}"]

                    matches_accepted = any(satisfies_rule(rule, SA1, OA1, EA1) for rule in accepted_rules) if has_accepted else False
                    matches_denied = any(satisfies_rule(rule, SA1, OA1, EA1) for rule in denied_rules) if has_denied else False

                    if not has_accepted and not has_denied:
                        A[i][j][k] = 0
                    elif has_accepted and not has_denied:
                        A[i][j][k] = 1 if matches_accepted else 0
                    elif not has_accepted and has_denied:
                        A[i][j][k] = 0 if matches_denied else 1
                    else:
                        A[i][j][k] = 1 if matches_accepted else 0

                    no_of_ones += A[i][j][k]

    fill_matrix_with_rules(A, SV, OV, EV, accepted_rules, denied_rules, n1, n2, n3)
    print("No. of ones in ACM : ", no_of_ones)

    with open(os.path.join(OUTPUT_FOLDER, "ACM.txt"), "w") as file:
        for i in range(n1):
            for row in A[i]:
                file.write(" ".join(map(str, row)) + "\n")
            file.write("\n")

    def prepare_access_data(S, O, E, SV, OV, EV, A):
        access_data = []
        for i in range(len(S)):
            for j in range(len(O)):
                for k in range(len(E)):
                    subject = S[i]
                    obj = O[j]
                    env = E[k]
                    T = SV[subject] + OV[obj] + EV[env] + [A[i][j][k]]
                    access_data.append(T)
        return access_data

    access_data = prepare_access_data(S, O, E, SV, OV, EV, A)
    with open(os.path.join(OUTPUT_FOLDER, "access_data.txt"), "w") as file:
        for row in access_data:
            file.write(" ".join(map(str, row)) + "\n")

    print("Skipping meaningful-name generation and related output files.")
    print("✓ Regular access_data written to access_data.txt")
    print("\n" + "=" * 60)
    print("SUMMARY OF GENERATED FILES")
    print("=" * 60)
    print("✓ output.json - Regular ABAC data")
    print("✓ access_data.txt - Regular access control data")
    print("✓ ACM.txt - Access Control Matrix")
    sys.exit(0)



# ==================== HEALTHCARE DATA GENERATION ====================
print("\n" + "="*60)
print("HEALTHCARE DATA GENERATION WITH GEMINI")
print("="*60)



# Generate healthcare data from Gemini
healthcare_data = healthcare_generator.generate_healthcare_data_with_gemini(config_path)


# SAVE Gemini output to file
gemini_output_path = os.path.join(OUTPUT_FOLDER, 'healthcare_gemini_raw.json')
with open(gemini_output_path, 'w') as f:
    json.dump(healthcare_data, f, indent=4)

print(f"✓ Gemini raw output saved: {gemini_output_path}")



# Convert to indexed format using original SV, OV, EV indices (NO RESAMPLING)
print("\n" + "="*60)
print("CONVERTING TO INDEXED FORMAT")
print("="*60)

healthcare_output_format = healthcare_generator.convert_to_indexed_format(
    healthcare_data,
    SV,  # Original subject-attribute assignments
    OV,  # Original object-attribute assignments
    EV,  # Original environment-attribute assignments
    n1,  # subject_count
    n2,  # object_count
    n3   # environment_count
)

print(f"✓ Healthcare data converted to actual names/attributes format")
print(f"  - S_HC: {len(healthcare_output_format['S_HC'])} subjects with actual names")
print(f"  - O_HC: {len(healthcare_output_format['O_HC'])} objects with actual names")
print(f"  - E_HC: {len(healthcare_output_format['E_HC'])} environments with actual names")
print(f"  - SA_HC: {len(healthcare_output_format['SA_HC'])} subject attributes")
print(f"  - OA_HC: {len(healthcare_output_format['OA_HC'])} object attributes")
print(f"  - EA_HC: {len(healthcare_output_format['EA_HC'])} environment attributes")

# ==================== CONVERT RULES TO HEALTHCARE FORMAT ====================
print("\n" + "="*60)
print("CONVERTING RULES TO HEALTHCARE FORMAT")
print("="*60)

def convert_rules_to_healthcare(rules, subject_attrs, object_attrs, environment_attrs, 
                                subject_av, object_av, environment_av):
    """
    Convert indexed rules (SA_1 = SA_1_5) to healthcare format (employee_id = licensed).
    
    Mapping:
    - SA_1 → first subject attribute name
    - SA_1_5 → 5th value of first subject attribute
    - SA_1 = * → keep * but replace SA_1 with actual attribute name
    """
    healthcare_rules = []
    
    for rule in rules:
        rule_parts = rule.split(", ")
        healthcare_rule_parts = []
        
        for part in rule_parts:
            if "=" in part:
                key, value = part.split(" = ")
                
                # Parse the key (e.g., "SA_1" → attr_type="SA", attr_idx=1)
                key_parts = key.split("_")
                if len(key_parts) >= 2:
                    attr_type = key_parts[0]  # "SA", "OA", "EA"
                    try:
                        attr_idx = int(key_parts[1]) - 1  # Convert to 0-based
                    except:
                        healthcare_rule_parts.append(part)
                        continue
                    
                    # Get the actual attribute name
                    if attr_type == "SA" and 0 <= attr_idx < len(subject_attrs):
                        attr_name = subject_attrs[attr_idx]
                    elif attr_type == "OA" and 0 <= attr_idx < len(object_attrs):
                        attr_name = object_attrs[attr_idx]
                    elif attr_type == "EA" and 0 <= attr_idx < len(environment_attrs):
                        attr_name = environment_attrs[attr_idx]
                    else:
                        healthcare_rule_parts.append(part)
                        continue
                    
                    # Handle value
                    if value.strip() == "*":
                        # Keep * as is
                        healthcare_rule_parts.append(f"{attr_name} = *")
                    else:
                        # Parse value (e.g., "SA_1_5" → value_idx=5)
                        value_parts = value.split("_")
                        if len(value_parts) >= 3:
                            try:
                                value_idx = int(value_parts[-1]) - 1  # Convert to 0-based
                                
                                # Get the actual attribute value
                                if attr_type == "SA" and 0 <= attr_idx < len(subject_attrs):
                                    values = subject_av.get(attr_name, [])
                                elif attr_type == "OA" and 0 <= attr_idx < len(object_attrs):
                                    values = object_av.get(attr_name, [])
                                elif attr_type == "EA" and 0 <= attr_idx < len(environment_attrs):
                                    values = environment_av.get(attr_name, [])
                                else:
                                    values = []
                                
                                if 0 <= value_idx < len(values):
                                    actual_value = values[value_idx]
                                    healthcare_rule_parts.append(f"{attr_name} = {actual_value}")
                                else:
                                    healthcare_rule_parts.append(part)
                            except:
                                healthcare_rule_parts.append(part)
                        else:
                            healthcare_rule_parts.append(part)
                else:
                    healthcare_rule_parts.append(part)
            else:
                healthcare_rule_parts.append(part)
        
        healthcare_rules.append(", ".join(healthcare_rule_parts))
    
    return healthcare_rules

# Convert accepted and denied rules to healthcare format
accepted_healthcare_rules = convert_rules_to_healthcare(
    accepted_rules,
    healthcare_output_format["SA_HC"],
    healthcare_output_format["OA_HC"],
    healthcare_output_format["EA_HC"],
    healthcare_output_format["SAV_HC"],
    healthcare_output_format["OAV_HC"],
    healthcare_output_format["EAV_HC"]
)

denied_healthcare_rules = convert_rules_to_healthcare(
    denied_rules,
    healthcare_output_format["SA_HC"],
    healthcare_output_format["OA_HC"],
    healthcare_output_format["EA_HC"],
    healthcare_output_format["SAV_HC"],
    healthcare_output_format["OAV_HC"],
    healthcare_output_format["EAV_HC"]
)

print(f"✓ Accepted rules converted to healthcare format ({len(accepted_healthcare_rules)} rules)")
print(f"✓ Denied rules converted to healthcare format ({len(denied_healthcare_rules)} rules)")
print(f"  Example healthcare rules:")
for rule in accepted_healthcare_rules[:1]:
    print(f"    {rule[:100]}...")
for rule in denied_healthcare_rules[:1]:
    print(f"    {rule[:100]}...")

# ==================== CONVERT ACCESS_DATA TO HEALTHCARE FORMAT ====================
# print("\n" + "="*60)
# print("CONVERTING ACCESS_DATA TO HEALTHCARE FORMAT")
# print("="*60)

# Initialize the Access Control Matrix (ACM)
A = [[[0] * n3 for _ in range(n2)] for _ in range(n1)]

no_of_ones = 0
def satisfies_rule(rule, SA1, OA1, EA1):
    rule_parts = rule.split(", ")
    for part in rule_parts:
        key, value = part.split(" = ")
        if key.startswith("SA_") and value not in SA1 and value != '*':
            return False
        if key.startswith("OA_") and value not in OA1 and value != '*':
            return False
        if key.startswith("EA_") and value not in EA1 and value != '*':
            return False
    return True

def fill_matrix_with_rules(A, SV, OV, EV, accepted_rules, denied_rules, n1, n2, n3):
    """
    Fill ACM with precedence logic:
    - If both accepted_rules = 0 and denied_rules = 0: everything is denied (acm = 0)
    - If accepted_rules > 0 and denied_rules = 0: current behavior (acm = 1 if matches accepted, else 0)
    - If accepted_rules = 0 and denied_rules > 0: acm = 0 if matches denied, else 1
    - If both > 0: accepted rules take precedence (if matches both, acm = 1; if matches neither, acm = 0)
    """
    global no_of_ones
    
    has_accepted = len(accepted_rules) > 0
    has_denied = len(denied_rules) > 0
    
    for i in range(n1):
        for j in range(n2):
            for k in range(n3):
                SA1 = SV[f"S_{i + 1}"]
                OA1 = OV[f"O_{j + 1}"]
                EA1 = EV[f"E_{k + 1}"]
                
                matches_accepted = any(satisfies_rule(rule, SA1, OA1, EA1) for rule in accepted_rules) if has_accepted else False
                matches_denied = any(satisfies_rule(rule, SA1, OA1, EA1) for rule in denied_rules) if has_denied else False
                
                # Apply precedence logic
                if not has_accepted and not has_denied:
                    # Case 1: Both 0 - everything denied
                    A[i][j][k] = 0
                elif has_accepted and not has_denied:
                    # Case 2: Only accepted rules - current behavior
                    A[i][j][k] = 1 if matches_accepted else 0
                elif not has_accepted and has_denied:
                    # Case 3: Only denied rules - inverse logic
                    A[i][j][k] = 0 if matches_denied else 1
                else:
                    # Case 4: Both accepted and denied - accepted takes precedence
                    # Accepted matches: 1, else deny: 0
                    A[i][j][k] = 1 if matches_accepted else 0
                
                no_of_ones += A[i][j][k]

fill_matrix_with_rules(A, SV, OV, EV, accepted_rules, denied_rules, n1, n2, n3)
print("No. of ones in ACM : ", no_of_ones)

# Write ACM to ACM.txt
with open(os.path.join(OUTPUT_FOLDER, "ACM.txt"), "w") as file:
    for i in range(n1):
        for row in A[i]:
            file.write(" ".join(map(str, row)) + "\n")
        file.write("\n")

# Prepare access_data.txt
def prepare_access_data(S, O, E, SV, OV, EV, A):
    access_data = []
    for i in range(len(S)):
        for j in range(len(O)):
            for k in range(len(E)):
                # Access SV, OV, and EV using subject, object, and environment names
                subject = S[i]
                obj = O[j]
                env = E[k]
                T = SV[subject] + OV[obj] + EV[env] + [A[i][j][k]]  # Concatenate attributes and access decision
                access_data.append(T)
    return access_data

access_data = prepare_access_data(S, O, E, SV, OV, EV, A)







def prepare_access_data_healthcare(S, O, E, SV_HC, OV_HC, EV_HC, A):
    """
    Prepare healthcare access_data by converting indexed attribute values to actual values.
    
    Input row example:  SA_1_4 SA_2_2 SA_3_1 ... EA_7_1 0
    Output row example: physician medical_assistant cardiology ... physical_security_level_value 0
    """
    access_data_hc = []
    
    for i in range(len(S)):
        subject = S[i]
        for j in range(len(O)):
            obj = O[j]
            for k in range(len(E)):
                env = E[k]
                
                # Get healthcare attribute values (actual values, not indices)
                subject_values = SV_HC.get(subject, [])
                object_values = OV_HC.get(obj, [])
                environment_values = EV_HC.get(env, [])
                
                # Concatenate all attribute values with access decision
                T = subject_values + object_values + environment_values + [A[i][j][k]]
                access_data_hc.append(T)
    
    return access_data_hc

# ==================== APPEND HEALTHCARE DATA TO OUTPUT.JSON ====================
# print("\n" + "="*60)
# print("APPENDING HEALTHCARE DATA TO OUTPUT.JSON")
# print("="*60)

# Convert access_data to healthcare format (after A is filled)
print("\nConverting access_data to healthcare format...")
access_data_hc = prepare_access_data_healthcare(
    healthcare_output_format["S_HC"],
    healthcare_output_format["O_HC"],
    healthcare_output_format["E_HC"],
    healthcare_output_format["SV_HC"],
    healthcare_output_format["OV_HC"],
    healthcare_output_format["EV_HC"],
    A
)

print(f"✓ Access data converted to healthcare format ({len(access_data_hc)} records)")
print(f"  Example healthcare access_data records:")
for record in access_data_hc[:2]:
    print(f"    {record[:5]} ... {record[-1]}")

# Write healthcare access_data to separate file
with open(os.path.join(OUTPUT_FOLDER, "access_data_healthcare.txt"), "w") as file:
    for row in access_data_hc:
        row_str = [str(val) for val in row]
        file.write(" ".join(row_str) + "\n")

print(f"✓ Healthcare access_data written to access_data_healthcare.txt")

# Append all healthcare data using same nomenclature
output_data_hc = {}
output_data_hc["S_HC"] = healthcare_output_format["S_HC"]
output_data_hc["O_HC"] = healthcare_output_format["O_HC"]
output_data_hc["E_HC"] = healthcare_output_format["E_HC"]

output_data_hc["SA_HC"] = healthcare_output_format["SA_HC"]
output_data_hc["OA_HC"] = healthcare_output_format["OA_HC"]
output_data_hc["EA_HC"] = healthcare_output_format["EA_HC"]

output_data_hc["SAV_HC"] = healthcare_output_format["SAV_HC"]
output_data_hc["OAV_HC"] = healthcare_output_format["OAV_HC"]
output_data_hc["EAV_HC"] = healthcare_output_format["EAV_HC"]

output_data_hc["SV_HC"] = healthcare_output_format["SV_HC"]
output_data_hc["OV_HC"] = healthcare_output_format["OV_HC"]
output_data_hc["EV_HC"] = healthcare_output_format["EV_HC"]

# Append healthcare rules (both accepted and denied)
output_data_hc["accepted_rules_HC"] = accepted_healthcare_rules
output_data_hc["denied_rules_HC"] = denied_healthcare_rules

print(f"✓ Healthcare data prepared successfully")
print(f"  Example SV_HC mapping (actual names with attribute values):")
for key in list(output_data_hc["SV_HC"].items())[:2]:
    print(f"    {key[0]}: {key[1]}")

# Write healthcare output.json
with open(os.path.join(OUTPUT_FOLDER, 'output_healthcare.json'), 'w') as f:
    json.dump(output_data_hc, f, indent=4)

print(f"✓ Healthcare output.json written successfully")









# # Initialize the Access Control Matrix (ACM)
# A = [[[0] * n3 for _ in range(n2)] for _ in range(n1)]

# no_of_ones = 0
# def satisfies_rule(rule, SA1, OA1, EA1):
#     rule_parts = rule.split(", ")
#     for part in rule_parts:
#         key, value = part.split(" = ")
#         if key.startswith("SA_") and value not in SA1 and value != '*':
#             return False
#         if key.startswith("OA_") and value not in OA1 and value != '*':
#             return False
#         if key.startswith("EA_") and value not in EA1 and value != '*':
#             return False
#     return True

# def fill_matrix(A, SV, OV, EV, rules, n1, n2, n3):
#     global no_of_ones
#     for i in range(n1):
#         for j in range(n2):
#             for k in range(n3):
#                 # Access SV, OV, and EV using subject, object, and environment names
#                 SA1 = SV[f"S_{i + 1}"]
#                 OA1 = OV[f"O_{j + 1}"]
#                 EA1 = EV[f"E_{k + 1}"]
                
#                 # Check if any rule is satisfied
#                 A[i][j][k] = 1 if any(satisfies_rule(rule, SA1, OA1, EA1) for rule in rules) else 0
#                 no_of_ones += A[i][j][k]

# fill_matrix(A, SV, OV, EV, rules, n1, n2, n3)
# print("No. of ones in ACM : ", no_of_ones)

# # Write ACM to ACM.txt
# with open(os.path.join(OUTPUT_FOLDER, "ACM.txt"), "w") as file:
#     for i in range(n1):
#         for row in A[i]:
#             file.write(" ".join(map(str, row)) + "\n")
#         file.write("\n")

# # Prepare access_data.txt
# def prepare_access_data(S, O, E, SV, OV, EV, A):
#     access_data = []
#     for i in range(len(S)):
#         for j in range(len(O)):
#             for k in range(len(E)):
#                 # Access SV, OV, and EV using subject, object, and environment names
#                 subject = S[i]
#                 obj = O[j]
#                 env = E[k]
#                 T = SV[subject] + OV[obj] + EV[env] + [A[i][j][k]]  # Concatenate attributes and access decision
#                 access_data.append(T)
#     return access_data

# access_data = prepare_access_data(S, O, E, SV, OV, EV, A)

# ==================== GENERATE UNIVERSITY DATA ====================
print("\n" + "="*60)
print("GENERATING UNIVERSITY DATA")
print("="*60)

uni_data = university_generator.generate_university_data_with_gemini(config_path)
university_output_format = university_generator.convert_to_university_format(
    uni_data, SV, OV, EV, n1, n2, n3
)

print(f"✓ University data generated and converted")
print(f"  Subjects: {len(university_output_format['S_UNI'])} ({', '.join(university_output_format['S_UNI'][:2])}...)")
print(f"  Objects: {len(university_output_format['O_UNI'])} ({', '.join(university_output_format['O_UNI'][:2])}...)")
print(f"  Environments: {len(university_output_format['E_UNI'])} ({', '.join(university_output_format['E_UNI'][:2])}...)")

# Convert rules to university format
def convert_rules_to_university(rules, subject_attrs, object_attrs, environment_attrs,
                               subject_av, object_av, environment_av):
    """
    Convert indexed rules (e.g., "SA_1 = SA_1_5") to university readable rules
    using actual attribute names and values.
    """
    university_rules = []
    for rule in rules:
        rule_parts = rule.split(", ")
        university_rule_parts = []
        
        for part in rule_parts:
            if " = " in part:
                key, value = part.split(" = ")
                attr_type = key.split("_")[0]
                
                try:
                    attr_idx = int(key.split("_")[1]) - 1
                except:
                    university_rule_parts.append(part)
                    continue
                
                # Get the actual attribute name
                if attr_type == "SA" and 0 <= attr_idx < len(subject_attrs):
                    attr_name = subject_attrs[attr_idx]
                elif attr_type == "OA" and 0 <= attr_idx < len(object_attrs):
                    attr_name = object_attrs[attr_idx]
                elif attr_type == "EA" and 0 <= attr_idx < len(environment_attrs):
                    attr_name = environment_attrs[attr_idx]
                else:
                    university_rule_parts.append(part)
                    continue
                
                # Handle value
                if value.strip() == "*":
                    university_rule_parts.append(f"{attr_name} = *")
                else:
                    value_parts = value.split("_")
                    if len(value_parts) >= 3:
                        try:
                            value_idx = int(value_parts[-1]) - 1
                            
                            if attr_type == "SA" and 0 <= attr_idx < len(subject_attrs):
                                values = subject_av.get(attr_name, [])
                            elif attr_type == "OA" and 0 <= attr_idx < len(object_attrs):
                                values = object_av.get(attr_name, [])
                            elif attr_type == "EA" and 0 <= attr_idx < len(environment_attrs):
                                values = environment_av.get(attr_name, [])
                            else:
                                values = []
                            
                            if 0 <= value_idx < len(values):
                                actual_value = values[value_idx]
                                university_rule_parts.append(f"{attr_name} = {actual_value}")
                            else:
                                university_rule_parts.append(part)
                        except:
                            university_rule_parts.append(part)
                    else:
                        university_rule_parts.append(part)
            else:
                university_rule_parts.append(part)
        
        university_rules.append(", ".join(university_rule_parts))
    
    return university_rules

university_rules = convert_rules_to_university(
    accepted_rules,
    university_output_format["SA_UNI"],
    university_output_format["OA_UNI"],
    university_output_format["EA_UNI"],
    university_output_format["SAV_UNI"],
    university_output_format["OAV_UNI"],
    university_output_format["EAV_UNI"]
)

denied_university_rules = convert_rules_to_university(
    denied_rules,
    university_output_format["SA_UNI"],
    university_output_format["OA_UNI"],
    university_output_format["EA_UNI"],
    university_output_format["SAV_UNI"],
    university_output_format["OAV_UNI"],
    university_output_format["EAV_UNI"]
)

print(f"✓ Accepted rules converted to university format ({len(university_rules)} rules)")
print(f"✓ Denied rules converted to university format ({len(denied_university_rules)} rules)")
print(f"  Example university rules:")
for rule in university_rules[:1]:
    print(f"    {rule[:100]}...")
for rule in denied_university_rules[:1]:
    print(f"    {rule[:100]}...")

# Convert access_data to university format
def prepare_access_data_university(S, O, E, SV_UNI, OV_UNI, EV_UNI, A):
    """
    Prepare university access_data using actual attribute values.
    """
    access_data_uni = []
    
    for i in range(len(S)):
        subject = S[i]
        for j in range(len(O)):
            obj = O[j]
            for k in range(len(E)):
                env = E[k]
                
                subject_values = SV_UNI.get(subject, [])
                object_values = OV_UNI.get(obj, [])
                environment_values = EV_UNI.get(env, [])
                
                T = subject_values + object_values + environment_values + [A[i][j][k]]
                access_data_uni.append(T)
    
    return access_data_uni

access_data_uni = prepare_access_data_university(
    university_output_format["S_UNI"],
    university_output_format["O_UNI"],
    university_output_format["E_UNI"],
    university_output_format["SV_UNI"],
    university_output_format["OV_UNI"],
    university_output_format["EV_UNI"],
    A
)

print(f"✓ Access data converted to university format ({len(access_data_uni)} records)")
print(f"  Example university access_data records:")
for record in access_data_uni[:2]:
    print(f"    {record[:5]} ... {record[-1]}")

# Write university access_data to separate file
with open(os.path.join(OUTPUT_FOLDER, "access_data_university.txt"), "w") as file:
    for row in access_data_uni:
        row_str = [str(val) for val in row]
        file.write(" ".join(row_str) + "\n")

print(f"✓ University access_data written to access_data_university.txt")

# Append all university data to separate output.json
output_data_uni = {}
output_data_uni["S_UNI"] = university_output_format["S_UNI"]
output_data_uni["O_UNI"] = university_output_format["O_UNI"]
output_data_uni["E_UNI"] = university_output_format["E_UNI"]

output_data_uni["SA_UNI"] = university_output_format["SA_UNI"]
output_data_uni["OA_UNI"] = university_output_format["OA_UNI"]
output_data_uni["EA_UNI"] = university_output_format["EA_UNI"]

output_data_uni["SAV_UNI"] = university_output_format["SAV_UNI"]
output_data_uni["OAV_UNI"] = university_output_format["OAV_UNI"]
output_data_uni["EAV_UNI"] = university_output_format["EAV_UNI"]

output_data_uni["SV_UNI"] = university_output_format["SV_UNI"]
output_data_uni["OV_UNI"] = university_output_format["OV_UNI"]
output_data_uni["EV_UNI"] = university_output_format["EV_UNI"]

# Append university rules (both accepted and denied)
output_data_uni["accepted_rules_UNI"] = university_rules
output_data_uni["denied_rules_UNI"] = denied_university_rules

print(f"✓ University data prepared successfully")
print(f"  Example SV_UNI mapping (actual names with attribute values):")
for key in list(output_data_uni["SV_UNI"].items())[:2]:
    print(f"    {key[0]}: {key[1]}")

# Write university output.json
with open(os.path.join(OUTPUT_FOLDER, 'output_university.json'), 'w') as f:
    json.dump(output_data_uni, f, indent=4)

print(f"✓ University output.json written successfully")

# Write regular access_data to access_data.txt
with open(os.path.join(OUTPUT_FOLDER, "access_data.txt"), "w") as file:
    for row in access_data:
        file.write(" ".join(map(str, row)) + "\n")

print(f"✓ Regular access_data written to access_data.txt")
print(f"\n" + "="*60)
print("SUMMARY OF GENERATED FILES")
print("="*60)
print(f"✓ output.json - Regular ABAC data")
print(f"✓ output_healthcare.json - Healthcare domain data")
print(f"✓ output_university.json - University domain data")
print(f"✓ access_data.txt - Regular access control data")
print(f"✓ access_data_healthcare.txt - Healthcare access control data")
print(f"✓ access_data_university.txt - University access control data")
print(f"✓ ACM.txt - Access Control Matrix")
