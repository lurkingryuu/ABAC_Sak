
import configparser
import json
import os
import random
import numpy as np
from math import factorial,exp
from scipy.stats import truncnorm
import gen_rules

ENV_CONFIG_INI = "ABAC_CONFIG_INI"
ENV_OUTPUT_DIR = "ABAC_OUTPUT_DIR"

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
                idx = np.random.choice(np.arange(n))
                entity_values[key].append(values[idx])

            else:
                raise ValueError(f"Unsupported distribution type: {dist['distribution']}")

    return entity_values


# Assign values to subjects, objects, and environments
n1 = int(config["NUMBERS"]["n1"])
n2 = int(config["NUMBERS"]["n2"])
n3 = int(config["NUMBERS"]["n3"])
SV = assign_values(SAV, subject_attributes["distributions"], n1, "SA", "S")
OV = assign_values(OAV, object_attributes["distributions"], n2, "OA", "O")
EV = assign_values(EAV, environment_attributes["distributions"], n3, "EA", "E")

# Generate users, objects, and environments
S = [f"S_{i}" for i in range(1, n1 + 1)]
O = [f"O_{i}" for i in range(1, n2 + 1)]
E = [f"E_{i}" for i in range(1, n3 + 1)]

# Generate attribute names
n4 = int(config["NUMBERS"]["n4"])
n5 = int(config["NUMBERS"]["n5"])
n6 = int(config["NUMBERS"]["n6"])
SA = [f"SA_{i}" for i in range(1, n4 + 1)]
OA = [f"OA_{i}" for i in range(1, n5 + 1)]
EA = [f"EA_{i}" for i in range(1, n6 + 1)]

# Create output.json
output_data = {
    "S": S,
    "O": O,
    "E": E,
    "SA": SA,
    "OA": OA,
    "EA": EA,
    "SAV": SAV,
    "OAV": OAV,
    "EAV": EAV,
    "SV": SV,
    "OV": OV,
    "EV": EV
}

# Generate rules and write to rules_temp.txt
N = int(config["RULES"]["rules_count"])
rules = gen_rules.generate_rules_2(N, n4, n5, n6, SV, OV, EV)

# Integrate generated rules into output.json as well
output_data["rules"] = rules

with open(os.path.join(OUTPUT_FOLDER, 'output.json'), 'w') as f:
    json.dump(output_data, f, indent=4)

with open(os.path.join(OUTPUT_FOLDER, "rules_temp.txt"), "w") as file:
    for rule in rules:
        file.write(rule + "\n")

print("Output generated successfully.")




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

def fill_matrix(A, SV, OV, EV, rules, n1, n2, n3):
    global no_of_ones
    for i in range(n1):
        for j in range(n2):
            for k in range(n3):
                # Access SV, OV, and EV using subject, object, and environment names
                SA1 = SV[f"S_{i + 1}"]
                OA1 = OV[f"O_{j + 1}"]
                EA1 = EV[f"E_{k + 1}"]
                
                # Check if any rule is satisfied
                A[i][j][k] = 1 if any(satisfies_rule(rule, SA1, OA1, EA1) for rule in rules) else 0
                no_of_ones += A[i][j][k]

fill_matrix(A, SV, OV, EV, rules, n1, n2, n3)
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

with open(os.path.join(OUTPUT_FOLDER, "access_data.txt"), "w") as file:
    for row in access_data:
        file.write(" ".join(map(str, row)) + "\n")