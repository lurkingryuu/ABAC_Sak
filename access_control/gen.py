
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
    correlations_raw = config[section].get("correlations", "{}")
    count = int(config[section]["count"])
    stars_raw = config[section].get("stars", "")
    if stars_raw.strip():
        stars = list(map(int, stars_raw.split(",")))
    else:
        stars = [0] * count
    # Defensive: ensure stars list length matches count.
    if len(stars) != count:
        stars = (stars + ([0] * count))[:count]
    return {
        "count": count,
        "values": list(map(int, config[section]["values"].split(','))),
        "stars": stars,
        "distributions": json.loads(config[section]["distributions"]),
        "correlations": json.loads(correlations_raw) if correlations_raw else {},
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

def _normalize_prob_vector(probs):
    probs = np.asarray(probs, dtype=float)
    probs[probs < 0] = 0
    total = probs.sum()
    if total <= 0:
        return np.ones_like(probs, dtype=float) / len(probs)
    return probs / total


def _normal_bin_probabilities(n, dist):
    mean = float(dist.get("mean", (n + 1) / 2.0))
    variance = float(dist.get("variance", (n / 6.0) ** 2))
    sigma = max(np.sqrt(variance), 1e-9)
    a = (0.0 - mean) / sigma
    b = (float(n) - mean) / sigma
    if hasattr(truncnorm, "cdf"):
        cdf_edges = np.array([
            truncnorm.cdf(float(i), a, b, loc=mean, scale=sigma) for i in range(n + 1)
        ])
        probs = np.diff(cdf_edges)
        return _normalize_prob_vector(probs)
    # Fallback for environments without scipy.stats truncnorm.cdf.
    draws = np.array([sample_truncated_normal(mean, variance, 0.0, float(n)) for _ in range(2000)])
    bins = np.clip(draws.astype(int), 0, n - 1)
    counts = np.bincount(bins, minlength=n)
    return _normalize_prob_vector(counts)


def _build_base_sampler(distributions, attribute_values, attribute_prefix):
    samplers = []
    marginals = []
    for i, dist in enumerate(distributions):
        n = len(attribute_values[f"{attribute_prefix}_{i+1}"])
        dtype = dist["distribution"]
        if dtype == "N":
            probs = _normal_bin_probabilities(n, dist)
        elif dtype == "P":
            lam = float(dist["lambda"])
            probs = _normalize_prob_vector([poisson_pmf(lam, k=j + 1) for j in range(n)])
        elif dtype == "U":
            low_idx = max(0, int(dist.get("low", 0)))
            high_idx = min(n, int(dist.get("high", n)))
            if high_idx <= low_idx:
                low_idx, high_idx = 0, n
            probs = np.zeros(n, dtype=float)
            probs[low_idx:high_idx] = 1.0
            probs = _normalize_prob_vector(probs)
        else:
            raise ValueError(f"Unsupported distribution type: {dtype}")
        marginals.append(probs)
        samplers.append(np.arange(n))
    return samplers, marginals


def _sample_from_probs(probs):
    return np.random.choice(np.arange(len(probs)), p=probs)


def _sample_rows_from_prob_matrix(prob_matrix):
    cdf = np.cumsum(prob_matrix, axis=1)
    r = np.random.rand(prob_matrix.shape[0], 1)
    return (cdf < r).sum(axis=1)


def _normalize_joint_target(pair, cardinality_a, cardinality_b, marginal_a, marginal_b):
    target = pair.get("target", {})
    if "joint_table" in target:
        table = np.array(target["joint_table"], dtype=float)
        if table.shape != (cardinality_a, cardinality_b):
            raise ValueError("Correlation joint_table dimensions do not match attribute domains")
        return _normalize_prob_vector(table.reshape(-1)).reshape(cardinality_a, cardinality_b)
    if "cramers_v" in target:
        strength = float(target.get("cramers_v", 0.0))
        base = np.outer(marginal_a, marginal_b)
        diagonal = np.zeros_like(base)
        diag_len = min(cardinality_a, cardinality_b)
        for d in range(diag_len):
            diagonal[d, d] = 1.0 / diag_len
        table = (1.0 - strength) * base + strength * diagonal
        return _normalize_prob_vector(table.reshape(-1)).reshape(cardinality_a, cardinality_b)
    raise ValueError("Each correlation pair target must define joint_table or cramers_v")


def _js_divergence(p, q):
    p = _normalize_prob_vector(p)
    q = _normalize_prob_vector(q)
    m = 0.5 * (p + q)
    def _kl(a, b):
        eps = 1e-12
        mask = a > 0
        return np.sum(a[mask] * np.log((a[mask] + eps) / (b[mask] + eps)))
    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def _compute_pair_l1(samples, idx_a, idx_b, card_a, card_b, target_joint):
    observed = np.zeros((card_a, card_b), dtype=float)
    np.add.at(observed, (samples[:, idx_a], samples[:, idx_b]), 1.0)
    observed /= max(len(samples), 1)
    return float(np.abs(observed - target_joint).sum())


def _compute_metrics(samples, base_marginals, pair_specs):
    marginal_error = 0.0
    for idx, base_prob in enumerate(base_marginals):
        observed = np.bincount(samples[:, idx], minlength=len(base_prob))
        observed = observed / max(len(samples), 1)
        marginal_error += _js_divergence(observed, base_prob)
    pair_error = 0.0
    for spec in pair_specs:
        pair_error += _compute_pair_l1(
            samples,
            spec["idx_a"],
            spec["idx_b"],
            spec["card_a"],
            spec["card_b"],
            spec["target_joint"],
        )
    if pair_specs:
        pair_error /= len(pair_specs)
    if base_marginals:
        marginal_error /= len(base_marginals)
    return marginal_error, pair_error


def assign_values(attribute_values, distributions, entity_count, attribute_prefix, entity_prefix, correlations=None, sampling_config=None):
    correlations = correlations or {}
    sampling_config = sampling_config or {}
    _, base_marginals = _build_base_sampler(distributions, attribute_values, attribute_prefix)
    attr_count = len(base_marginals)
    samples = np.zeros((entity_count, attr_count), dtype=int)

    # Independent initialization (vectorized).
    for idx, probs in enumerate(base_marginals):
        samples[:, idx] = np.random.choice(np.arange(len(probs)), size=entity_count, p=probs)

    # Pairwise soft calibration (hybrid phase-1 implementation).
    pair_specs = []
    for pair in correlations.get("pairs", []):
        idx_a = int(pair["attr_a"]) - 1
        idx_b = int(pair["attr_b"]) - 1
        if idx_a < 0 or idx_b < 0 or idx_a >= attr_count or idx_b >= attr_count or idx_a == idx_b:
            continue
        card_a = len(base_marginals[idx_a])
        card_b = len(base_marginals[idx_b])
        target_joint = _normalize_joint_target(
            pair,
            card_a,
            card_b,
            base_marginals[idx_a],
            base_marginals[idx_b],
        )
        pair_specs.append({
            "idx_a": idx_a,
            "idx_b": idx_b,
            "card_a": card_a,
            "card_b": card_b,
            "target_joint": target_joint,
            "weight": float(pair.get("weight", 1.0)),
        })

    alpha = float(sampling_config.get("alpha", 0.8))
    beta = float(sampling_config.get("beta", 0.2))
    max_iters = int(sampling_config.get("max_calibration_iters", 25))
    marginal_tolerance = float(sampling_config.get("marginal_tolerance", 0.02))
    pair_tolerance = float(sampling_config.get("pair_tolerance", 0.15))

    diagnostics = {
        "mode": "independent",
        "iterations": 0,
        "marginal_error": 0.0,
        "pair_error": 0.0,
        "marginal_tolerance": marginal_tolerance,
        "pair_tolerance": pair_tolerance,
    }
    if pair_specs and entity_count > 0:
        diagnostics["mode"] = "pairwise_soft_calibrated"
        marginal_mix = alpha / max(alpha + beta, 1e-9)
        correlation_mix = 1.0 - marginal_mix
        for iteration in range(max_iters):
            for spec in pair_specs:
                idx_a = spec["idx_a"]
                idx_b = spec["idx_b"]
                target_joint = spec["target_joint"]
                cond_b_given_a = np.zeros_like(target_joint)
                row_sums = target_joint.sum(axis=1, keepdims=True)
                np.divide(target_joint, row_sums, out=cond_b_given_a, where=row_sums > 0)
                per_row = cond_b_given_a[samples[:, idx_a]]
                blended = correlation_mix * per_row + marginal_mix * base_marginals[idx_b][None, :]
                blended = np.apply_along_axis(_normalize_prob_vector, 1, blended)
                samples[:, idx_b] = _sample_rows_from_prob_matrix(blended)

                # Symmetric update for A given B to avoid directional bias.
                cond_a_given_b = np.zeros_like(target_joint.T)
                col_sums = target_joint.sum(axis=0, keepdims=True)
                np.divide(target_joint.T, col_sums, out=cond_a_given_b, where=col_sums > 0)
                per_row_a = cond_a_given_b[samples[:, idx_b]]
                blended_a = correlation_mix * per_row_a + marginal_mix * base_marginals[idx_a][None, :]
                blended_a = np.apply_along_axis(_normalize_prob_vector, 1, blended_a)
                samples[:, idx_a] = _sample_rows_from_prob_matrix(blended_a)
            diagnostics["iterations"] = iteration + 1
            m_err, p_err = _compute_metrics(samples, base_marginals, pair_specs)
            diagnostics["marginal_error"] = round(float(m_err), 6)
            diagnostics["pair_error"] = round(float(p_err), 6)
            if m_err <= marginal_tolerance and p_err <= pair_tolerance:
                break
    else:
        m_err, p_err = _compute_metrics(samples, base_marginals, [])
        diagnostics["marginal_error"] = round(float(m_err), 6)
        diagnostics["pair_error"] = round(float(p_err), 6)

    entity_values = {}
    for entity in range(entity_count):
        key = f"{entity_prefix}_{entity + 1}"
        row = []
        for i in range(attr_count):
            values = attribute_values[f"{attribute_prefix}_{i+1}"]
            row.append(values[int(samples[entity, i])])
        entity_values[key] = row
    return entity_values, diagnostics


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
seed_raw = config["NUMBERS"].get("seed", "").strip()
if seed_raw:
    seed = int(seed_raw)
    random.seed(seed)
    np.random.seed(seed)
else:
    seed = None
sampling_config = json.loads(config.get("SAMPLING", "config", fallback="{}"))

SV, subject_diag = assign_values(
    SAV, subject_attributes["distributions"], n1, "SA", "S",
    correlations=subject_attributes.get("correlations", {}),
    sampling_config=sampling_config,
)
OV, object_diag = assign_values(
    OAV, object_attributes["distributions"], n2, "OA", "O",
    correlations=object_attributes.get("correlations", {}),
    sampling_config=sampling_config,
)
EV, environment_diag = assign_values(
    EAV, environment_attributes["distributions"], n3, "EA", "E",
    correlations=environment_attributes.get("correlations", {}),
    sampling_config=sampling_config,
)

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
output_data["generation_diagnostics"] = {
    "seed": seed,
    "sampling_config": sampling_config,
    "subject": subject_diag,
    "object": object_diag,
    "environment": environment_diag,
}


#############################

# Generate permit and deny rules
permit_rules_count = int(config["RULES"]["permit_rules_count"])
deny_rules_count = int(config["RULES"]["deny_rules_count"])

permit_rules = gen_rules.generate_rules_2(
    permit_rules_count,
    n4,
    n5,
    n6,
    SV,
    OV,
    EV,
    subject_stars=subject_attributes.get("stars"),
    object_stars=object_attributes.get("stars"),
    environment_stars=environment_attributes.get("stars"),
) if permit_rules_count > 0 else []

deny_rules = gen_rules.generate_rules_2(
    deny_rules_count,
    n4,
    n5,
    n6,
    SV,
    OV,
    EV,
    subject_stars=subject_attributes.get("stars"),
    object_stars=object_attributes.get("stars"),
    environment_stars=environment_attributes.get("stars"),
) if deny_rules_count > 0 else []

# Integrate generated rules into output.json as well
output_data["permit_rules"] = permit_rules
output_data["deny_rules"] = deny_rules

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

    def fill_matrix_with_rules(A, SV, OV, EV, permit_rules, deny_rules, n1, n2, n3):
        global no_of_ones

        has_permit = len(permit_rules) > 0
        has_deny = len(deny_rules) > 0

        for i in range(n1):
            for j in range(n2):
                for k in range(n3):
                    SA1 = SV[f"S_{i + 1}"]
                    OA1 = OV[f"O_{j + 1}"]
                    EA1 = EV[f"E_{k + 1}"]

                    matches_permit = any(satisfies_rule(rule, SA1, OA1, EA1) for rule in permit_rules) if has_permit else False
                    matches_deny = any(satisfies_rule(rule, SA1, OA1, EA1) for rule in deny_rules) if has_deny else False

                    if not has_permit and not has_deny:
                        A[i][j][k] = 0
                    elif has_permit and not has_deny:
                        A[i][j][k] = 1 if matches_permit else 0
                    elif not has_permit and has_deny:
                        A[i][j][k] = 0 if matches_deny else 1
                    else:
                        A[i][j][k] = 1 if (matches_permit and not matches_deny) else 0

                    no_of_ones += A[i][j][k]

    fill_matrix_with_rules(A, SV, OV, EV, permit_rules, deny_rules, n1, n2, n3)
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

def convert_rules_to_domain_format(rules, subject_attrs, object_attrs, environment_attrs,
                                   subject_av, object_av, environment_av):
    """
    Convert indexed rules (SA_1 = SA_1_5) to domain format (role = physician).
    
    Mapping:
    - SA_1 = SA_1_5: Use subject_attrs[0] as key, subject_av[subject_attrs[0]][4] as value
    - SA_1 = *: Keep wildcard, use subject_attrs[0] as key
    - Maintains order of attributes in rule
    
    Args:
        rules: List of rules with indexed notation (SA_1 = SA_1_5, ...)
        subject_attrs: List of subject attribute names
        object_attrs: List of object attribute names
        environment_attrs: List of environment attribute names
        subject_av: Dict[attr_name] -> List[values]
        object_av: Dict[attr_name] -> List[values]
        environment_av: Dict[attr_name] -> List[values]
    
    Returns:
        List of converted rules with actual attribute names and values
    """
    converted_rules = []
    
    for rule in rules:
        rule_parts = rule.split(", ")
        converted_parts = []
        
        for part in rule_parts:
            if "=" not in part:
                continue
            
            key, value = part.split(" = ", 1)
            key = key.strip()
            value = value.strip()
            
            # Parse the key (e.g., "SA_1" → attr_type="SA", attr_idx=1)
            key_parts = key.split("_")
            if len(key_parts) < 2:
                continue
            
            attr_type = key_parts[0]  # "SA", "OA", "EA"
            try:
                attr_idx = int(key_parts[1]) - 1  # Convert to 0-based indexing
            except (ValueError, IndexError):
                continue
            
            # Determine attribute list and value mapping
            if attr_type == "SA":
                attrs = subject_attrs
                av = subject_av
            elif attr_type == "OA":
                attrs = object_attrs
                av = object_av
            elif attr_type == "EA":
                attrs = environment_attrs
                av = environment_av
            else:
                continue
            
            # Validate attribute index
            if not (0 <= attr_idx < len(attrs)):
                continue
            
            attr_name = attrs[attr_idx]
            
            # Handle wildcard
            if value == "*":
                converted_parts.append(f"{attr_name} = *")
            else:
                # Parse value (e.g., "SA_1_5" → last part is value index)
                value_parts = value.split("_")
                if len(value_parts) < 3:
                    continue
                
                try:
                    value_idx = int(value_parts[-1]) - 1  # Convert to 0-based indexing
                except (ValueError, IndexError):
                    continue
                
                # Get the attribute values list for this attribute
                values = av.get(attr_name, [])
                
                # Validate value index
                if not (0 <= value_idx < len(values)):
                    continue
                
                actual_value = values[value_idx]
                converted_parts.append(f"{attr_name} = {actual_value}")
        
        if converted_parts:
            converted_rules.append(", ".join(converted_parts))
    
    return converted_rules

# Convert permit and deny rules to healthcare format
permit_healthcare_rules = convert_rules_to_domain_format(
    permit_rules,
    healthcare_output_format["SA_HC"],
    healthcare_output_format["OA_HC"],
    healthcare_output_format["EA_HC"],
    healthcare_output_format["SAV_HC"],
    healthcare_output_format["OAV_HC"],
    healthcare_output_format["EAV_HC"]
)

deny_healthcare_rules = convert_rules_to_domain_format(
    deny_rules,
    healthcare_output_format["SA_HC"],
    healthcare_output_format["OA_HC"],
    healthcare_output_format["EA_HC"],
    healthcare_output_format["SAV_HC"],
    healthcare_output_format["OAV_HC"],
    healthcare_output_format["EAV_HC"]
)

print(f"✓ Permit rules converted to healthcare format ({len(permit_healthcare_rules)} rules)")
print(f"✓ Deny rules converted to healthcare format ({len(deny_healthcare_rules)} rules)")
print(f"  Example healthcare rules:")
for rule in permit_healthcare_rules[:1]:
    print(f"    {rule[:100]}...")
for rule in deny_healthcare_rules[:1]:
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

def fill_matrix_with_rules(A, SV, OV, EV, permit_rules, deny_rules, n1, n2, n3):
    """
    Fill ACM with precedence logic:
    - If both permit_rules = 0 and deny_rules = 0: everything is denied (acm = 0)
    - If permit_rules > 0 and deny_rules = 0: current behavior (acm = 1 if matches permit, else 0)
    - If permit_rules = 0 and deny_rules > 0: acm = 0 if matches deny, else 1
    - If both > 0: permit rules take precedence (if matches both, acm = 1; if matches neither, acm = 0)
    """
    global no_of_ones
    
    has_permit = len(permit_rules) > 0
    has_deny = len(deny_rules) > 0
    
    for i in range(n1):
        for j in range(n2):
            for k in range(n3):
                SA1 = SV[f"S_{i + 1}"]
                OA1 = OV[f"O_{j + 1}"]
                EA1 = EV[f"E_{k + 1}"]
                
                matches_permit = any(satisfies_rule(rule, SA1, OA1, EA1) for rule in permit_rules) if has_permit else False
                matches_deny = any(satisfies_rule(rule, SA1, OA1, EA1) for rule in deny_rules) if has_deny else False
                
                # Apply precedence logic
                if not has_permit and not has_deny:
                    # Case 1: Both 0 - everything denied
                    A[i][j][k] = 0
                elif has_permit and not has_deny:
                    # Case 2: Only permit rules - current behavior
                    A[i][j][k] = 1 if matches_permit else 0
                elif not has_permit and has_deny:
                    # Case 3: Only deny rules - inverse logic
                    A[i][j][k] = 0 if matches_deny else 1
                else:
                    # Case 4: Both accepted and denied - accepted takes precedence
                    # 1 is matched only when it is accepted and does not match deny
                    A[i][j][k] = 1 if (matches_permit and not matches_deny) else 0
                
                no_of_ones += A[i][j][k]

fill_matrix_with_rules(A, SV, OV, EV, permit_rules, deny_rules, n1, n2, n3)
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

# Append healthcare rules (both permit and deny)
output_data_hc["permit_rules_HC"] = permit_healthcare_rules
output_data_hc["deny_rules_HC"] = deny_healthcare_rules

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

university_rules = convert_rules_to_domain_format(
    permit_rules,
    university_output_format["SA_UNI"],
    university_output_format["OA_UNI"],
    university_output_format["EA_UNI"],
    university_output_format["SAV_UNI"],
    university_output_format["OAV_UNI"],
    university_output_format["EAV_UNI"]
)

deny_university_rules = convert_rules_to_domain_format(
    deny_rules,
    university_output_format["SA_UNI"],
    university_output_format["OA_UNI"],
    university_output_format["EA_UNI"],
    university_output_format["SAV_UNI"],
    university_output_format["OAV_UNI"],
    university_output_format["EAV_UNI"]
)

print(f"✓ Permit rules converted to university format ({len(university_rules)} rules)")
print(f"✓ Deny rules converted to university format ({len(deny_university_rules)} rules)")
print(f"  Example university rules:")
for rule in university_rules[:1]:
    print(f"    {rule[:100]}...")
for rule in deny_university_rules[:1]:
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

# Append university rules (both permit and deny)
output_data_uni["permit_rules_UNI"] = university_rules
output_data_uni["deny_rules_UNI"] = deny_university_rules

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
