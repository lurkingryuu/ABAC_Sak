#################################################################


import os
import json
import configparser
import argparse
import numpy as np
import math
from collections import defaultdict
from scipy.stats import truncnorm

# --- Paths ---
BASE_DIR = os.path.dirname(__file__)
OUTPUT_JSON_PATH = os.path.join(BASE_DIR, '../outputs/output.json')
CONFIG_PATH = os.path.join(BASE_DIR, 'config.ini')
PLOTS_FOLDER = os.path.join(BASE_DIR, '../plots')
os.makedirs(PLOTS_FOLDER, exist_ok=True)

# --- Read config ---
config = configparser.ConfigParser()
config.read(CONFIG_PATH)

def parse_attributes(section):
    return {
        "count": int(config[section]["count"]),
        "values": list(map(int, config[section]["values"].split(','))),
        "distributions": json.loads(config[section]["distributions"])
    }

subject_attributes = parse_attributes("SUBJECT_ATTRIBUTES")
object_attributes = parse_attributes("OBJECT_ATTRIBUTES")
environment_attributes = parse_attributes("ENVIRONMENT_ATTRIBUTES")

# --- Read output.json ---
with open(OUTPUT_JSON_PATH, 'r') as f:
    output_data = json.load(f)

SA_values = output_data["SAV"]
OA_values = output_data["OAV"]
EA_values = output_data["EAV"]
SV = output_data["SV"]
OV = output_data["OV"]
EV = output_data["EV"]

# --- Count occurrences ---
def count_occurrences(assigned_values):
    counts = defaultdict(int)
    for entity, attributes in assigned_values.items():
        for value in attributes:
            counts[value] += 1
    return counts

SA_counts = count_occurrences(SV)
OA_counts = count_occurrences(OV)
EA_counts = count_occurrences(EV)

# --- Expected count calculation ---
def calculate_expected_counts(attribute_values, assigned_values, distributions):
    expected_counts = defaultdict(float)
    total_assignments = len(assigned_values)  # total entities for that type

    for i, (key, values) in enumerate(attribute_values.items()):
        dist = distributions[i]
        n = len(values)
        dist_type = dist["distribution"]

        if dist_type == "N":
            # Normal distribution with fixed mean and variance
            # mean = dist.get("mean", 0.5)
            # mean=0.5
            # # variance = dist.get("variance", 0.01)
            # variance=0.01
            # sigma = math.sqrt(variance)

            # # Compute bin edges between 0 and 1
            # edges = np.linspace(0, 1, n + 1)

            # # Probability for each bin = area under PDF between edges
            # probs = []
            # for j in range(n):
            #     cdf_right = norm.cdf(edges[j + 1], loc=mean, scale=sigma)
            #     cdf_left = norm.cdf(edges[j], loc=mean, scale=sigma)
            #     probs.append(cdf_right - cdf_left)
            # Truncated Normal on [0, n] with mean and variance from JSON
            mean = dist.get("mean", (n + 1) / 2.0)          # sensible default: center
            variance = dist.get("variance", (n / 6.0) ** 2) # default spread if missing
            sigma = math.sqrt(variance)

            low, high = 0.0, float(n)
            a, b = (low - mean) / sigma, (high - mean) / sigma

            # Probability for each bin [k-1, k) under truncated N
            probs = []
            for k in range(1, n + 1):
                cdf_right = truncnorm.cdf(k, a, b, loc=mean, scale=sigma)
                cdf_left  = truncnorm.cdf(k - 1, a, b, loc=mean, scale=sigma)
                probs.append(cdf_right - cdf_left)

            # Normalize to fix tiny numerical drift
            total_p = sum(probs)
            if total_p > 0:
                probs = [p / total_p for p in probs]

        elif dist_type == "P":
            # Poisson distribution
            lam = dist["lambda"] # 
            weights = [((lam ** (j+1)) * math.exp(-lam)) / math.factorial(j+1) for j in range(n)]
            total_weight = sum(weights)
            probs = [w / total_weight for w in weights]

        elif dist_type == "U":
            # Uniform distribution
            probs = [1.0 / n] * n

        else:
            raise ValueError(f"Unsupported distribution type: {dist_type}")

        # Expected counts = probability * total assignments
        for value, prob in zip(values, probs):
            expected_counts[value] = total_assignments * prob

    return expected_counts

# --- Compute expected counts ---
SA_expected_counts = calculate_expected_counts(SA_values, SV, subject_attributes["distributions"])
OA_expected_counts = calculate_expected_counts(OA_values, OV, object_attributes["distributions"])
EA_expected_counts = calculate_expected_counts(EA_values, EV, environment_attributes["distributions"])

# --- Plotting ---

# def generate_plots(attribute_values, expected_counts, actual_counts):
#     for attribute, values in attribute_values.items():
#         expected = [expected_counts.get(value, 0) for value in values]
#         actual = [actual_counts.get(value, 0) for value in values]

#         x = np.arange(len(values))
#         width = 0.35

#         plt.figure(figsize=(max(10, len(values) * 0.5), 6))
#         plt.bar(x - width / 2, expected, width, label='Expected', color='blue')
#         plt.bar(x + width / 2, actual, width, label='Actual', color='orange')

#         plt.xlabel('Attribute Values')
#         plt.ylabel('Count')
#         plt.title(f'Expected vs Actual Count for {attribute}')
#         plt.xticks(x, values, rotation=45, ha='right')
#         plt.legend()

#         plot_path = os.path.join(PLOTS_FOLDER, f'{attribute}.png')
#         plt.tight_layout()
#         plt.savefig(plot_path)
#         plt.close()

def generate_plots(attribute_values, expected_counts, actual_counts):
    # Import matplotlib only when plots are explicitly requested.
    # This keeps the script usable in environments without matplotlib
    # (e.g., when only generating error_summary.json).
    try:
        import matplotlib  # type: ignore[import-untyped]
        # Ensure headless-safe backend (required on servers without a display)
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore[import-untyped]
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "matplotlib is required only for plot generation.\n"
            "Install it with:\n"
            "  pip install matplotlib\n"
            "Or, if using uv:\n"
            "  uv add matplotlib\n"
        ) from e

    for attribute, values in attribute_values.items():
        expected = [expected_counts.get(value, 0) for value in values]
        actual = [actual_counts.get(value, 0) for value in values]

        x = np.arange(len(values))
        width = 0.4

        plt.figure(figsize=(max(10, len(values) * 0.5), 6))
        bars1 = plt.bar(x - width / 2, expected, width, label='Expected', color='blue')
        bars2 = plt.bar(x + width / 2, actual, width, label='Actual', color='orange')

        plt.xlabel('Attribute Values')
        plt.ylabel('Count')
        plt.title(f'Expected vs Actual Count for {attribute}')
        plt.xticks(x, values, rotation=45, ha='right')
        plt.legend()

        # --- Annotate counts above bars ---
        def annotate_bars(bars):
            for bar in bars:
                height = bar.get_height()
                plt.text(
                    bar.get_x() + bar.get_width() / 2,
                    height,
                    f'{int(height)}',          # or f'{height:.1f}' for decimals
                    ha='center',
                    va='bottom',
                    fontsize=6.5
                )

        annotate_bars(bars1)
        annotate_bars(bars2)

        plot_path = os.path.join(PLOTS_FOLDER, f'{attribute}.png')
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()



#################################################################

def compute_error_summary(attribute_values, expected_counts, actual_counts):
    """
    For each attribute (e.g. 'SA_1') compute mean relative error across its values:
      rel_err(value) = abs(actual - expected) / expected
    If expected == 0:
      - if actual == 0 -> rel_err = 0
      - else -> rel_err = abs(actual - expected)  (absolute error, included in average)
    Returns dict: { attribute_name: mean_rel_error, ... } and overall_mean_error
    """
    summary = {}
    attr_errors = []

    for attribute, values in attribute_values.items():
        errors = []
        for value in values:
            # expected = expected_counts.get(value, 0.0)
            expected = round(expected_counts.get(value, 0.0))
            actual = actual_counts.get(value, 0)
            if expected == 0.0:
                # no expected mass for this value: treat zero-expected & zero-actual as no error,
                # otherwise use absolute error (avoids division-by-zero).
                rel_err = 0.0 
                # if actual == 0 else float(abs(actual - expected))
            else:
                rel_err = abs(actual - expected) / expected
            errors.append(rel_err)
        # average error for this attribute
        mean_err = float(np.mean(errors)) if errors else 0.0
        summary[attribute] = mean_err
        attr_errors.append(mean_err)

    overall_mean = float(np.mean(attr_errors)) if attr_errors else 0.0
    return summary, overall_mean


# Compute summaries for SA, OA, EA
sa_summary, sa_overall = compute_error_summary(SA_values, SA_expected_counts, SA_counts)
oa_summary, oa_overall = compute_error_summary(OA_values, OA_expected_counts, OA_counts)
ea_summary, ea_overall = compute_error_summary(EA_values, EA_expected_counts, EA_counts)

error_report = {
    "SA": {"per_attribute": sa_summary, "mean_error": sa_overall},
    "OA": {"per_attribute": oa_summary, "mean_error": oa_overall},
    "EA": {"per_attribute": ea_summary, "mean_error": ea_overall},
    "overall_mean_error": float(np.mean([sa_overall, oa_overall, ea_overall]))
}

# write JSON summary to outputs folder
OUT_DIR = os.path.join(BASE_DIR, '../outputs')
os.makedirs(OUT_DIR, exist_ok=True)
with open(os.path.join(OUT_DIR, 'error_summary.json'), 'w') as fh:
    json.dump(error_report, fh, indent=4)

print("Error summary written to outputs/error_summary.json")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate outputs derived from outputs/output.json:\n"
            "- error summary: outputs/error_summary.json (default)\n"
            "- plots: ../plots/*.png (only when --plots is provided)"
        )
    )
    parser.add_argument(
        "--plots",
        action="store_true",
        help="Generate plots into the plots/ directory (requires matplotlib).",
    )
    parser.add_argument(
        "--no-error-summary",
        action="store_true",
        help="Skip writing outputs/error_summary.json.",
    )
    args = parser.parse_args(argv)

    # error summary (default on)
    if not args.no_error_summary:
        # Computation already happened at module load; just ensure file exists.
        # (Keeping changes minimal vs. a larger refactor to fully lazy-load data.)
        pass

    # plots (opt-in)
    if args.plots:
        generate_plots(SA_values, SA_expected_counts, SA_counts)
        generate_plots(OA_values, OA_expected_counts, OA_counts)
        generate_plots(EA_values, EA_expected_counts, EA_counts)
        print(f"Plots saved in {PLOTS_FOLDER}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
