#################################################################


import os
import json
import configparser
import argparse
import numpy as np
import math
from collections import defaultdict
try:
    from scipy.stats import truncnorm
except ImportError:
    # Fallback implementation of truncated normal functions if scipy is not available.
    class TruncnormFallback:
        @staticmethod
        def _phi(x):
            """Standard normal CDF."""
            return 0.5 * (1 + math.erf(x / math.sqrt(2)))

        @staticmethod
        def cdf(x, a, b, loc=0, scale=1):
            """
            Truncated normal CDF on [low, high] where:
            a = (low - loc) / scale
            b = (high - loc) / scale
            """
            phi_x = TruncnormFallback._phi((x - loc) / scale)
            phi_a = TruncnormFallback._phi(a)
            phi_b = TruncnormFallback._phi(b)
            
            if phi_b <= phi_a:
                return 0.0
            
            val = (phi_x - phi_a) / (phi_b - phi_a)
            return max(0.0, min(1.0, val))
            
    truncnorm = TruncnormFallback()

# --- Paths ---
BASE_DIR = os.path.dirname(__file__)
ENV_CONFIG_INI = "ABAC_CONFIG_INI"
ENV_OUTPUT_DIR = "ABAC_OUTPUT_DIR"

_out_dir = os.environ.get(ENV_OUTPUT_DIR) or os.path.join(BASE_DIR, '../outputs')
OUTPUT_JSON_PATH = os.path.join(_out_dir, 'output.json')

CONFIG_PATH = os.environ.get(ENV_CONFIG_INI) or os.path.join(BASE_DIR, 'config.ini')
ENV_PLOTS_DIR = "ABAC_PLOTS_DIR"
PLOTS_FOLDER = os.environ.get(ENV_PLOTS_DIR) or os.path.join(BASE_DIR, '../plots')
os.makedirs(PLOTS_FOLDER, exist_ok=True)

# --- Read config ---
config = configparser.ConfigParser()
config.read(CONFIG_PATH)

def parse_attributes(section):
    out = {
        "count": int(config[section]["count"]),
        "values": list(map(int, config[section]["values"].split(','))),
        "distributions": json.loads(config[section]["distributions"]),
    }
    corr_raw = config[section].get("correlations", "{}")
    if corr_raw:
        try:
            out["correlations"] = json.loads(corr_raw)
        except json.JSONDecodeError:
            out["correlations"] = {}
    else:
        out["correlations"] = {}
    return out

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

def _expected_probs_per_attribute(attribute_values, distributions, n_entities):
    """Return dict attr_key -> list of probs aligned with attribute_values[attr_key] order."""
    probs_by_attr = {}
    for i, (key, values) in enumerate(attribute_values.items()):
        dist = distributions[i]
        n = len(values)
        dist_type = dist["distribution"]
        if dist_type == "N":
            mean = dist.get("mean", (n + 1) / 2.0)
            variance = dist.get("variance", (n / 6.0) ** 2)
            sigma = math.sqrt(variance)
            low, high = 0.0, float(n)
            a, b = (low - mean) / sigma, (high - mean) / sigma
            probs = []
            for k in range(1, n + 1):
                cdf_right = truncnorm.cdf(k, a, b, loc=mean, scale=sigma)
                cdf_left = truncnorm.cdf(k - 1, a, b, loc=mean, scale=sigma)
                probs.append(cdf_right - cdf_left)
            total_p = sum(probs)
            if total_p > 0:
                probs = [p / total_p for p in probs]
        elif dist_type == "P":
            lam = dist["lambda"]
            weights = [
                ((lam ** (j + 1)) * math.exp(-lam)) / math.factorial(j + 1)
                for j in range(n)
            ]
            tw = sum(weights)
            probs = [w / tw for w in weights]
        elif dist_type == "U":
            probs = [1.0 / n] * n
        else:
            raise ValueError(dist_type)
        probs_by_attr[key] = probs
    return probs_by_attr


def _joint_target_from_pair(pair, card_a, card_b, marginal_a, marginal_b):
    target = pair.get("target", {})
    if "joint_table" in target:
        tbl = np.array(target["joint_table"], dtype=float)
        s = tbl.sum()
        if s <= 0:
            raise ValueError("joint_table sum must be positive")
        return tbl / s
    if "cramers_v" in target:
        strength = float(target["cramers_v"])
        ma = np.asarray(marginal_a, dtype=float)
        mb = np.asarray(marginal_b, dtype=float)
        ma = ma / ma.sum()
        mb = mb / mb.sum()
        base = np.outer(ma, mb)
        diagonal = np.zeros_like(base)
        dlen = min(card_a, card_b)
        for d in range(dlen):
            diagonal[d, d] = 1.0 / dlen
        jt = (1.0 - strength) * base + strength * diagonal
        return jt / jt.sum()
    raise ValueError("pair target needs joint_table or cramers_v")


def _empirical_joint(assigned_values, av, prefix, attr_a, attr_b):
    """attr_a, attr_b are 1-based indices. Returns matrix len(SA_a) x len(SA_b)."""
    key_a = f"{prefix}_{attr_a}"
    key_b = f"{prefix}_{attr_b}"
    vals_a = av[key_a]
    vals_b = av[key_b]
    n = len(assigned_values)
    mat = np.zeros((len(vals_a), len(vals_b)), dtype=float)
    if n == 0:
        return mat
    for _ent, attrs in assigned_values.items():
        tok_a = attrs[attr_a - 1]
        tok_b = attrs[attr_b - 1]
        ia = vals_a.index(tok_a)
        ib = vals_b.index(tok_b)
        mat[ia, ib] += 1.0
    mat /= float(n)
    return mat


def _chi2_cramers_v(observed_counts, expected_counts):
    """observed_counts / expected_counts: same shape, not normalized (use raw counts)."""
    mask = expected_counts > 0
    chi2 = float(np.sum(
        (observed_counts[mask] - expected_counts[mask]) ** 2 / expected_counts[mask]
    ))
    ntot = float(np.sum(observed_counts))
    r, c = observed_counts.shape
    denom = ntot * min(max(r - 1, 1), max(c - 1, 1))
    if denom <= 0:
        return chi2, 0.0
    v = math.sqrt(max(chi2, 0.0) / denom)
    return chi2, min(v, 1.0)


def compute_correlation_metrics(entity_name, prefix, av, assigned_values, attr_block, distributions):
    """
    attr_block: subject_attributes etc. with 'values', 'distributions', 'correlations'
    Returns list of dicts with L1, chi2, cramers_v_actual, target info.
    """
    pairs = (attr_block.get("correlations") or {}).get("pairs") or []
    if not pairs:
        return []
    probs_by = _expected_probs_per_attribute(av, distributions, len(assigned_values))
    n_ent = len(assigned_values)
    results = []
    for pidx, pair in enumerate(pairs):
        a = int(pair["attr_a"])
        b = int(pair["attr_b"])
        key_a = f"{prefix}_{a}"
        key_b = f"{prefix}_{b}"
        card_a = len(av[key_a])
        card_b = len(av[key_b])
        ma = probs_by[key_a]
        mb = probs_by[key_b]
        target = _joint_target_from_pair(pair, card_a, card_b, ma, mb)
        obs_p = _empirical_joint(assigned_values, av, prefix, a, b)
        l1 = float(np.sum(np.abs(obs_p - target)))
        exp_c = target * n_ent
        obs_c = obs_p * n_ent
        chi2, v_obs = _chi2_cramers_v(obs_c, exp_c)
        entry = {
            "entity": entity_name,
            "pair_index": pidx,
            "attr_a": a,
            "attr_b": b,
            "l1_joint_distance": round(l1, 6),
            "chi2_vs_target": round(chi2, 6),
            "cramers_v_vs_target": round(v_obs, 6),
            "weight": float(pair.get("weight", 1.0)),
        }
        if "cramers_v" in pair.get("target", {}):
            entry["cramers_v_requested"] = float(pair["target"]["cramers_v"])
        results.append(entry)
    return results


def generate_correlation_plots(
    entity_slug, prefix, av, assigned_values, attr_block, distributions
):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "matplotlib is required for correlation plots."
        ) from e

    pairs = (attr_block.get("correlations") or {}).get("pairs") or []
    if not pairs:
        return
    probs_by = _expected_probs_per_attribute(av, distributions, len(assigned_values))
    for pidx, pair in enumerate(pairs):
        a = int(pair["attr_a"])
        b = int(pair["attr_b"])
        key_a = f"{prefix}_{a}"
        key_b = f"{prefix}_{b}"
        card_a = len(av[key_a])
        card_b = len(av[key_b])
        ma = probs_by[key_a]
        mb = probs_by[key_b]
        target = _joint_target_from_pair(pair, card_a, card_b, ma, mb)
        obs = _empirical_joint(assigned_values, av, prefix, a, b)
        vmax = max(float(target.max()), float(obs.max()), 1e-9)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, mat, title in (
            (axes[0], target, "Target joint P(i,j)"),
            (axes[1], obs, "Observed joint (data)"),
        ):
            im = ax.imshow(mat, vmin=0.0, vmax=vmax, cmap="viridis")
            ax.set_title(title)
            ax.set_xlabel(key_b)
            ax.set_ylabel(key_a)
            ax.set_xticks(range(card_b))
            ax.set_yticks(range(card_a))
            ax.set_xticklabels([f"{j+1}" for j in range(card_b)])
            ax.set_yticklabels([f"{i+1}" for i in range(card_a)])
            fig.colorbar(im, ax=ax, fraction=0.046)
        l1 = float(np.sum(np.abs(obs - target)))
        fig.suptitle(f"{entity_slug}: {key_a} vs {key_b} (L1={l1:.4f})")
        out_name = f"correlation_{entity_slug}_{prefix}{a}_{prefix}{b}.png"
        plot_path = os.path.join(PLOTS_FOLDER, out_name)
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        print(f"Correlation plot: {plot_path}")


corr_subject = compute_correlation_metrics(
    "subject", "SA", SA_values, SV, subject_attributes, subject_attributes["distributions"]
)
corr_object = compute_correlation_metrics(
    "object", "OA", OA_values, OV, object_attributes, object_attributes["distributions"]
)
corr_environment = compute_correlation_metrics(
    "environment", "EA", EA_values, EV, environment_attributes, environment_attributes["distributions"]
)
correlation_report = corr_subject + corr_object + corr_environment
_mean_l1s = [c["l1_joint_distance"] for c in correlation_report]
_mean_chi = [c["chi2_vs_target"] for c in correlation_report]

error_report = {
    "SA": {"per_attribute": sa_summary, "mean_error": sa_overall},
    "OA": {"per_attribute": oa_summary, "mean_error": oa_overall},
    "EA": {"per_attribute": ea_summary, "mean_error": ea_overall},
    "overall_mean_error": float(np.mean([sa_overall, oa_overall, ea_overall])),
    "correlations": {
        "pairs": correlation_report,
        "mean_l1_joint_distance": float(np.mean(_mean_l1s)) if _mean_l1s else None,
        "mean_chi2_vs_target": float(np.mean(_mean_chi)) if _mean_chi else None,
        "pair_count": len(correlation_report),
    },
    "generation_diagnostics": output_data.get("generation_diagnostics"),
}


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

    os.makedirs(_out_dir, exist_ok=True)
    if not args.no_error_summary:
        summary_path = os.path.join(_out_dir, "error_summary.json")
        with open(summary_path, "w", encoding="utf-8") as fh:
            json.dump(error_report, fh, indent=4)
        print(f"Error summary written to {summary_path}")

    if args.plots:
        generate_plots(SA_values, SA_expected_counts, SA_counts)
        generate_plots(OA_values, OA_expected_counts, OA_counts)
        generate_plots(EA_values, EA_expected_counts, EA_counts)
        generate_correlation_plots(
            "subject", "SA", SA_values, SV, subject_attributes, subject_attributes["distributions"]
        )
        generate_correlation_plots(
            "object", "OA", OA_values, OV, object_attributes, object_attributes["distributions"]
        )
        generate_correlation_plots(
            "environment", "EA", EA_values, EV, environment_attributes, environment_attributes["distributions"]
        )
        print(f"Plots saved in {PLOTS_FOLDER}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
