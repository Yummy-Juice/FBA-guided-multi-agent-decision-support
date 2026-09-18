#!/usr/bin/env python3
"""Generate the versioned submission analysis and publication figures.

This is the sole analysis entry point for the journal-submission package. It
uses the anonymous public rating data to provide ordinal summaries, uncertainty
intervals, multiplicity-adjusted tests, and the documented two-round expert
Delphi records. These analyses describe teacher perceptions and expert ratings
only; they do not establish diagnostic accuracy, equivalence to professional
FBA, student outcomes, or autonomous decision-making capability.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import MultipleLocator, PercentFormatter
from scipy import stats

from validate_inputs import DIMENSIONS, load_and_validate_submission_inputs

# Keep vector text editable in every export. The plotting layer intentionally
# overrides the family to a compact serif setting below to match the manuscript
# figure vocabulary used across the submission package.
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PACKAGE_ROOT / "data" / "public_ratings"
DEFAULT_OUTPUT_DIR = PACKAGE_ROOT / "results"
THEORETICAL_MIDPOINT = 3.0
RATING_LEVELS = (1, 2, 3, 4, 5)

DIMENSION_LABELS = {
    "antecedent_identification": "Antecedent identification",
    "consequence_identification": "Consequence identification",
    "setting_event_identification": "Setting-event identification",
    "core_function_hypothesis": "Core function hypothesis",
    "replacement_behavior_recommendation": "Replacement-behavior recommendation",
    "overall_rating": "Overall rating",
}

DIMENSION_SHORT_LABELS = {
    "antecedent_identification": "Antecedent",
    "consequence_identification": "Consequence",
    "setting_event_identification": "Setting event",
    "core_function_hypothesis": "Core hypothesis",
    "replacement_behavior_recommendation": "Replacement\nbehavior",
    "overall_rating": "Overall",
}

ROUND_LABELS = {
    "round_1": "Round 1",
    "round_2": "Round 2",
    "round_average": "Arithmetic mean of rounds",
}

COLORS = {
    "blue": "#355C7D",
    "blue_soft": "#C5D2DD",
    "orange": "#B86449",
    "orange_soft": "#E6C3B5",
    "green": "#2F7A74",
    "purple": "#746581",
    "gray": "#68737B",
    "light_gray": "#D7DADD",
    "pale_gray": "#F5F3EF",
    "dark": "#24323D",
    "rating_1": "#D3DDE4",
    "rating_2": "#A9BECE",
    "rating_3": "#E3C16A",
    "rating_4": "#4F8C9A",
    "rating_5": "#2F5F83",
}

# Ordered light-to-dark blue scale reserved for the ordinal-proportion stacks.
ORDINAL_PROPORTION_COLORS = {
    1: "#D6EAF8",
    2: "#A4C8E8",
    3: "#6A9FD1",
    4: "#2F6FA8",
    5: "#164B78",
}

GENERAL_AI_ORDINAL_PROPORTION_COLORS = {
    1: "#FCE9D9",
    2: "#F7CAA6",
    3: "#EA9C6E",
    4: "#BB6648",
    5: "#884A3D",
}


def dimension_key(column: str) -> str:
    """Return the documented dimension name from a measure-specific column."""
    for prefix in ("quality_", "human_fba_similarity_", "general_ai_similarity_"):
        if column.startswith(prefix):
            return column.removeprefix(prefix)
    return column


def dimension_label(column: str) -> str:
    """Return the public English label for one documented rating dimension."""
    return DIMENSION_LABELS.get(dimension_key(column), column)


def configure_matplotlib() -> None:
    """Set the compact serif layout used by the submission figure set."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "STIX Two Text", "STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 7.4,
            "axes.titlesize": 9.2,
            "axes.labelsize": 7.8,
            "xtick.labelsize": 6.9,
            "ytick.labelsize": 6.9,
            "legend.fontsize": 6.5,
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.linewidth": 0.72,
            "axes.edgecolor": COLORS["dark"],
            "axes.labelcolor": COLORS["dark"],
            "xtick.color": COLORS["dark"],
            "ytick.color": COLORS["dark"],
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    """Write a UTF-8, LF-terminated CSV and create its parent directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")


def wilson_interval(successes: int, total: int, confidence: float = 0.95) -> tuple[float, float]:
    """Return a Wilson binomial confidence interval for an observed proportion."""
    if total <= 0:
        return (np.nan, np.nan)
    z_value = stats.norm.ppf(1 - (1 - confidence) / 2)
    proportion = successes / total
    denominator = 1 + z_value**2 / total
    centre = (proportion + z_value**2 / (2 * total)) / denominator
    half_width = (
        z_value
        * np.sqrt((proportion * (1 - proportion) + z_value**2 / (4 * total)) / total)
        / denominator
    )
    return (float(max(0.0, centre - half_width)), float(min(1.0, centre + half_width)))


def mean_ci(values: pd.Series, confidence: float = 0.95) -> tuple[float, float]:
    """Return a t-based confidence interval for a mean when it is estimable."""
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if len(numeric) < 2:
        return (np.nan, np.nan)
    standard_error = numeric.std(ddof=1) / np.sqrt(len(numeric))
    critical_value = stats.t.ppf(1 - (1 - confidence) / 2, len(numeric) - 1)
    centre = float(numeric.mean())
    return (centre - critical_value * standard_error, centre + critical_value * standard_error)


def holm_adjust(p_values: Iterable[float]) -> list[float]:
    """Apply Holm's family-wise adjustment while preserving input order."""
    raw = np.asarray(list(p_values), dtype=float)
    adjusted = np.full(raw.shape, np.nan, dtype=float)
    finite_indices = np.flatnonzero(np.isfinite(raw))
    if not len(finite_indices):
        return adjusted.tolist()
    ordered_indices = finite_indices[np.argsort(raw[finite_indices])]
    running_maximum = 0.0
    number_of_tests = len(ordered_indices)
    for rank, index in enumerate(ordered_indices):
        candidate = min(1.0, (number_of_tests - rank) * raw[index])
        running_maximum = max(running_maximum, candidate)
        adjusted[index] = running_maximum
    return adjusted.tolist()


def cronbach_alpha(frame: pd.DataFrame) -> tuple[float, int, int]:
    """Calculate alpha using complete cases for the documented rating scale."""
    complete = frame.apply(pd.to_numeric, errors="coerce").dropna(axis=0, how="any")
    number_of_cases, number_of_items = complete.shape
    if number_of_cases < 2 or number_of_items < 2:
        return (np.nan, number_of_cases, number_of_items)
    item_variance = complete.var(axis=0, ddof=1).sum()
    total_variance = complete.sum(axis=1).var(ddof=1)
    if total_variance <= 0:
        return (np.nan, number_of_cases, number_of_items)
    alpha = number_of_items / (number_of_items - 1) * (1 - item_variance / total_variance)
    return (float(alpha), number_of_cases, number_of_items)


def corrected_item_total_correlations(frame: pd.DataFrame, scale_label: str) -> pd.DataFrame:
    """Calculate item-level CITCs from listwise complete responses."""
    numeric = frame.apply(pd.to_numeric, errors="coerce").dropna(axis=0, how="any")
    alpha, sample_size, number_of_items = cronbach_alpha(numeric)
    rows: list[dict[str, object]] = []
    for column in numeric.columns:
        item = numeric[column]
        remainder = numeric.drop(columns=column).sum(axis=1)
        if len(item) < 3 or item.std(ddof=1) == 0 or remainder.std(ddof=1) == 0:
            correlation, p_value = np.nan, np.nan
        else:
            correlation, p_value = stats.pearsonr(item, remainder)
        rows.append(
            {
                "scale": scale_label,
                "dimension": dimension_label(column),
                "n_listwise_complete": sample_size,
                "number_of_items": number_of_items,
                "cronbach_alpha": alpha,
                "corrected_item_total_correlation": correlation,
                "citc_p_value": p_value,
                "citc_at_least_0_30": bool(np.isfinite(correlation) and correlation >= 0.30),
            }
        )
    return pd.DataFrame(rows)


def coefficient_of_variation(frame: pd.DataFrame, scale_label: str) -> pd.DataFrame:
    """Report dimension-specific variation without removing missing observations."""
    rows: list[dict[str, object]] = []
    for column in frame.columns:
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        standard_deviation = values.std(ddof=1) if len(values) > 1 else np.nan
        average = values.mean() if len(values) else np.nan
        rows.append(
            {
                "scale": scale_label,
                "dimension": dimension_label(column),
                "n": len(values),
                "mean": average,
                "sd": standard_deviation,
                "coefficient_of_variation": standard_deviation / average
                if np.isfinite(average) and average != 0
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def ordinal_summary(
    frame: pd.DataFrame,
    columns: list[str],
    *,
    study: str,
    sample: str,
    measure: str,
    score_set: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize bounded ratings without discarding recorded decimal values."""
    summary_rows: list[dict[str, object]] = []
    distribution_rows: list[dict[str, object]] = []
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        number_ratings = int(len(values))
        lower_quartile = float(values.quantile(0.25)) if number_ratings else np.nan
        upper_quartile = float(values.quantile(0.75)) if number_ratings else np.nan
        endorse_count = int((values >= 4).sum())
        ci_low, ci_high = wilson_interval(endorse_count, number_ratings)
        row: dict[str, object] = {
            "study": study,
            "sample": sample,
            "measure": measure,
            "score_set": score_set or "not_applicable",
            "dimension": dimension_label(column),
            "n": number_ratings,
            "mean": float(values.mean()) if number_ratings else np.nan,
            "sd": float(values.std(ddof=1)) if number_ratings > 1 else np.nan,
            "median": float(values.median()) if number_ratings else np.nan,
            "q1": lower_quartile,
            "q3": upper_quartile,
            "iqr": upper_quartile - lower_quartile if number_ratings else np.nan,
            "median_iqr": f"{values.median():.2f} ({lower_quartile:.2f}-{upper_quartile:.2f})"
            if number_ratings
            else "",
            "rating_at_least_4_n": endorse_count,
            "rating_at_least_4_proportion": endorse_count / number_ratings if number_ratings else np.nan,
            "rating_at_least_4_ci_95_low": ci_low,
            "rating_at_least_4_ci_95_high": ci_high,
            "rating_at_least_4_ci_method": "Wilson",
        }
        for rating in RATING_LEVELS:
            count = int((values == rating).sum())
            row[f"rating_{rating}_n"] = count
            row[f"rating_{rating}_proportion"] = count / number_ratings if number_ratings else np.nan

        observed_levels = {float(value) for value in values.unique()}
        distribution_levels = sorted({float(rating) for rating in RATING_LEVELS} | observed_levels)
        distribution_total = 0
        noninteger_total = 0
        for rating in distribution_levels:
            count = int((values == rating).sum())
            distribution_total += count
            if not float(rating).is_integer():
                noninteger_total += count
            distribution_rows.append(
                {
                    "study": study,
                    "sample": sample,
                    "measure": measure,
                    "score_set": score_set or "not_applicable",
                    "dimension": row["dimension"],
                    "rating": int(rating) if float(rating).is_integer() else rating,
                    "n": count,
                    "proportion": count / number_ratings if number_ratings else np.nan,
                }
            )
        if distribution_total != number_ratings:
            raise ValueError(
                f"Rating distribution lost observations for {score_set or 'not_applicable'} / "
                f"{row['dimension']}: expected {number_ratings}, counted {distribution_total}."
            )
        row["noninteger_rating_n"] = noninteger_total
        row["distribution_count_total"] = distribution_total
        summary_rows.append(row)
    return pd.DataFrame(summary_rows), pd.DataFrame(distribution_rows)


def expert_cvi_table(round_frames: dict[str, pd.DataFrame], columns: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate expert I-CVI and S-CVI/Ave separately for each documented score set."""
    item_rows: list[dict[str, object]] = []
    scale_rows: list[dict[str, object]] = []
    for score_set, frame in round_frames.items():
        values_for_scale: list[float] = []
        for column in columns:
            values = pd.to_numeric(frame[column], errors="coerce").dropna()
            agreeing = int((values >= 4).sum())
            i_cvi = agreeing / len(values) if len(values) else np.nan
            values_for_scale.append(i_cvi)
            item_rows.append(
                {
                    "score_set": ROUND_LABELS[score_set],
                    "dimension": dimension_label(column),
                    "n_experts": len(values),
                    "n_rating_at_least_4": agreeing,
                    "i_cvi_rating_at_least_4": i_cvi,
                    "threshold": "rating >= 4",
                }
            )
        scale_rows.append(
            {
                "score_set": ROUND_LABELS[score_set],
                "n_experts": len(frame),
                "number_of_dimensions": len(columns),
                "s_cvi_ave": float(np.nanmean(values_for_scale)),
                "threshold": "rating >= 4",
            }
        )
    return pd.DataFrame(item_rows), pd.DataFrame(scale_rows)


def kendalls_w(frame: pd.DataFrame, columns: list[str]) -> dict[str, float]:
    """Calculate Kendall's W across dimensions from the expert-by-dimension matrix."""
    matrix = frame.loc[:, columns].apply(pd.to_numeric, errors="coerce").dropna(axis=0, how="any").to_numpy()
    number_of_raters, number_of_items = matrix.shape
    if number_of_raters < 2 or number_of_items < 2:
        return {
            "n_experts": number_of_raters,
            "number_of_dimensions": number_of_items,
            "kendalls_w": np.nan,
            "chi_square": np.nan,
            "degrees_of_freedom": np.nan,
            "p_value": np.nan,
            "tie_correction": np.nan,
        }
    ranked = np.vstack([stats.rankdata(row, method="average") for row in matrix])
    rank_sums = ranked.sum(axis=0)
    centre = number_of_raters * (number_of_items + 1) / 2
    sum_squared_deviations = float(((rank_sums - centre) ** 2).sum())
    tie_correction = 0.0
    for row in matrix:
        _, counts = np.unique(row, return_counts=True)
        tie_correction += float(np.sum(counts**3 - counts))
    denominator = number_of_raters**2 * (number_of_items**3 - number_of_items) - number_of_raters * tie_correction
    concordance = 12 * sum_squared_deviations / denominator if denominator > 0 else np.nan
    chi_square = number_of_raters * (number_of_items - 1) * concordance if np.isfinite(concordance) else np.nan
    degrees_of_freedom = number_of_items - 1
    p_value = stats.chi2.sf(chi_square, degrees_of_freedom) if np.isfinite(chi_square) else np.nan
    return {
        "n_experts": number_of_raters,
        "number_of_dimensions": number_of_items,
        "kendalls_w": concordance,
        "chi_square": chi_square,
        "degrees_of_freedom": degrees_of_freedom,
        "p_value": p_value,
        "tie_correction": tie_correction,
    }


def expert_round_change(round_one: pd.DataFrame, round_two: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Describe paired score changes without treating four experts as definitive evidence."""
    left = round_one.set_index("expert_record_id").loc[:, columns]
    right = round_two.set_index("expert_record_id").loc[:, columns]
    changes = right - left
    rows: list[dict[str, object]] = []
    for column in columns:
        values = changes[column].dropna()
        nonzero = values[values != 0]
        if len(nonzero) and len(nonzero) < 4:
            try:
                wilcoxon_p = float(stats.wilcoxon(values, alternative="two-sided", zero_method="wilcox").pvalue)
            except ValueError:
                wilcoxon_p = np.nan
        elif len(nonzero) == 4:
            wilcoxon_p = float(stats.wilcoxon(values, alternative="two-sided", zero_method="wilcox").pvalue)
        else:
            wilcoxon_p = np.nan
        rows.append(
            {
                "dimension": dimension_label(column),
                "n_paired_experts": len(values),
                "mean_round_2_minus_round_1": float(values.mean()) if len(values) else np.nan,
                "median_round_2_minus_round_1": float(values.median()) if len(values) else np.nan,
                "n_increased": int((values > 0).sum()),
                "n_decreased": int((values < 0).sum()),
                "n_unchanged": int((values == 0).sum()),
                "wilcoxon_p_value_two_sided": wilcoxon_p,
                "note": "Exploratory paired description; n=4 experts.",
            }
        )
    return pd.DataFrame(rows)


def one_sample_tests(
    frame: pd.DataFrame,
    columns: list[str],
    *,
    study: str,
    sample: str,
    measure: str,
    alternative: str,
) -> pd.DataFrame:
    """Run prespecified-direction one-sample t tests and add Holm-adjusted p values."""
    rows: list[dict[str, object]] = []
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        if len(values) < 2 or values.std(ddof=1) == 0:
            statistic, p_value, effect_size = np.nan, np.nan, np.nan
        else:
            result = stats.ttest_1samp(values, popmean=THEORETICAL_MIDPOINT, alternative=alternative)
            statistic = float(result.statistic)
            p_value = float(result.pvalue)
            effect_size = float((values.mean() - THEORETICAL_MIDPOINT) / values.std(ddof=1))
        difference = values - THEORETICAL_MIDPOINT
        ci_low, ci_high = mean_ci(difference)
        rows.append(
            {
                "study": study,
                "sample": sample,
                "measure": measure,
                "dimension": dimension_label(column),
                "n": len(values),
                "theoretical_midpoint": THEORETICAL_MIDPOINT,
                "alternative": alternative,
                "mean": float(values.mean()) if len(values) else np.nan,
                "mean_difference_from_midpoint": float(difference.mean()) if len(values) else np.nan,
                "mean_difference_ci_95_low": ci_low,
                "mean_difference_ci_95_high": ci_high,
                "t_statistic": statistic,
                "degrees_of_freedom": len(values) - 1,
                "p_value_directional": p_value,
                "cohens_d": effect_size,
            }
        )
    result = pd.DataFrame(rows)
    result["p_value_holm_6"] = holm_adjust(result["p_value_directional"])
    return result


def repeated_measures_summary(
    frame: pd.DataFrame,
    columns: list[str],
    *,
    study: str,
    sample: str,
    measure: str,
) -> pd.DataFrame:
    """Provide parametric repeated-measures ANOVA and ordinal Friedman sensitivity."""
    complete = frame.loc[:, columns].apply(pd.to_numeric, errors="coerce").dropna(axis=0, how="any")
    number_of_people, number_of_dimensions = complete.shape
    if number_of_people < 2 or number_of_dimensions < 2:
        return pd.DataFrame()
    values = complete.to_numpy(dtype=float)
    grand_mean = float(values.mean())
    person_means = values.mean(axis=1)
    dimension_means = values.mean(axis=0)
    ss_total = float(((values - grand_mean) ** 2).sum())
    ss_people = float(number_of_dimensions * ((person_means - grand_mean) ** 2).sum())
    ss_dimension = float(number_of_people * ((dimension_means - grand_mean) ** 2).sum())
    ss_error = ss_total - ss_people - ss_dimension
    df_dimension = number_of_dimensions - 1
    df_error = (number_of_people - 1) * df_dimension
    ms_dimension = ss_dimension / df_dimension
    ms_error = ss_error / df_error
    f_statistic = ms_dimension / ms_error if ms_error > 0 else np.nan
    p_value = stats.f.sf(f_statistic, df_dimension, df_error) if np.isfinite(f_statistic) else np.nan
    partial_eta_squared = ss_dimension / (ss_dimension + ss_error) if ss_dimension + ss_error > 0 else np.nan
    try:
        friedman = stats.friedmanchisquare(*(complete[column] for column in columns))
        friedman_statistic = float(friedman.statistic)
        friedman_p_value = float(friedman.pvalue)
    except ValueError:
        friedman_statistic, friedman_p_value = np.nan, np.nan
    return pd.DataFrame(
        [
            {
                "study": study,
                "sample": sample,
                "measure": measure,
                "analysis": "repeated_measures_anova",
                "n_listwise_complete": number_of_people,
                "number_of_dimensions": number_of_dimensions,
                "statistic": f_statistic,
                "df_1": df_dimension,
                "df_2": df_error,
                "p_value": p_value,
                "effect_size": partial_eta_squared,
                "effect_size_name": "partial_eta_squared",
                "note": "Univariate repeated-measures ANOVA.",
            },
            {
                "study": study,
                "sample": sample,
                "measure": measure,
                "analysis": "friedman_sensitivity",
                "n_listwise_complete": number_of_people,
                "number_of_dimensions": number_of_dimensions,
                "statistic": friedman_statistic,
                "df_1": number_of_dimensions - 1,
                "df_2": np.nan,
                "p_value": friedman_p_value,
                "effect_size": friedman_statistic / (number_of_people * (number_of_dimensions - 1))
                if np.isfinite(friedman_statistic)
                else np.nan,
                "effect_size_name": "kendalls_w_friedman",
                "note": "Ordinal nonparametric sensitivity analysis.",
            },
        ]
    )


def paired_dimension_tests(
    frame: pd.DataFrame,
    columns: list[str],
    *,
    study: str,
    sample: str,
    measure: str,
) -> pd.DataFrame:
    """Add Holm-adjusted paired contrasts for interpretable follow-up analyses."""
    rows: list[dict[str, object]] = []
    for left, right in combinations(columns, 2):
        paired = frame.loc[:, [left, right]].apply(pd.to_numeric, errors="coerce").dropna()
        difference = paired[left] - paired[right]
        if len(difference) < 2 or difference.std(ddof=1) == 0:
            statistic, p_value, effect_size = np.nan, np.nan, np.nan
        else:
            t_result = stats.ttest_rel(paired[left], paired[right])
            statistic, p_value = float(t_result.statistic), float(t_result.pvalue)
            effect_size = float(difference.mean() / difference.std(ddof=1))
        ci_low, ci_high = mean_ci(difference)
        rows.append(
            {
                "study": study,
                "sample": sample,
                "measure": measure,
                "comparison": f"{dimension_label(left)} minus {dimension_label(right)}",
                "n_paired": len(paired),
                "mean_difference": float(difference.mean()) if len(difference) else np.nan,
                "mean_difference_ci_95_low": ci_low,
                "mean_difference_ci_95_high": ci_high,
                "t_statistic": statistic,
                "degrees_of_freedom": len(paired) - 1,
                "p_value_two_sided": p_value,
                "cohens_dz": effect_size,
            }
        )
    result = pd.DataFrame(rows)
    result["p_value_holm_15"] = holm_adjust(result["p_value_two_sided"])
    return result


def pearson_ci(correlation: float, n: int) -> tuple[float, float]:
    """Calculate a Fisher-z confidence interval for Pearson's r."""
    if n <= 3 or not np.isfinite(correlation) or abs(correlation) >= 1:
        return (np.nan, np.nan)
    transformed = np.arctanh(correlation)
    standard_error = 1 / np.sqrt(n - 3)
    lower, upper = transformed - 1.96 * standard_error, transformed + 1.96 * standard_error
    return (float(np.tanh(lower)), float(np.tanh(upper)))


def association_table(
    quality_frame: pd.DataFrame,
    similarity_frame: pd.DataFrame,
    quality_columns: list[str],
    similarity_columns: list[str],
    *,
    study: str,
    sample: str,
    similarity_measure: str,
) -> pd.DataFrame:
    """Report Pearson associations and Spearman ordinal sensitivity results."""
    rows: list[dict[str, object]] = []
    quality_with_total = quality_frame.copy()
    similarity_with_total = similarity_frame.copy()
    quality_with_total["total"] = quality_with_total[quality_columns].mean(axis=1, skipna=False)
    similarity_with_total["total"] = similarity_with_total[similarity_columns].mean(axis=1, skipna=False)
    pairs = [("Composite mean", "total", "total")]
    pairs.extend(
        (DIMENSION_LABELS[dimension], quality_column, similarity_column)
        for dimension, quality_column, similarity_column in zip(DIMENSIONS, quality_columns, similarity_columns)
    )
    for dimension, quality_column, similarity_column in pairs:
        paired = pd.DataFrame(
            {
                "quality": pd.to_numeric(quality_with_total[quality_column], errors="coerce"),
                "similarity": pd.to_numeric(similarity_with_total[similarity_column], errors="coerce"),
            }
        ).dropna()
        if len(paired) < 3 or paired["quality"].nunique() < 2 or paired["similarity"].nunique() < 2:
            pearson_r = pearson_p = spearman_rho = spearman_p = np.nan
        else:
            pearson_r, pearson_p = stats.pearsonr(paired["quality"], paired["similarity"])
            spearman_rho, spearman_p = stats.spearmanr(paired["quality"], paired["similarity"])
        ci_low, ci_high = pearson_ci(float(pearson_r), len(paired))
        rows.append(
            {
                "study": study,
                "sample": sample,
                "quality_measure": "Teacher-perceived quality rating",
                "similarity_measure": similarity_measure,
                "dimension": dimension,
                "n_paired": len(paired),
                "pearson_r": pearson_r,
                "pearson_r_ci_95_low": ci_low,
                "pearson_r_ci_95_high": ci_high,
                "pearson_p_value": pearson_p,
                "spearman_rho": spearman_rho,
                "spearman_p_value": spearman_p,
                "note": "Exploratory association; no causal interpretation.",
            }
        )
    result = pd.DataFrame(rows)
    result["pearson_p_value_holm_7"] = holm_adjust(result["pearson_p_value"])
    result["spearman_p_value_holm_7"] = holm_adjust(result["spearman_p_value"])
    return result


# ---------------------------------------------------------------------------
# Nature-style plotting layer
# ---------------------------------------------------------------------------


def export_figure(figure: plt.Figure, path_stem: Path) -> list[str]:
    """Export one fixed layout as editable vectors and high-resolution rasters."""
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs = {"facecolor": "white", "bbox_inches": "tight", "pad_inches": 0.04}
    svg_path = path_stem.with_suffix(".svg")
    pdf_path = path_stem.with_suffix(".pdf")
    tiff_path = path_stem.with_suffix(".tiff")
    png_path = path_stem.with_suffix(".png")
    figure.savefig(svg_path, **save_kwargs)
    figure.savefig(pdf_path, **save_kwargs)
    figure.savefig(tiff_path, dpi=600, pil_kwargs={"compression": "tiff_lzw"}, **save_kwargs)
    figure.savefig(png_path, dpi=600, **save_kwargs)
    plt.close(figure)
    return [svg_path.name, pdf_path.name, tiff_path.name, png_path.name]


def apply_axis_style(axis: plt.Axes, *, grid_axis: str = "x") -> None:
    """Apply the restrained serif axes used by the reference figure set."""
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)
    axis.spines["left"].set_color(COLORS["dark"])
    axis.spines["bottom"].set_color(COLORS["dark"])
    axis.spines["left"].set_linewidth(0.72)
    axis.spines["bottom"].set_linewidth(0.72)
    axis.tick_params(length=3, width=0.65, color=COLORS["dark"], pad=2.5)
    axis.grid(axis=grid_axis, color=COLORS["light_gray"], linewidth=0.45, zorder=0)
    axis.set_axisbelow(True)


def add_panel_label(axis: plt.Axes, label: str, title: str) -> None:
    """Add a small bold panel letter and a left-aligned serif heading."""
    axis.text(-0.16, 1.025, label, transform=axis.transAxes, fontsize=8.4,
              fontweight="bold", ha="left", va="bottom", color=COLORS["dark"])
    axis.set_title(title, loc="left", pad=4, fontsize=8.2,
                   fontweight="semibold", color=COLORS["dark"])


def comparison_condition_legend_handles() -> list[Line2D]:
    """Return the shared H/G condition legend for the similarity panels."""
    return [
        Line2D([0], [0], color=COLORS["blue"], marker="o", linewidth=1.2,
               markeredgecolor="white", markeredgewidth=0.6, markersize=4.8,
               label="Practitioner-FBA similarity"),
        Line2D([0], [0], color=COLORS["orange"], marker="D", linewidth=1.2,
               markeredgecolor="white", markeredgewidth=0.6, markersize=4.5,
               label="Matched general-purpose AI similarity"),
    ]


def draw_paired_ordinal_key(axis: plt.Axes) -> None:
    """Draw direct, row-specific ordinal keys immediately above Figure 3a."""
    axis.set_axis_off()
    label_x = 0.24
    swatch_x = 0.435
    swatch_width = 0.016
    swatch_height = 0.08
    swatch_step = 0.053
    rows = (
        (0.62, "Practitioner-FBA", ORDINAL_PROPORTION_COLORS),
        (0.35, "Matched general AI", GENERAL_AI_ORDINAL_PROPORTION_COLORS),
    )
    for y_position, label, color_map in rows:
        axis.text(label_x, y_position, label, transform=axis.transAxes,
                  ha="left", va="center", fontsize=5.0, color=COLORS["dark"])
        for index, rating in enumerate(RATING_LEVELS):
            x_position = swatch_x + index * swatch_step
            axis.add_patch(Rectangle(
                (x_position, y_position - swatch_height / 2),
                swatch_width, swatch_height,
                transform=axis.transAxes, facecolor=color_map[rating],
                edgecolor="white", linewidth=0.45, clip_on=False,
            ))
            axis.text(x_position + swatch_width + 0.004, y_position, str(rating),
                      transform=axis.transAxes, ha="left", va="center",
                      fontsize=5.0, color=COLORS["dark"])


def draw_rating_key(axis: plt.Axes) -> None:
    """Draw the compact single-row ordinal key above Figure 1a."""
    axis.set_axis_off()
    y_position = 0.50
    label_x = 0.28
    swatch_x = 0.405
    swatch_width = 0.018
    swatch_height = 0.30
    swatch_step = 0.065
    axis.text(label_x, y_position, "Rating", transform=axis.transAxes,
              ha="left", va="center", fontsize=5.0, color=COLORS["dark"])
    for index, rating in enumerate(RATING_LEVELS):
        x_position = swatch_x + index * swatch_step
        axis.add_patch(Rectangle(
            (x_position, y_position - swatch_height / 2),
            swatch_width, swatch_height,
            transform=axis.transAxes, facecolor=ORDINAL_PROPORTION_COLORS[rating],
            edgecolor="white", linewidth=0.4, clip_on=False,
        ))
        axis.text(x_position + swatch_width + 0.006, y_position, str(rating),
                  transform=axis.transAxes, ha="left", va="center",
                  fontsize=5.0, color=COLORS["dark"])


def plot_teacher_endorsement(summary: pd.DataFrame, distribution: pd.DataFrame, output_dir: Path) -> list[str]:
    """Show the teacher-rating distribution as the hero panel with two summaries."""
    labels = [DIMENSION_LABELS[item] for item in DIMENSIONS]
    short_labels = [DIMENSION_SHORT_LABELS[item] for item in DIMENSIONS]
    ordered = summary.set_index("dimension").loc[labels]
    figure = plt.figure(figsize=(7.2, 4.85))
    grid = figure.add_gridspec(
        2, 2, width_ratios=(1.30, 0.90), height_ratios=(1, 1),
        left=0.215, right=0.94, top=0.90, bottom=0.11,
        hspace=0.48, wspace=0.36,
    )
    axis_a = figure.add_subplot(grid[:, 0])
    axis_a_bounds = axis_a.get_position()
    legend_axis = figure.add_axes([
        axis_a_bounds.x0, 0.95, axis_a_bounds.width, 0.035,
    ])
    axis_c = figure.add_subplot(grid[0, 1], sharey=axis_a)
    axis_b = figure.add_subplot(grid[1, 1], sharey=axis_a)
    y_positions = np.arange(len(labels))[::-1]

    stacked = distribution.pivot(index="dimension", columns="rating", values="proportion").reindex(labels)
    left = np.zeros(len(labels))
    for rating in RATING_LEVELS:
        values = stacked[rating].to_numpy(dtype=float)
        bars = axis_a.barh(
            y_positions, values, left=left, height=0.60,
            color=ORDINAL_PROPORTION_COLORS[rating], edgecolor="white", linewidth=0.8,
            label=str(rating), zorder=2,
        )
        text_color = "white" if rating in (4, 5) else COLORS["dark"]
        for bar, start, width in zip(bars, left, values):
            if width >= 0.15:
                axis_a.text(start + width / 2, bar.get_y() + bar.get_height() / 2,
                            f"{width:.0%}", ha="center", va="center",
                            fontsize=6.7, color=text_color)
        left += values
    axis_a.set_yticks(y_positions, short_labels)
    axis_a.set_xlim(0, 1)
    axis_a.xaxis.set_major_formatter(PercentFormatter(1.0))
    axis_a.set_xlabel("Share of teacher responses")
    add_panel_label(axis_a, "a", "Teacher ratings across six FBA domains")
    draw_rating_key(legend_axis)
    apply_axis_style(axis_a)

    proportions = ordered["rating_at_least_4_proportion"].to_numpy(dtype=float)
    ci_low = ordered["rating_at_least_4_ci_95_low"].to_numpy(dtype=float)
    ci_high = ordered["rating_at_least_4_ci_95_high"].to_numpy(dtype=float)
    axis_b.hlines(
        y_positions, ci_low, ci_high, color=COLORS["blue"],
        linewidth=1.35, zorder=2,
    )
    axis_b.scatter(proportions, y_positions, color=COLORS["blue"], edgecolor="white",
                   linewidth=0.65, s=27, zorder=3)
    axis_b.set_xlim(0.74, 1.01)
    axis_b.xaxis.set_major_locator(MultipleLocator(0.10))
    axis_b.xaxis.set_major_formatter(PercentFormatter(1.0))
    axis_b.tick_params(labelleft=False)
    axis_b.set_xlabel("P(rating >= 4)")
    add_panel_label(axis_b, "c", "High-rating endorsement")
    apply_axis_style(axis_b)

    means = ordered["mean"].to_numpy(dtype=float)
    standard_deviations = ordered["sd"].to_numpy(dtype=float)
    axis_c.errorbar(
        means, y_positions, xerr=standard_deviations, fmt="o",
        color=COLORS["green"], ecolor=COLORS["green"], capsize=0,
        markeredgecolor="white", markeredgewidth=0.65, markersize=4.9,
        linewidth=1.35, zorder=3,
    )
    axis_c.axvline(THEORETICAL_MIDPOINT, color=COLORS["gray"], linewidth=0.75,
                   linestyle=(0, (3, 2)), zorder=1)
    axis_c.set_xlim(1, 5.35)
    axis_c.xaxis.set_major_locator(MultipleLocator(1))
    axis_c.tick_params(labelleft=False)
    axis_c.set_xlabel("Mean rating +/- SD")
    add_panel_label(axis_c, "b", "Mean rating and variability")
    apply_axis_style(axis_c)

    return export_figure(figure, output_dir / "figure_1_teacher_endorsement_profile")


def plot_delphi_rounds(summary: pd.DataFrame, scale_cvi: pd.DataFrame, concordance: pd.DataFrame, output_dir: Path) -> list[str]:
    """Show the documented expert rounds without implying robust consensus."""
    rounds = ["Round 1", "Round 2", "Arithmetic mean of rounds"]
    selected = summary[summary["score_set"].isin(rounds)].copy()
    labels = [DIMENSION_LABELS[item] for item in DIMENSIONS]
    short_labels = [DIMENSION_SHORT_LABELS[item] for item in DIMENSIONS]
    figure = plt.figure(figsize=(7.2, 5.15))
    grid = figure.add_gridspec(
        2, 2, width_ratios=(1.08, 0.92), height_ratios=(0.95, 1.12),
        left=0.19, right=0.985, top=0.84, bottom=0.11,
        hspace=0.56, wspace=0.20,
    )
    # The two compact descriptive panels sit above a single wide index panel.
    axis_a = figure.add_subplot(grid[0, 0])
    axis_b = figure.add_subplot(grid[0, 1])
    axis_c = figure.add_subplot(grid[1, :])
    y_positions = np.arange(len(labels))[::-1]
    first = selected[selected["score_set"] == "Round 1"].set_index("dimension").loc[labels]
    second = selected[selected["score_set"] == "Round 2"].set_index("dimension").loc[labels]

    offsets = 0.20
    for values, color, marker, offset in [
        (first, COLORS["orange"], "o", -offsets),
        (second, COLORS["blue"], "D", offsets),
    ]:
        means = values["mean"].to_numpy(dtype=float)
        standard_deviations = values["sd"].to_numpy(dtype=float)
        axis_a.errorbar(
            means, y_positions + offset, xerr=standard_deviations, fmt=marker,
            color=color, ecolor=color, capsize=0, markersize=5.0,
            markeredgecolor="white", markeredgewidth=0.65, linewidth=1.35, zorder=3,
        )
    axis_a.axvline(THEORETICAL_MIDPOINT, color=COLORS["gray"], linewidth=0.75,
                   linestyle=(0, (3, 2)), zorder=1)
    axis_a.set_yticks(y_positions, short_labels)
    axis_a.set_ylim(-0.5, len(labels) - 0.5)
    axis_a.set_xlim(1, 5.35)
    axis_a.xaxis.set_major_locator(MultipleLocator(1))
    axis_a.set_xlabel("Mean rating +/- SD")
    add_panel_label(axis_a, "a", "Mean rating and variability")
    apply_axis_style(axis_a)
    figure.legend(
        handles=[
            Line2D([0], [0], color=COLORS["orange"], marker="o", linewidth=1.2,
                   markeredgecolor="white", markeredgewidth=0.6, markersize=4.7,
                   label="Round 1"),
            Line2D([0], [0], color=COLORS["blue"], marker="D", linewidth=1.2,
                   markeredgecolor="white", markeredgewidth=0.6, markersize=4.4,
                   label="Round 2"),
        ],
        loc="upper center", bbox_to_anchor=(0.54, 0.910), ncol=2, frameon=False,
        fontsize=6.0, handlelength=1.1, columnspacing=1.1,
    )

    start_values = first["rating_at_least_4_proportion"].to_numpy(dtype=float)
    end_values = second["rating_at_least_4_proportion"].to_numpy(dtype=float)
    axis_b.hlines(y_positions, np.minimum(start_values, end_values),
                  np.maximum(start_values, end_values), color=COLORS["light_gray"],
                  linewidth=2.6, zorder=1)
    axis_b.scatter(start_values, y_positions, color=COLORS["orange"], marker="o",
                   edgecolor="white", linewidth=0.65, s=31, zorder=3)
    axis_b.scatter(end_values, y_positions, color=COLORS["blue"], marker="D",
                   edgecolor="white", linewidth=0.65, s=31, zorder=3)
    for y_position, start, end in zip(y_positions, start_values, end_values):
        if np.isclose(start, end):
            axis_b.plot([start - 0.018, start + 0.018], [y_position, y_position],
                        color=COLORS["gray"], linewidth=1.35, solid_capstyle="round", zorder=1)
        else:
            axis_b.annotate(
                "", xy=(end, y_position), xytext=(start, y_position),
                arrowprops={
                    "arrowstyle": "-|>", "color": COLORS["gray"], "linewidth": 1.15,
                    "mutation_scale": 8, "shrinkA": 6, "shrinkB": 6,
                }, zorder=1,
            )
    axis_b.set_yticks(y_positions)
    axis_b.set_ylim(-0.5, len(labels) - 0.5)
    axis_b.set_xlim(-0.03, 1.04)
    axis_b.xaxis.set_major_formatter(PercentFormatter(1.0))
    axis_b.tick_params(labelleft=False)
    axis_b.set_xlabel("P(rating >= 4)")
    add_panel_label(axis_b, "b", "Endorsement by round")
    apply_axis_style(axis_b)

    cvi_values = scale_cvi.set_index("score_set").reindex(rounds)
    concordance_display = concordance.set_index("score_set").reindex(rounds)
    positions = np.arange(len(rounds))
    cvi_series = cvi_values["s_cvi_ave"].to_numpy(dtype=float)
    concordance_series = concordance_display["kendalls_w"].to_numpy(dtype=float)
    axis_c.plot(positions, cvi_series, color=COLORS["green"], marker="o",
                markersize=4.4, linewidth=1.5, label="S-CVI/Ave", zorder=3)
    axis_c.plot(positions, concordance_series, color=COLORS["purple"], marker="D",
                markersize=4.0, linewidth=1.5, label="Kendall's W", zorder=3)
    for x_position, cvi, concordance_value in zip(positions, cvi_series, concordance_series):
        cvi_offset = -11 if x_position == 1 else 7
        cvi_va = "top" if x_position == 1 else "bottom"
        axis_c.annotate(f"{cvi:.2f}", (x_position, cvi), xytext=(0, cvi_offset),
                        textcoords="offset points", ha="center", va=cvi_va,
                        fontsize=5.8, color=COLORS["green"],
                        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.2,
                              "alpha": 0.92})
        concordance_offset = 7
        concordance_va = "bottom"
        axis_c.annotate(f"{concordance_value:.2f}", (x_position, concordance_value),
                        xytext=(0, concordance_offset), textcoords="offset points", ha="center",
                        va=concordance_va, fontsize=5.8, color=COLORS["purple"],
                        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.2,
                              "alpha": 0.92})
    axis_c.set_xticks(positions, ["Round 1", "Round 2", "Round\naverage"])
    axis_c.set_xlim(-0.25, 2.25)
    axis_c.set_ylim(0, 1.04)
    axis_c.yaxis.set_major_formatter(PercentFormatter(1.0))
    axis_c.set_ylabel("Index value")
    add_panel_label(axis_c, "c", "Scale index and concordance")
    axis_c.legend(loc="lower right", bbox_to_anchor=(1.0, 1.01), ncol=2,
                  frameon=False, fontsize=5.8, handlelength=1.0,
                  columnspacing=0.8, borderaxespad=0.0)
    apply_axis_style(axis_c, grid_axis="y")

    return export_figure(figure, output_dir / "figure_2_expert_delphi_rounds")


def plot_similarity_profiles(
    study_two: pd.DataFrame,
    study_three: pd.DataFrame,
    study_two_distribution: pd.DataFrame,
    study_three_distribution: pd.DataFrame,
    output_dir: Path,
) -> list[str]:
    """Compare ordinal composition, perception profiles, and endorsement."""
    labels = [DIMENSION_LABELS[item] for item in DIMENSIONS]
    short_labels = [DIMENSION_SHORT_LABELS[item] for item in DIMENSIONS]
    profile_two = study_two.set_index("dimension").loc[labels]
    profile_three = study_three.set_index("dimension").loc[labels]
    figure = plt.figure(figsize=(7.2, 5.65))
    grid = figure.add_gridspec(
        2, 2, width_ratios=(1.30, 0.90), height_ratios=(0.14, 2.0),
        left=0.205, right=0.985, top=0.90, bottom=0.10,
        hspace=0.07, wspace=0.32,
    )
    # Dedicated legend cells keep both panel groups vertically aligned.
    legend_axis_a = figure.add_subplot(grid[0, 0])
    legend_axis_bc = figure.add_subplot(grid[0, 1])
    axis_a = figure.add_subplot(grid[1, 0])
    right_grid = grid[1, 1].subgridspec(2, 1, hspace=0.36)
    axis_b = figure.add_subplot(right_grid[0, 0])
    axis_c = figure.add_subplot(right_grid[1, 0])
    y_positions = np.arange(len(labels))[::-1]
    offset = 0.18

    distribution_two = (study_two_distribution.pivot(index="dimension", columns="rating", values="proportion")
                        .reindex(labels).fillna(0.0).reindex(columns=RATING_LEVELS))
    distribution_three = (study_three_distribution.pivot(index="dimension", columns="rating", values="proportion")
                          .reindex(labels).fillna(0.0).reindex(columns=RATING_LEVELS))

    # Panel a: Figure-1-style horizontal stacks, paired by comparison condition.
    for index, y_position in enumerate(y_positions):
        if index % 2 == 0:
            axis_a.axhspan(y_position - 0.48, y_position + 0.48,
                           color=COLORS["pale_gray"], alpha=0.42, zorder=-3)
    for values, color_map, row_offset, label_color in [
        (distribution_two.to_numpy(dtype=float), ORDINAL_PROPORTION_COLORS, offset, "white"),
        (distribution_three.to_numpy(dtype=float), GENERAL_AI_ORDINAL_PROPORTION_COLORS, -offset, COLORS["dark"]),
    ]:
        # Keep existing readable labels and add the largest non-zero segments
        # until every row has at least three percentage labels when possible.
        label_indices_by_row = []
        for row_index, row_values in enumerate(values):
            label_indices = {
                rating_index
                for rating_index, width in enumerate(row_values)
                if width >= 0.12
            }
            nonzero_indices = [
                rating_index for rating_index, width in enumerate(row_values) if width > 0
            ]
            ranked_indices = sorted(
                nonzero_indices, key=lambda rating_index: row_values[rating_index], reverse=True
            )
            for rating_index in ranked_indices:
                if len(label_indices) >= min(3, len(nonzero_indices)):
                    break
                label_indices.add(rating_index)
            label_indices_by_row.append(label_indices)
        left = np.zeros(len(labels))
        for rating in RATING_LEVELS:
            widths = values[:, rating - 1]
            bars = axis_a.barh(
                y_positions + row_offset, widths, left=left, height=0.28,
                color=color_map[rating], edgecolor="white",
                linewidth=0.6, zorder=2,
            )
            for row_index, (bar, start, width) in enumerate(zip(bars, left, widths)):
                if (rating - 1) in label_indices_by_row[row_index]:
                    axis_a.text(start + width / 2, bar.get_y() + bar.get_height() / 2,
                                f"{width:.0%}", ha="center", va="center",
                                fontsize=5.0, color=label_color)
            left += widths
    axis_a.set_yticks(y_positions, short_labels)
    axis_a.set_ylim(-0.50, len(labels) - 0.34)
    axis_a.set_xlim(0, 1.01)
    axis_a.xaxis.set_major_formatter(PercentFormatter(1.0))
    axis_a.set_xlabel("Share of ordinal responses")
    add_panel_label(axis_a, "a", "Ordinal proportions")
    apply_axis_style(axis_a)

    means_two = profile_two["mean"].to_numpy(dtype=float)
    means_three = profile_three["mean"].to_numpy(dtype=float)
    critical_two = stats.t.ppf(0.975, profile_two["n"].to_numpy(dtype=float) - 1)
    half_two = critical_two * profile_two["sd"].to_numpy(dtype=float) / np.sqrt(profile_two["n"].to_numpy(dtype=float))
    critical_three = stats.t.ppf(0.975, profile_three["n"].to_numpy(dtype=float) - 1)
    half_three = critical_three * profile_three["sd"].to_numpy(dtype=float) / np.sqrt(profile_three["n"].to_numpy(dtype=float))
    n_two = int(profile_two["n"].iloc[0])
    n_three = int(profile_three["n"].iloc[0])

    # Panel b: the original paired mean-profile display.
    for index, y_position in enumerate(y_positions):
        if index % 2 == 0:
            axis_b.axhspan(y_position - 0.48, y_position + 0.48,
                           color=COLORS["pale_gray"], alpha=0.42, zorder=-3)
    axis_b.axvspan(2.85, 3.15, color=COLORS["light_gray"], alpha=0.28, zorder=-2)
    for mean_three, mean_two, y_position in zip(means_three, means_two, y_positions):
        axis_b.plot([mean_three, mean_two], [y_position + offset, y_position - offset],
                    color=COLORS["light_gray"], linewidth=1.4, zorder=1)
    axis_b.errorbar(means_two, y_positions - offset, xerr=half_two, fmt="o",
                    color=COLORS["blue"], ecolor=COLORS["blue"], capsize=0,
                    markeredgecolor="white", markeredgewidth=0.65,
                    markersize=5.1, label=f"Practitioner-FBA similarity (n = {n_two})", zorder=3)
    axis_b.errorbar(means_three, y_positions + offset, xerr=half_three, fmt="D",
                    color=COLORS["orange"], ecolor=COLORS["orange"], capsize=0,
                    markeredgecolor="white", markeredgewidth=0.65,
                    markersize=4.8, label=f"Matched general-purpose AI similarity (n = {n_three})", zorder=3)
    axis_b.axvline(THEORETICAL_MIDPOINT, color=COLORS["gray"], linestyle=(0, (3, 2)), linewidth=0.8, zorder=1)
    axis_b.set_yticks(y_positions, short_labels)
    axis_b.set_ylim(-0.50, len(labels) - 0.34)
    axis_b.set_xlim(1, 5.05)
    axis_b.xaxis.set_major_locator(MultipleLocator(1))
    axis_b.set_xlabel("Mean similarity rating")
    add_panel_label(axis_b, "b", "Similarity profiles")
    apply_axis_style(axis_b)

    # Panel c: the original high-rating probability display.
    for index, y_position in enumerate(y_positions):
        if index % 2 == 0:
            axis_c.axhspan(y_position - 0.48, y_position + 0.48,
                           color=COLORS["pale_gray"], alpha=0.42, zorder=-3)
    proportions_two = profile_two["rating_at_least_4_proportion"].to_numpy(dtype=float)
    lower_two = profile_two["rating_at_least_4_ci_95_low"].to_numpy(dtype=float)
    upper_two = profile_two["rating_at_least_4_ci_95_high"].to_numpy(dtype=float)
    proportions_three = profile_three["rating_at_least_4_proportion"].to_numpy(dtype=float)
    lower_three = profile_three["rating_at_least_4_ci_95_low"].to_numpy(dtype=float)
    upper_three = profile_three["rating_at_least_4_ci_95_high"].to_numpy(dtype=float)
    axis_c.hlines(y_positions, proportions_three, proportions_two,
                  color=COLORS["light_gray"], linewidth=1.8, zorder=1)
    axis_c.hlines(y_positions - offset, lower_two, upper_two,
                  color=COLORS["blue"], linewidth=1.45, zorder=2)
    axis_c.scatter(proportions_two, y_positions - offset, color=COLORS["blue"], marker="o",
                   edgecolor="white", linewidth=0.65, s=25, zorder=3)
    axis_c.hlines(y_positions + offset, lower_three, upper_three,
                  color=COLORS["orange"], linewidth=1.45, zorder=2)
    axis_c.scatter(proportions_three, y_positions + offset, color=COLORS["orange"], marker="D",
                   edgecolor="white", linewidth=0.65, s=24, zorder=3)
    axis_c.axvline(0.50, color=COLORS["gray"], linestyle=(0, (3, 2)), linewidth=0.7, zorder=1)
    axis_c.set_yticks(y_positions, short_labels)
    axis_c.set_ylim(-0.50, len(labels) - 0.34)
    axis_c.set_xlim(0, 1.03)
    axis_c.xaxis.set_major_formatter(PercentFormatter(1.0))
    axis_c.set_xlabel("P(rating >= 4)")
    add_panel_label(axis_c, "c", "High-rating probability")
    apply_axis_style(axis_c)
    draw_paired_ordinal_key(legend_axis_a)
    legend_axis_bc.legend(
        handles=comparison_condition_legend_handles(),
        loc="center", ncol=2, frameon=False,
        fontsize=5.0, handlelength=0.90, handletextpad=0.25, columnspacing=0.65,
        borderaxespad=0.0,
    )
    legend_axis_bc.set_axis_off()
    return export_figure(figure, output_dir / "figure_3_teacher_similarity_profiles")


def plot_associations(study_two: pd.DataFrame, study_three: pd.DataFrame, output_dir: Path) -> list[str]:
    """Display the exploratory association families as vertically aligned forests."""
    order = ["Composite mean", *[DIMENSION_LABELS[item] for item in DIMENSIONS]]
    short_labels = ["Composite", *[DIMENSION_SHORT_LABELS[item] for item in DIMENSIONS]]
    figure = plt.figure(figsize=(7.2, 5.65))
    grid = figure.add_gridspec(
        2, 1, height_ratios=(1, 1),
        left=0.19, right=0.985, top=0.86, bottom=0.10,
        hspace=0.32,
    )
    axis_a = figure.add_subplot(grid[0, 0])
    axis_b = figure.add_subplot(grid[1, 0])
    y_positions = np.arange(len(order))[::-1]

    def draw_forest(axis: plt.Axes, frame: pd.DataFrame, color: str,
                    title: str, panel: str) -> None:
        subset = frame.set_index("dimension").loc[order]
        pearson = subset["pearson_r"].to_numpy(dtype=float)
        lower = subset["pearson_r_ci_95_low"].to_numpy(dtype=float)
        upper = subset["pearson_r_ci_95_high"].to_numpy(dtype=float)
        spearman = subset["spearman_rho"].to_numpy(dtype=float)
        pearson_significant = (
            np.isfinite(subset["pearson_p_value_holm_7"].to_numpy(dtype=float))
            & (subset["pearson_p_value_holm_7"].to_numpy(dtype=float) < 0.05)
        )
        spearman_significant = (
            np.isfinite(subset["spearman_p_value_holm_7"].to_numpy(dtype=float))
            & (subset["spearman_p_value_holm_7"].to_numpy(dtype=float) < 0.05)
        )
        pearson_y = y_positions + 0.13
        spearman_y = y_positions - 0.13
        for index, y_position in enumerate(y_positions):
            if index % 2 == 0:
                axis.axhspan(y_position - 0.48, y_position + 0.48,
                             color=COLORS["pale_gray"], alpha=0.42, zorder=-3)
        axis.hlines(pearson_y, lower, upper, color=color, linewidth=1.35, zorder=2)
        axis.scatter(pearson, pearson_y, color=color, edgecolor="white", linewidth=0.6,
                     s=26, marker="o", zorder=3)
        axis.scatter(spearman, spearman_y, color=color, edgecolor="white", linewidth=0.6,
                     s=24, marker="D", zorder=3)
        axis.scatter(pearson[pearson_significant], pearson_y[pearson_significant],
                     facecolors="none", edgecolors=COLORS["dark"], linewidths=1.05,
                     s=64, marker="o", zorder=4)
        axis.scatter(spearman[spearman_significant], spearman_y[spearman_significant],
                     facecolors="none", edgecolors=COLORS["dark"], linewidths=1.05,
                     s=60, marker="D", zorder=4)
        axis.axvline(0, color=COLORS["gray"], linewidth=0.8,
                     linestyle=(0, (3, 2)), zorder=1)
        axis.set_xlim(-0.66, 0.76)
        axis.xaxis.set_major_locator(MultipleLocator(0.20))
        axis.set_xlabel("Correlation coefficient")
        axis.set_yticks(y_positions, short_labels)
        axis.set_ylim(-0.55, len(order) - 0.45)
        add_panel_label(axis, panel, title)
        apply_axis_style(axis)

    draw_forest(axis_a, study_two, COLORS["blue"], "Practitioner-FBA similarity", "a")
    draw_forest(axis_b, study_three, COLORS["orange"], "Matched general-purpose AI similarity", "b")
    axis_a.set_ylabel("Rating domain")
    axis_b.set_ylabel("Rating domain")

    figure.legend(
        handles=[
            Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["gray"],
                   markeredgecolor="white", markersize=4.8, label="Pearson r (95% CI)"),
            Line2D([0], [0], marker="D", color="none", markerfacecolor=COLORS["gray"],
                   markeredgecolor="white", markersize=4.6, label="Spearman rho"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor="none",
                   markeredgecolor=COLORS["dark"], markeredgewidth=1.05, markersize=6.6,
                   label="Outline: Holm-adjusted p < .05"),
        ], loc="upper center", bbox_to_anchor=(0.58, 0.930), frameon=False,
        ncol=3, fontsize=5.8, handletextpad=0.35, columnspacing=0.75,
    )
    return export_figure(figure, output_dir / "figure_4_exploratory_associations")


def make_figure_readme(figure_dir: Path) -> None:
    """Write concise figure-level context and accessible text alternatives."""
    content = """# Submission Figures

The PNG files are 600 dpi raster review images. Matching TIFF files are
LZW-compressed, 600 dpi camera-ready rasters; PDF and SVG files are opaque,
editable-vector-oriented exports for manuscript production. Every figure is
regenerated by `code/run_submission_analysis.py`; do not edit a graphic and then
reuse it as if it were code-generated.

| Figure | Main content | Accessible summary |
| --- | --- | --- |
| `figure_1_teacher_endorsement_profile` | Report quality evaluation | Six quality domains show their full 1-5 distributions, mean +/- SD, plus P(rating >= 4) for 121 complete teacher records. |
| `figure_2_expert_delphi_rounds` | Documented Round 1, Round 2, and round-average expert data | Panels a and b occupy the top row; panel c is a wide index/concordance plot below. Four paired expert records are shown descriptively, without a robust-consensus claim. |
| `figure_3_teacher_similarity_profiles` | Perceived-similarity profiles | A large ordinal-proportion panel is paired with mean profiles and high-rating probabilities for the practitioner-referenced similarity evaluation and matched general-purpose AI comparison. |
| `figure_4_exploratory_associations` | Quality-similarity associations | Two vertically aligned Pearson/Spearman forest plots display the exploratory seven-correlation families. |

The figures are public aggregate outputs generated from the anonymous rating
files included in this package.
"""
    (figure_dir / "README.md").write_text(content, encoding="utf-8")


def make_results_readme(output_dir: Path) -> None:
    """Write a root-level guide to the regenerated submission outputs."""
    content = """# Submission Results

All contents of this directory are regenerated by
`code/run_submission_analysis.py`. `run_manifest.json` records sample counts,
methods, input filenames, and generated figure names for this run.

| Directory | Content |
| --- | --- |
| `report_quality_evaluation/` | Teacher-perceived report quality distributions, expert two-round Delphi summaries, expert content-validity indices, and reliability/variation tables. |
| `practitioner_referenced_similarity_evaluation/` | Teacher ratings of perceived similarity to practitioner-conducted FBA. |
| `matched_general_purpose_ai_comparison/` | Teacher ratings of perceived similarity to the matched general-purpose AI condition. |
| `figures/` | PNG, TIFF, PDF, SVG exports and figure-level accessible summaries. |

The results are public aggregate outputs. The analyses concern teacher
perceptions and expert ratings. Diagnostic accuracy, student outcomes, and
professional decision making require separate evaluation designs.

Rating distributions retain every observed value. Expert overall ratings include
recorded decimal values and are not rounded into integer categories.

The practitioner-referenced similarity evaluation and matched general-purpose AI comparison use perceived-similarity ratings.
These workflow-level measures provide a basis for teacher-perception evidence;
concurrent and discriminant validity can be examined in dedicated future designs.
"""
    (output_dir / "README.md").write_text(content, encoding="utf-8")


def run(data_dir: Path, output_dir: Path, overwrite: bool) -> dict[str, object]:
    """Run all versioned analyses and write tables, figures, and provenance."""
    teachers, final_experts, delphi_rounds = load_and_validate_submission_inputs(data_dir)
    n_before_complete_case_filter = int(len(teachers))
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Output directory is not empty: {output_dir}. Use --overwrite to refresh generated files."
            )
        package_results = (PACKAGE_ROOT / "results").resolve()
        if output_dir.resolve() != package_results:
            raise ValueError(
                "For safety, --overwrite can only refresh this package's default results directory."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    configure_matplotlib()

    quality_columns = [f"quality_{dimension}" for dimension in DIMENSIONS]
    human_columns = [f"human_fba_similarity_{dimension}" for dimension in DIMENSIONS]
    general_ai_columns = [f"general_ai_similarity_{dimension}" for dimension in DIMENSIONS]

    study_one_teachers = teachers.dropna(subset=quality_columns, how="any").copy()
    n_after_complete_case_filter = int(len(study_one_teachers))
    study_one_summary, study_one_distribution = ordinal_summary(
        study_one_teachers,
        quality_columns,
        study="report_quality_evaluation",
        sample="Teachers",
        measure="Teacher-perceived endorsement proportions",
    )

    round_frames = {
        "round_1": delphi_rounds[delphi_rounds["delphi_round"] == "round_1"].copy(),
        "round_2": delphi_rounds[delphi_rounds["delphi_round"] == "round_2"].copy(),
        "round_average": final_experts.copy(),
    }
    expert_summaries: list[pd.DataFrame] = []
    expert_distributions: list[pd.DataFrame] = []
    concordance_rows: list[dict[str, object]] = []
    for score_set, frame in round_frames.items():
        summary, distribution = ordinal_summary(
            frame,
            quality_columns,
            study="report_quality_evaluation",
            sample="Experts",
            measure="Expert quality ratings",
            score_set=ROUND_LABELS[score_set],
        )
        expert_summaries.append(summary)
        expert_distributions.append(distribution)
        concordance_rows.append({"score_set": ROUND_LABELS[score_set], **kendalls_w(frame, quality_columns)})
    expert_summary = pd.concat(expert_summaries, ignore_index=True)
    expert_distribution = pd.concat(expert_distributions, ignore_index=True)
    expert_cvi, expert_scale_cvi = expert_cvi_table(round_frames, quality_columns)
    expert_concordance = pd.DataFrame(concordance_rows)
    expert_change = expert_round_change(round_frames["round_1"], round_frames["round_2"], quality_columns)

    write_csv(study_one_summary, output_dir / "report_quality_evaluation" / "table_1_teacher_ordinal_summary.csv")
    write_csv(study_one_distribution, output_dir / "report_quality_evaluation" / "table_2_teacher_rating_distribution.csv")
    write_csv(expert_summary, output_dir / "report_quality_evaluation" / "table_3_expert_delphi_ordinal_summary.csv")
    write_csv(expert_distribution, output_dir / "report_quality_evaluation" / "table_4_expert_delphi_rating_distribution.csv")
    write_csv(expert_cvi, output_dir / "report_quality_evaluation" / "table_5_expert_delphi_item_content_validity.csv")
    write_csv(expert_scale_cvi, output_dir / "report_quality_evaluation" / "table_5s_expert_delphi_scale_content_validity.csv")
    write_csv(expert_concordance, output_dir / "report_quality_evaluation" / "table_6_expert_delphi_kendalls_w.csv")
    write_csv(expert_change, output_dir / "report_quality_evaluation" / "table_7_expert_round_change.csv")
    write_csv(
        corrected_item_total_correlations(study_one_teachers[quality_columns], "Teacher quality ratings"),
        output_dir / "report_quality_evaluation" / "table_8_teacher_internal_consistency_citc.csv",
    )
    write_csv(
        coefficient_of_variation(study_one_teachers[quality_columns], "Teacher quality ratings"),
        output_dir / "report_quality_evaluation" / "table_9_teacher_coefficient_of_variation.csv",
    )
    write_csv(
        coefficient_of_variation(final_experts[quality_columns], "Expert final arithmetic-average ratings"),
        output_dir / "report_quality_evaluation" / "table_9e_expert_final_average_coefficient_of_variation.csv",
    )

    study_two_teachers = teachers[teachers[human_columns].notna().all(axis=1)].copy()
    study_two_quality = study_two_teachers.dropna(subset=quality_columns, how="any")
    study_two_human = study_two_teachers.dropna(subset=human_columns, how="any")
    study_two_quality_summary, study_two_quality_distribution = ordinal_summary(
        study_two_quality,
        quality_columns,
        study="practitioner_referenced_similarity_evaluation",
        sample="Practitioner-referenced teacher records",
        measure="Teacher-perceived quality ratings",
    )
    study_two_human_summary, study_two_human_distribution = ordinal_summary(
        study_two_human,
        human_columns,
        study="practitioner_referenced_similarity_evaluation",
        sample="Practitioner-referenced teacher records",
        measure="Perceived similarity to human-conducted FBA",
    )
    study_two_tests = one_sample_tests(
        study_two_human,
        human_columns,
        study="practitioner_referenced_similarity_evaluation",
        sample="Practitioner-referenced teacher records",
        measure="Perceived similarity to human-conducted FBA",
        alternative="greater",
    )
    study_two_anova = repeated_measures_summary(
        study_two_human,
        human_columns,
        study="practitioner_referenced_similarity_evaluation",
        sample="Practitioner-referenced teacher records",
        measure="Perceived similarity to human-conducted FBA",
    )
    study_two_pairwise = paired_dimension_tests(
        study_two_human,
        human_columns,
        study="practitioner_referenced_similarity_evaluation",
        sample="Practitioner-referenced teacher records",
        measure="Perceived similarity to human-conducted FBA",
    )
    study_two_associations = association_table(
        study_two_teachers[quality_columns],
        study_two_teachers[human_columns],
        quality_columns,
        human_columns,
        study="practitioner_referenced_similarity_evaluation",
        sample="Practitioner-referenced teacher records",
        similarity_measure="Perceived similarity to human-conducted FBA",
    )
    study_two_dir = output_dir / "practitioner_referenced_similarity_evaluation"
    write_csv(study_two_quality_summary, study_two_dir / "table_1a_quality_ordinal_summary.csv")
    write_csv(study_two_quality_distribution, study_two_dir / "table_1as_quality_rating_distribution.csv")
    write_csv(study_two_human_summary, study_two_dir / "table_1b_practitioner_fba_similarity_ordinal_summary.csv")
    write_csv(study_two_human_distribution, study_two_dir / "table_1bs_practitioner_fba_similarity_distribution.csv")
    write_csv(
        pd.concat(
            [
                corrected_item_total_correlations(study_two_quality[quality_columns], "Teacher quality ratings"),
                corrected_item_total_correlations(study_two_human[human_columns], "Practitioner-FBA similarity ratings"),
            ],
            ignore_index=True,
        ),
        study_two_dir / "table_2_internal_consistency_citc.csv",
    )
    write_csv(
        pd.concat(
            [
                coefficient_of_variation(study_two_teachers[quality_columns], "Teacher quality ratings"),
                coefficient_of_variation(study_two_teachers[human_columns], "Practitioner-FBA similarity ratings"),
            ],
            ignore_index=True,
        ),
        study_two_dir / "table_3_coefficient_of_variation.csv",
    )
    write_csv(study_two_tests, study_two_dir / "table_4_one_sample_t_tests_holm.csv")
    write_csv(study_two_anova, study_two_dir / "table_5_repeated_measures_anova_and_friedman.csv")
    write_csv(study_two_pairwise, study_two_dir / "table_5s_pairwise_paired_tests_holm.csv")
    write_csv(study_two_associations, study_two_dir / "table_6_exploratory_associations.csv")

    study_three_teachers = teachers[teachers[general_ai_columns].notna().all(axis=1)].copy()
    study_three_quality = study_three_teachers.dropna(subset=quality_columns, how="any")
    study_three_general_ai = study_three_teachers.dropna(subset=general_ai_columns, how="any")
    study_three_quality_summary, study_three_quality_distribution = ordinal_summary(
        study_three_quality,
        quality_columns,
        study="matched_general_purpose_ai_comparison",
        sample="Teachers",
        measure="Teacher-perceived quality ratings",
    )
    study_three_general_ai_summary, study_three_general_ai_distribution = ordinal_summary(
        study_three_general_ai,
        general_ai_columns,
        study="matched_general_purpose_ai_comparison",
        sample="Teachers",
        measure="Perceived similarity to general-purpose AI",
    )
    study_three_tests = one_sample_tests(
        study_three_general_ai,
        general_ai_columns,
        study="matched_general_purpose_ai_comparison",
        sample="Teachers",
        measure="Perceived similarity to general-purpose AI",
        alternative="less",
    )
    study_three_anova = repeated_measures_summary(
        study_three_general_ai,
        general_ai_columns,
        study="matched_general_purpose_ai_comparison",
        sample="Teachers",
        measure="Perceived similarity to general-purpose AI",
    )
    study_three_pairwise = paired_dimension_tests(
        study_three_general_ai,
        general_ai_columns,
        study="matched_general_purpose_ai_comparison",
        sample="Teachers",
        measure="Perceived similarity to general-purpose AI",
    )
    study_three_associations = association_table(
        study_three_teachers[quality_columns],
        study_three_teachers[general_ai_columns],
        quality_columns,
        general_ai_columns,
        study="matched_general_purpose_ai_comparison",
        sample="Teachers",
        similarity_measure="Perceived similarity to general-purpose AI",
    )
    study_three_dir = output_dir / "matched_general_purpose_ai_comparison"
    write_csv(study_three_quality_summary, study_three_dir / "table_1a_quality_ordinal_summary.csv")
    write_csv(study_three_quality_distribution, study_three_dir / "table_1as_quality_rating_distribution.csv")
    write_csv(study_three_general_ai_summary, study_three_dir / "table_1b_matched_general_purpose_ai_similarity_ordinal_summary.csv")
    write_csv(study_three_general_ai_distribution, study_three_dir / "table_1bs_matched_general_purpose_ai_similarity_distribution.csv")
    write_csv(
        pd.concat(
            [
                corrected_item_total_correlations(study_three_quality[quality_columns], "Teacher quality ratings"),
                corrected_item_total_correlations(study_three_general_ai[general_ai_columns], "Matched general-purpose AI similarity ratings"),
            ],
            ignore_index=True,
        ),
        study_three_dir / "table_2_internal_consistency_citc.csv",
    )
    write_csv(
        pd.concat(
            [
                coefficient_of_variation(study_three_teachers[quality_columns], "Teacher quality ratings"),
                coefficient_of_variation(study_three_teachers[general_ai_columns], "Matched general-purpose AI similarity ratings"),
            ],
            ignore_index=True,
        ),
        study_three_dir / "table_3_coefficient_of_variation.csv",
    )
    write_csv(study_three_tests, study_three_dir / "table_4_one_sample_t_tests_holm.csv")
    write_csv(study_three_anova, study_three_dir / "table_5_repeated_measures_anova_and_friedman.csv")
    write_csv(study_three_pairwise, study_three_dir / "table_5s_pairwise_paired_tests_holm.csv")
    write_csv(study_three_associations, study_three_dir / "table_6_exploratory_associations.csv")

    figure_exports = {
        "figure_1_teacher_endorsement_profile": plot_teacher_endorsement(study_one_summary, study_one_distribution, figure_dir),
        "figure_2_expert_delphi_rounds": plot_delphi_rounds(expert_summary, expert_scale_cvi, expert_concordance, figure_dir),
        "figure_3_teacher_similarity_profiles": plot_similarity_profiles(
            study_two_human_summary,
            study_three_general_ai_summary,
            study_two_human_distribution,
            study_three_general_ai_distribution,
            figure_dir,
        ),
        "figure_4_exploratory_associations": plot_associations(
            study_two_associations, study_three_associations, figure_dir
        ),
    }
    make_figure_readme(figure_dir)
    make_results_readme(output_dir)

    manifest = {
        "created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "analysis_entry_point": "code/run_submission_analysis.py",
        "input_files": [
            "data/public_ratings/teacher_ratings_deidentified.csv",
            "data/public_ratings/expert_ratings_deidentified.csv",
            "data/public_ratings/expert_delphi_rounds_deidentified.csv",
        ],
        "sample_counts": {
            "teacher_rows": n_before_complete_case_filter,
            "report_quality_complete_teacher_records": n_after_complete_case_filter,
            "expert_records_per_delphi_round": int(len(round_frames["round_1"])),
            "practitioner_similarity_records": int(len(study_two_human)),
            "practitioner_similarity_quality_records": int(len(study_two_quality)),
            "general_ai_comparison_records": int(len(study_three_general_ai)),
        },
        "public_data_design": {
            "teacher_records": "137 anonymous teacher records numbered P001 through P137 with rating variables only.",
            "practitioner_referenced_cohort": "75 distinct teachers identified by complete six-dimension practitioner-FBA similarity ratings.",
            "matched_comparison_cohort": "62 teachers identified by complete six-dimension general-purpose-AI similarity ratings.",
            "expert_records": "Four anonymous expert records across two paired rounds plus arithmetic-average final scores.",
            "privacy_fields_removed": "Collection mode, original identifiers, names, teaching stage, reported grade, and other source metadata are absent from the public files.",
            "source_reconciliation_correction": "One overall general-purpose-AI similarity response was corrected from 4 to 3 before public release after comparison with the source response label.",
        },
        "public_labels": {
            "report_quality_evaluation": "Teacher-Perceived Endorsement Proportions",
            "matched_general_purpose_ai_comparison_sample": "Teachers",
        },
        "methods": {
            "ordinal_descriptives": "Complete recorded-value distribution, median (IQR), P(rating >= 4), Wilson 95% CI; decimal expert overall ratings are preserved",
            "multiple_testing": "Holm adjustment within each six-test or seven-correlation family",
            "ordinal_sensitivity": "Friedman omnibus and Spearman correlations",
            "expert_round_rule": "Final expert scores equal the arithmetic mean of paired Round 1 and Round 2 ratings.",
            "comparison_measure_interpretation": "The practitioner-referenced similarity evaluation and matched general-purpose AI comparison use perceived-similarity ratings; concurrent and discriminant validity require dedicated future designs.",
        },
        "figures": figure_exports,
        "scope_note": "The analyses estimate teacher-perception and expert-rating quantities; diagnostic accuracy, student outcomes, professional equivalence, and autonomous decision making are separate evaluation targets.",
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true", help="Allow refresh of a nonempty generated output directory.")
    args = parser.parse_args()
    manifest = run(args.data_dir.resolve(), args.output_dir.resolve(), args.overwrite)
    from build_consolidated_supplement import main as build_consolidated_supplement

    build_consolidated_supplement()
    print(f"Generated submission analyses and {len(manifest['figures'])} figures in {args.output_dir.resolve()}.")


if __name__ == "__main__":
    main()
