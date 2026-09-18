#!/usr/bin/env python3
"""Build six publication-facing supplementary tables without changing figures.

The script reorganizes validated aggregate outputs and anonymous public rating
matrices. It does not alter source analyses.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DATA = ROOT / "data" / "public_ratings"
OUTPUT = RESULTS / "consolidated_supplement"

DIMENSIONS = [
    "Antecedent identification",
    "Consequence identification",
    "Setting-event identification",
    "Core function hypothesis",
    "Replacement-behavior recommendation",
    "Overall rating",
]
QUALITY_COLUMNS = [
    "quality_antecedent_identification",
    "quality_consequence_identification",
    "quality_setting_event_identification",
    "quality_core_function_hypothesis",
    "quality_replacement_behavior_recommendation",
    "quality_overall_rating",
]
HUMAN_COLUMNS = [column.replace("quality_", "human_fba_similarity_") for column in QUALITY_COLUMNS]
GENERAL_COLUMNS = [column.replace("quality_", "general_ai_similarity_") for column in QUALITY_COLUMNS]


def read_result(relative: str) -> pd.DataFrame:
    return pd.read_csv(RESULTS / relative)


def write(frame: pd.DataFrame, name: str) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT / name, index=False, encoding="utf-8", lineterminator="\n")


def holm(values: pd.Series) -> pd.Series:
    raw = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    adjusted = np.full(raw.shape, np.nan)
    finite = np.flatnonzero(np.isfinite(raw))
    if not len(finite):
        return pd.Series(adjusted, index=values.index)
    ordered = finite[np.argsort(raw[finite])]
    running = 0.0
    count = len(ordered)
    for rank, index in enumerate(ordered):
        running = max(running, min(1.0, raw[index] * (count - rank)))
        adjusted[index] = running
    return pd.Series(adjusted, index=values.index)


def table_s1() -> pd.DataFrame:
    rows = [
        ("Workflow module", "Stage and question selection", "Conversation state, completed fields, and remaining targets", "Selects the next FBA information target; unsupported additions are prohibited."),
        ("Workflow module", "Natural-language question generation", "Selected target, context, and rule guidance", "Produces one or two teacher-facing questions using observable, fact-focused wording."),
        ("Workflow module", "Structured recording and scoring", "Teacher response and stage-specific items", "Maps explicit information into basic, setting-event, antecedent, consequence, and positive-strategy fields."),
        ("Workflow module", "Functional-hypothesis generation", "Accumulated record and FBA constraints", "Separates direct evidence from a provisional hypothesis restricted to four functional categories."),
        ("Workflow module", "Function-matched recommendation drafting", "Provisional hypothesis, rule base, and deidentified retrieval support", "Drafts antecedent, replacement-skill, and consequence strategies for human review."),
        ("Human oversight", "Educator and qualified-professional review", "Record, hypothesis, and recommendation draft", "Required before high-stakes interpretation or action."),
        ("Platform", "Workflow export", "Baidu AppBuilder workflow; reported version 3.4.9", "Architecture description only; operational access details are excluded."),
        ("Retrieval", "Knowledge support", "VectorDB and multilingual embedding", "FBA-oriented rules and case-linked retrieval materials were used during the study; knowledge-base contents are withheld for privacy and security."),
        ("Model", "Conversational generation", "DeepSeek-V3 as reported in the study record", "Study-time configuration reported for interpretation of the evaluated system."),
        ("Model", "Structured judgment and analysis", "Qwen3-235B-A22B as reported in the study record", "Study-time configuration reported for interpretation of the evaluated system."),
        ("Process parameter", "Interview stages", "Four stage loops", "Basic/problem definition; life background; antecedent contexts and reactions; social skills and positive strategies."),
        ("Process parameter", "Node inventory", "73 nodes", "Nine LLM, 30 code, nine memory, eight switch, seven text-processing, four merge, three chat, and supporting nodes."),
        ("Process parameter", "Conversation memory", "13 lifecycle variables", "Carries structured state across stages rather than relying on an unstructured transcript alone."),
        ("Prompt contract", "Basic-information extraction", "Teacher reply, previous question, and existing record", "Outputs sex, age, diagnosis-report status if discussed, behavior, trigger context, and frequency."),
        ("Prompt contract", "Question selection", "Filled and unfilled items plus conversation history", "Outputs selected item numbers and a selection rationale."),
        ("Prompt contract", "Stage scoring", "Candidate questions and teacher response", "Outputs item scores, explicit follow-up information, and scoring rationale."),
        ("Prompt contract", "Final feedback", "Basic, background, antecedent/reaction, and positive-strategy records", "Outputs a provisional functional hypothesis and function-matched draft recommendations."),
        ("Functional constraint", "Permitted categories", "Attention; tangible/activity access; escape/avoidance; automatic/sensory reinforcement", "Context-dependent or multiple hypotheses may be retained when uncertainty remains."),
        ("Evidence boundary", "Not implemented or validated", "Direct observation, experimental functional analysis, diagnostic accuracy, and outcome monitoring", "The system supports preparation and documentation, not professional replacement."),
    ]
    return pd.DataFrame(rows, columns=["section", "component", "configuration_or_input", "publication_interpretation"])


QUALITY_DEFINITIONS = {
    "Antecedent identification": "指在行为发生前即刻发生的具体事件或情境。评审系统是否识别可观察的触发事件，而非模糊描述。",
    "Consequence identification": "指在行为发生后即刻发生、可能维持行为的事件。评审系统是否说明行为带来的实际结果。",
    "Setting-event identification": "指较早发生并改变个体对前因反应价值的背景事件。评审系统是否区分设置事件与直接前因。",
    "Core function hypothesis": "评审系统是否依据前因与后果证据，将功能假设限制为获得关注、获得有形物或活动、逃避或回避、自动或感觉强化。",
    "Replacement-behavior recommendation": "评审建议是否与假设功能匹配，并说明由谁、在何时、如何教授功能等价的适应性行为。",
    "Overall rating": "综合评价报告的连贯性、可信度、质量和实际可用性。",
}


def table_s2() -> pd.DataFrame:
    english = {
        "Antecedent identification": "Concrete, observable event or context immediately preceding behavior.",
        "Consequence identification": "Immediate event following behavior that may maintain or reinforce it.",
        "Setting-event identification": "Background event that changes the probability or value of an antecedent without being the immediate trigger.",
        "Core function hypothesis": "Evidence-linked summary of what behavior may obtain or avoid, restricted to the four documented categories.",
        "Replacement-behavior recommendation": "Operational, socially adaptive behavior that is function-equivalent to the problem behavior.",
        "Overall rating": "Coherence, credibility, quality, and practical usefulness of the complete report.",
    }
    rows = []
    for dimension in DIMENSIONS:
        rows.append({
            "evaluation": "Teacher and expert report-quality review",
            "item": dimension,
            "english_item_definition": english[dimension],
            "chinese_source_wording": QUALITY_DEFINITIONS[dimension],
            "scoring": "1=无效；2=较差；3=一般；4=良好；5=优秀",
        })
    for dimension in DIMENSIONS:
        rows.append({
            "evaluation": "Perceived-similarity review",
            "item": "Overall similarity" if dimension == "Overall rating" else dimension,
            "english_item_definition": "Compare the two reports on this FBA domain and describe any substantive difference.",
            "chinese_source_wording": "综合评价两份分析报告在核心要素和结论上的整体相似程度。" if dimension == "Overall rating" else "比较两份报告在该评审要素上的一致性，并简要描述不一致之处。",
            "scoring": "1=完全不一致；2=基本不一致；3=部分一致；4=基本一致；5=完全一致",
        })
    return pd.DataFrame(rows)


def table_s3() -> pd.DataFrame:
    sources = [
        ("Evaluation 1", "Teacher report quality", "report_quality_evaluation/table_2_teacher_rating_distribution.csv", None),
        ("Evaluation 1", "Similarity to practitioner-conducted FBA", "practitioner_referenced_similarity_evaluation/table_1bs_practitioner_fba_similarity_distribution.csv", None),
        ("Evaluation 2", "Expert report quality", "report_quality_evaluation/table_4_expert_delphi_rating_distribution.csv", "Arithmetic mean of rounds"),
        ("Evaluation 3", "Similarity to general-purpose AI", "matched_general_purpose_ai_comparison/table_1bs_matched_general_purpose_ai_similarity_distribution.csv", None),
    ]
    frames = []
    for evaluation, measure, relative, score_set in sources:
        frame = read_result(relative)
        if score_set:
            frame = frame[frame["score_set"] == score_set]
        frame = frame[["score_set", "dimension", "rating", "n", "proportion"]].copy()
        frame.insert(0, "measure", measure)
        frame.insert(0, "evaluation", evaluation)
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    grouped = combined.groupby(["evaluation", "measure", "score_set", "dimension"], dropna=False)
    totals = grouped.agg(distribution_n=("n", "sum"), proportion_sum=("proportion", "sum"))
    invalid = totals[(totals["distribution_n"] <= 0) | ~np.isclose(totals["proportion_sum"], 1.0)]
    if not invalid.empty:
        raise ValueError(f"Incomplete rating distribution detected:\n{invalid}")
    return combined


def alpha(frame: pd.DataFrame) -> float:
    values = frame.apply(pd.to_numeric, errors="coerce").dropna()
    k = values.shape[1]
    return float(k / (k - 1) * (1 - values.var(ddof=1).sum() / values.sum(axis=1).var(ddof=1)))


def scale_summary(data: pd.DataFrame, columns: list[str], label: str) -> dict[str, object]:
    complete = data[columns].apply(pd.to_numeric, errors="coerce").dropna()
    composites = complete.mean(axis=1)
    q1, q3 = composites.quantile([0.25, 0.75])
    return {
        "row_type": "scale_summary", "scale": label, "dimension": "Composite mean",
        "n": len(complete), "number_of_items": len(columns), "cronbach_alpha": alpha(complete),
        "corrected_item_total_r": np.nan, "composite_mean": composites.mean(),
        "composite_sd": composites.std(ddof=1), "composite_median": composites.median(),
        "composite_iqr": q3 - q1,
    }


def item_rows(path: str, source_scale: str, target_scale: str) -> pd.DataFrame:
    frame = read_result(path)
    frame = frame[frame["scale"] == source_scale].copy()
    return pd.DataFrame({
        "row_type": "item_detail", "scale": target_scale, "dimension": frame["dimension"],
        "n": frame["n_listwise_complete"], "number_of_items": frame["number_of_items"],
        "cronbach_alpha": frame["cronbach_alpha"],
        "corrected_item_total_r": frame["corrected_item_total_correlation"],
        "composite_mean": np.nan, "composite_sd": np.nan,
        "composite_median": np.nan, "composite_iqr": np.nan,
    })


def table_s4() -> pd.DataFrame:
    teachers = pd.read_csv(DATA / "teacher_ratings_deidentified.csv")
    experienced = teachers[teachers[HUMAN_COLUMNS].notna().all(axis=1)]
    comparison = teachers[teachers[GENERAL_COLUMNS].notna().all(axis=1)]
    definitions = [
        (teachers, QUALITY_COLUMNS, "Teacher quality ratings", "report_quality_evaluation/table_8_teacher_internal_consistency_citc.csv", "Teacher quality ratings"),
        (experienced, QUALITY_COLUMNS, "Practitioner-reference quality ratings", "practitioner_referenced_similarity_evaluation/table_2_internal_consistency_citc.csv", "Teacher quality ratings"),
        (experienced, HUMAN_COLUMNS, "Practitioner-FBA similarity ratings", "practitioner_referenced_similarity_evaluation/table_2_internal_consistency_citc.csv", "Practitioner-FBA similarity ratings"),
        (comparison, QUALITY_COLUMNS, "Matched-comparison quality ratings", "matched_general_purpose_ai_comparison/table_2_internal_consistency_citc.csv", "Teacher quality ratings"),
        (comparison, GENERAL_COLUMNS, "General-purpose AI similarity ratings", "matched_general_purpose_ai_comparison/table_2_internal_consistency_citc.csv", "Matched general-purpose AI similarity ratings"),
    ]
    frames = []
    for data, columns, label, path, source_scale in definitions:
        frames.append(pd.DataFrame([scale_summary(data, columns, label)]))
        frames.append(item_rows(path, source_scale, label))
    return pd.concat(frames, ignore_index=True)


def normalize_inference(frame: pd.DataFrame, evaluation: str, family: str) -> pd.DataFrame:
    rows = []
    for _, row in frame.iterrows():
        rows.append({
            "evaluation": evaluation,
            "test_family": family,
            "contrast_or_dimension": row.get("dimension", row.get("comparison", "")),
            "n": row.get("n", row.get("n_paired", row.get("n_paired_experts", ""))),
            "statistic": row.get("t_statistic", ""),
            "degrees_of_freedom": row.get("degrees_of_freedom", ""),
            "effect_size": row.get("cohens_d", row.get("cohens_dz", "")),
            "raw_p": row.get("p_value_directional", row.get("p_value_two_sided", row.get("wilcoxon_p_value_two_sided", ""))),
            "holm_p": row.get("p_value_holm_6", row.get("p_value_holm_15", "")),
            "alternative_or_note": row.get("alternative", row.get("note", "")),
        })
    return pd.DataFrame(rows)


def table_s5() -> pd.DataFrame:
    frames = [
        normalize_inference(read_result("practitioner_referenced_similarity_evaluation/table_4_one_sample_t_tests_holm.csv"), "Evaluation 1", "Similarity midpoint tests"),
        normalize_inference(read_result("practitioner_referenced_similarity_evaluation/table_5s_pairwise_paired_tests_holm.csv"), "Evaluation 1", "Similarity pairwise domain tests"),
        normalize_inference(read_result("matched_general_purpose_ai_comparison/table_4_one_sample_t_tests_holm.csv"), "Evaluation 3", "Similarity midpoint tests"),
        normalize_inference(read_result("matched_general_purpose_ai_comparison/table_5s_pairwise_paired_tests_holm.csv"), "Evaluation 3", "Similarity pairwise domain tests"),
    ]
    expert_source = read_result("report_quality_evaluation/table_7_expert_round_change.csv")
    expert_rows = []
    for _, row in expert_source.iterrows():
        expert_rows.append({
            "evaluation": "Evaluation 2",
            "test_family": "Expert round-change tests",
            "contrast_or_dimension": row["dimension"],
            "n": row["n_paired_experts"],
            "statistic": row["mean_round_2_minus_round_1"],
            "degrees_of_freedom": "",
            "effect_size": (
                f"median change={row['median_round_2_minus_round_1']}; "
                f"increased={row['n_increased']}; decreased={row['n_decreased']}; "
                f"unchanged={row['n_unchanged']}"
            ),
            "raw_p": row["wilcoxon_p_value_two_sided"],
            "holm_p": np.nan,
            "alternative_or_note": row["note"],
        })
    expert = pd.DataFrame(expert_rows)
    expert["holm_p"] = holm(pd.to_numeric(expert["raw_p"], errors="coerce"))
    frames.append(expert)
    return pd.concat(frames, ignore_index=True)


def table_s6() -> pd.DataFrame:
    rows = []
    for evaluation, base in [
        ("Evaluation 1", "practitioner_referenced_similarity_evaluation"),
        ("Evaluation 3", "matched_general_purpose_ai_comparison"),
    ]:
        associations = read_result(f"{base}/table_6_exploratory_associations.csv")
        for _, row in associations.iterrows():
            rows.append({
                "evaluation": evaluation, "analysis": "Pearson correlation",
                "target": row["dimension"], "n": row["n_paired"],
                "statistic": row["pearson_r"],
                "interval_or_df": f"[{row['pearson_r_ci_95_low']}, {row['pearson_r_ci_95_high']}]",
                "raw_p": row["pearson_p_value"], "holm_p": row["pearson_p_value_holm_7"],
                "effect_or_note": "Exploratory association",
            })
            rows.append({
                "evaluation": evaluation, "analysis": "Spearman correlation",
                "target": row["dimension"], "n": row["n_paired"],
                "statistic": row["spearman_rho"], "interval_or_df": "",
                "raw_p": row["spearman_p_value"], "holm_p": row["spearman_p_value_holm_7"],
                "effect_or_note": "Ordinal sensitivity",
            })
        sensitivity = read_result(f"{base}/table_5_repeated_measures_anova_and_friedman.csv")
        for _, row in sensitivity.iterrows():
            rows.append({
                "evaluation": evaluation, "analysis": row["analysis"],
                "target": "Six FBA domains", "n": row["n_listwise_complete"],
                "statistic": row["statistic"],
                "interval_or_df": f"df1={row['df_1']}; df2={row['df_2']}",
                "raw_p": row["p_value"], "holm_p": "",
                "effect_or_note": f"{row['effect_size_name']}={row['effect_size']}",
            })
    return pd.DataFrame(rows)


def main() -> None:
    outputs = {
        "table_s1_system_modules_models_prompts_parameters.csv": table_s1(),
        "table_s2_scale_items_chinese_scoring.csv": table_s2(),
        "table_s3_complete_rating_distributions.csv": table_s3(),
        "table_s4_reliability_item_total_composites.csv": table_s4(),
        "table_s5_inferential_tests_raw_holm.csv": table_s5(),
        "table_s6_correlations_sensitivity.csv": table_s6(),
    }
    for name, frame in outputs.items():
        write(frame, name)
    manifest = {
        "generated_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "builder": "code/build_consolidated_supplement.py",
        "figures_modified": False,
        "tables": {name: int(len(frame)) for name, frame in outputs.items()},
        "scope_note": "The tables report teacher-perception ratings and a four-expert review; diagnostic accuracy, professional equivalence, intervention effects, and student outcomes are separate evaluation targets.",
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {len(outputs)} consolidated supplementary tables to {OUTPUT}")


if __name__ == "__main__":
    main()
