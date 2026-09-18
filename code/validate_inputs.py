#!/usr/bin/env python3
"""Validate the anonymous public rating data used by this package."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DIMENSIONS = [
    "antecedent_identification",
    "consequence_identification",
    "setting_event_identification",
    "core_function_hypothesis",
    "replacement_behavior_recommendation",
    "overall_rating",
]

TEACHER_COLUMNS = [
    "participant_id",
    *[f"quality_{dimension}" for dimension in DIMENSIONS],
    *[f"human_fba_similarity_{dimension}" for dimension in DIMENSIONS],
    *[f"general_ai_similarity_{dimension}" for dimension in DIMENSIONS],
]

EXPERT_COLUMNS = [
    "expert_record_id",
    *[f"quality_{dimension}" for dimension in DIMENSIONS],
]

DELPHI_ROUND_COLUMNS = [
    "expert_record_id",
    "delphi_round",
    *[f"quality_{dimension}" for dimension in DIMENSIONS],
]

ALLOWED_DELPHI_ROUNDS = {"round_1", "round_2"}


def _require_columns(frame: pd.DataFrame, expected: list[str], label: str) -> None:
    actual = list(frame.columns)
    if actual != expected:
        missing = [column for column in expected if column not in actual]
        unexpected = [column for column in actual if column not in expected]
        raise ValueError(
            f"{label} columns do not match the documented schema. "
            f"Missing: {missing or 'none'}; unexpected: {unexpected or 'none'}."
        )


def _validate_rating_columns(frame: pd.DataFrame, columns: list[str], label: str) -> None:
    for column in columns:
        numeric = pd.to_numeric(frame[column], errors="coerce")
        invalid_nonmissing = frame[column].notna() & numeric.isna()
        out_of_range = numeric.notna() & ~numeric.between(1, 5)
        if invalid_nonmissing.any() or out_of_range.any():
            raise ValueError(
                f"{label}.{column} contains nonnumeric or out-of-range values; "
                "ratings must be missing or numeric values from 1 through 5."
            )


def validate_teacher_frame(frame: pd.DataFrame, *, require_snapshot_counts: bool = True) -> None:
    _require_columns(frame, TEACHER_COLUMNS, "Teacher data")
    if require_snapshot_counts and len(frame) != 137:
        raise ValueError(f"Teacher data must contain 137 rows for this verified snapshot, found {len(frame)}.")
    if not frame["participant_id"].astype(str).str.fullmatch(r"P\d{3}").all():
        raise ValueError("Teacher participant_id values must use the nonidentifying P001-style format.")
    if frame["participant_id"].duplicated().any():
        raise ValueError("Teacher participant_id values must be unique.")
    rating_columns = TEACHER_COLUMNS[1:]
    _validate_rating_columns(frame, rating_columns, "Teacher data")

    quality_columns = [f"quality_{dimension}" for dimension in DIMENSIONS]
    human_columns = [f"human_fba_similarity_{dimension}" for dimension in DIMENSIONS]
    general_columns = [f"general_ai_similarity_{dimension}" for dimension in DIMENSIONS]
    for label, columns in (
        ("quality", quality_columns),
        ("practitioner-FBA similarity", human_columns),
        ("general-purpose-AI similarity", general_columns),
    ):
        nonmissing = frame[columns].notna().sum(axis=1)
        if (~nonmissing.isin([0, len(columns)])).any():
            raise ValueError(f"Each {label} block must contain either all six ratings or no ratings.")

    human_complete = frame[human_columns].notna().all(axis=1)
    general_complete = frame[general_columns].notna().all(axis=1)
    if (human_complete & general_complete).any():
        raise ValueError("Teacher records may belong to only one similarity evaluation.")
    if (~(human_complete | general_complete)).any():
        raise ValueError("Every teacher record must belong to one similarity evaluation.")
    if require_snapshot_counts:
        quality_complete = int(frame[quality_columns].notna().all(axis=1).sum())
        if quality_complete != 121:
            raise ValueError(f"Expected 121 complete teacher quality records, found {quality_complete}.")
        if int(human_complete.sum()) != 75:
            raise ValueError(f"Expected 75 practitioner-referenced teacher records, found {int(human_complete.sum())}.")
        if int(general_complete.sum()) != 62:
            raise ValueError(f"Expected 62 matched-comparison teacher records, found {int(general_complete.sum())}.")


def validate_expert_frame(frame: pd.DataFrame, *, require_snapshot_counts: bool = True) -> None:
    _require_columns(frame, EXPERT_COLUMNS, "Expert data")
    if require_snapshot_counts and len(frame) != 4:
        raise ValueError(f"Expert data must contain 4 rows for this verified snapshot, found {len(frame)}.")
    if not frame["expert_record_id"].astype(str).str.fullmatch(r"E\d{3}").all():
        raise ValueError("Expert record_id values must use the nonidentifying E001-style format.")
    if frame["expert_record_id"].duplicated().any():
        raise ValueError("Expert record_id values must be unique.")
    _validate_rating_columns(frame, EXPERT_COLUMNS[1:], "Expert data")


def validate_delphi_round_frame(frame: pd.DataFrame, *, require_snapshot_counts: bool = True) -> None:
    """Validate paired, deidentified Round 1 and Round 2 expert ratings."""
    _require_columns(frame, DELPHI_ROUND_COLUMNS, "Delphi-round expert data")
    if require_snapshot_counts and len(frame) != 8:
        raise ValueError(
            "Delphi-round expert data must contain 8 rows for the verified snapshot, "
            f"found {len(frame)}."
        )
    if not frame["expert_record_id"].astype(str).str.fullmatch(r"E\d{3}").all():
        raise ValueError("Delphi-round expert_record_id values must use the nonidentifying E001-style format.")
    if not set(frame["delphi_round"].dropna()).issubset(ALLOWED_DELPHI_ROUNDS):
        raise ValueError("Delphi-round expert data contains an undocumented round label.")
    if frame.duplicated(subset=["expert_record_id", "delphi_round"]).any():
        raise ValueError("Each expert may appear only once within each Delphi round.")
    expected_rounds = ALLOWED_DELPHI_ROUNDS
    observed_by_expert = frame.groupby("expert_record_id")["delphi_round"].agg(set)
    if not observed_by_expert.map(lambda rounds: rounds == expected_rounds).all():
        raise ValueError("Each deidentified expert must have one score record in both Delphi rounds.")
    _validate_rating_columns(frame, DELPHI_ROUND_COLUMNS[2:], "Delphi-round expert data")


def load_and_validate(data_dir: Path, *, require_snapshot_counts: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    teacher_path = data_dir / "teacher_ratings_deidentified.csv"
    expert_path = data_dir / "expert_ratings_deidentified.csv"
    if not teacher_path.is_file() or not expert_path.is_file():
        raise FileNotFoundError(
            "Both public rating CSV files are required: "
            f"{teacher_path.name} and {expert_path.name}."
        )
    teachers = pd.read_csv(teacher_path)
    experts = pd.read_csv(expert_path)
    validate_teacher_frame(teachers, require_snapshot_counts=require_snapshot_counts)
    validate_expert_frame(experts, require_snapshot_counts=require_snapshot_counts)
    return teachers, experts


def load_and_validate_submission_inputs(
    data_dir: Path, *, require_snapshot_counts: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the final scores and paired expert-round score records."""
    teachers, experts = load_and_validate(data_dir, require_snapshot_counts=require_snapshot_counts)
    delphi_path = data_dir / "expert_delphi_rounds_deidentified.csv"
    if not delphi_path.is_file():
        raise FileNotFoundError(f"Required Delphi-round CSV is missing: {delphi_path.name}.")
    delphi_rounds = pd.read_csv(delphi_path)
    validate_delphi_round_frame(delphi_rounds, require_snapshot_counts=require_snapshot_counts)

    score_columns = EXPERT_COLUMNS[1:]
    round_average = (
        delphi_rounds.groupby("expert_record_id", as_index=False)[score_columns]
        .mean()
        .sort_values("expert_record_id")
        .reset_index(drop=True)
    )
    final_scores = experts.sort_values("expert_record_id").reset_index(drop=True)
    if list(round_average["expert_record_id"]) != list(final_scores["expert_record_id"]):
        raise ValueError("Delphi-round expert IDs do not match the final expert-score IDs.")
    differences = (round_average[score_columns] - final_scores[score_columns]).abs()
    if differences.to_numpy().max(initial=0.0) > 1e-9:
        raise ValueError("Final expert scores must equal the arithmetic mean of the two Delphi rounds.")
    return teachers, experts, delphi_rounds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "public_ratings",
        help="Directory containing the anonymous public rating CSV files.",
    )
    parser.add_argument(
        "--allow-different-row-counts",
        action="store_true",
        help="Validate schema and values without enforcing the verified 137-teacher/4-expert snapshot counts.",
    )
    args = parser.parse_args()
    teachers, experts, delphi_rounds = load_and_validate_submission_inputs(
        args.data_dir,
        require_snapshot_counts=not args.allow_different_row_counts,
    )
    print(f"Validated teacher rows: {len(teachers)}")
    print(f"Validated final expert rows: {len(experts)}")
    print(f"Validated paired Delphi-round rows: {len(delphi_rounds)}")


if __name__ == "__main__":
    main()
