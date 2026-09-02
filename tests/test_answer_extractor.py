from __future__ import annotations

from egms_qa.translator.answer_extractor import (
    AMBIGUOUS,
    PARSED,
    UNPARSED,
    extract_answer,
    extraction_is_correct,
)
from egms_qa.translator.evaluate import summarize_task
from egms_qa.qa.qa_lib import MISSING_VALUE, TaskRecord


def task(task_id: str, label_type: str, target_column: str = "") -> TaskRecord:
    return TaskRecord(
        task_id=task_id,
        family=task_id[0],
        name=task_id,
        target_column=target_column or task_id,
        label_type=label_type,
        probe_applicable=label_type != "refusal",
    )


def test_categorical_contraction_maps_to_canonical_label() -> None:
    result = extract_answer(
        "This area doesn't show a clear average subsidence signal.",
        task("B12", "categorical"),
        ["clear_subsidence", "no_clear_subsidence"],
    )
    assert result.status == PARSED
    assert result.value == "no_clear_subsidence"


def test_spoken_decimal_uses_the_task_unit_to_ignore_other_numbers() -> None:
    result = extract_answer(
        "The mean velocity is minus one point four millimetres per year, based on 64 observations.",
        task("B21", "numeric", "mean_velocity_mm_yr"),
    )
    assert result.status == PARSED
    assert result.value == -1.4


def test_numeric_task_rejects_unrelated_number_without_expected_unit() -> None:
    result = extract_answer(
        "Those quantities require at least three observations across the area.",
        task("A21", "numeric", "masked_global_mse_z"),
    )
    assert result.status == UNPARSED
    assert result.value is None


def test_dimensionless_fraction_takes_precedence_over_acceleration_keyword() -> None:
    result = extract_answer(
        "The fraction of spatial bins supporting the acceleration signal is 0.622.",
        task("D32", "numeric", "acceleration_support_fraction"),
    )
    assert result.status == PARSED
    assert result.value == 0.622


def test_percent_is_normalized_for_fraction_task() -> None:
    result = extract_answer(
        "About 43 percent of observed points move beyond their noise level.",
        task("C11", "numeric", "moving_fraction"),
    )
    assert result.status == PARSED
    assert result.value == 0.43


def test_s12_dimensionless_distance_uses_explicit_decimal_literal() -> None:
    result = extract_answer(
        "The nearest reference profile has a distance of 0.500 million units.",
        task("S12", "numeric", "S12_reference_anchor_distance"),
    )
    assert result.status == PARSED
    assert result.value == 0.5


def test_s12_magnitude_phrase_without_numeric_literal_is_unparsed() -> None:
    result = extract_answer(
        "The signed perl million foot squared foot squared.",
        task("S12", "numeric", "S12_reference_anchor_distance"),
    )
    assert result.status == UNPARSED
    assert result.value is None


def test_dimensionless_task_rejects_physical_unit() -> None:
    result = extract_answer(
        "The seasonal phase has 77.7 days.",
        task("D22", "numeric", "seasonal_phase_coherence"),
    )
    assert result.status == UNPARSED
    assert result.value is None


def test_fraction_task_rejects_value_outside_unit_interval() -> None:
    result = extract_answer(
        "Fast local motion appears in 29 spatial bins.",
        task("C41", "numeric", "fast_motion_bin_fraction"),
    )
    assert result.status == UNPARSED
    assert result.value is None


def test_numeric_range_is_not_read_as_its_midpoint() -> None:
    result = extract_answer(
        "Fast local motion does not appear in 0-100-000-000.",
        task("C41", "numeric", "fast_motion_bin_fraction"),
    )
    assert result.status == UNPARSED
    assert result.value is None


def test_multiple_plausible_values_are_ambiguous() -> None:
    result = extract_answer(
        "The spatial concentration score is either 0.4 or 0.5.",
        task("C21", "numeric", "concentration_score"),
    )
    assert result.status == AMBIGUOUS
    assert result.value is None


def test_negated_class_alias_is_not_treated_as_the_class() -> None:
    result = extract_answer(
        "Measurement noise is not high.",
        task("A42", "categorical"),
        ["low_noise", "moderate_noise", "high_noise", "very_high_noise"],
    )
    assert result.status == UNPARSED


def test_independent_class_aliases_are_ambiguous() -> None:
    result = extract_answer(
        "Measurement noise looks low in places but very high elsewhere.",
        task("A42", "categorical"),
        ["low_noise", "moderate_noise", "high_noise", "very_high_noise"],
    )
    assert result.status == AMBIGUOUS


def test_numeric_correctness_uses_renderer_precision() -> None:
    numeric_task = task("B21", "numeric", "mean_velocity_mm_yr")
    row = {
        "answer_type": "numeric",
        "answer_value": -1.416,
        "rendered_target_value": "-1.42",
    }
    result = extract_answer("The mean vertical ground velocity is -1.42 mm/yr.", numeric_task)
    assert extraction_is_correct(row, numeric_task, result)


def test_missing_outcome_is_scored_before_numeric_conversion() -> None:
    numeric_task = task("D14", "numeric", "change_point_year")
    row = {
        "answer_type": "missing",
        "answer_value": MISSING_VALUE,
        "rendered_target_value": None,
    }
    result = extract_answer("This result is not available for this area.", numeric_task)
    assert extraction_is_correct(row, numeric_task, result)


def test_missing_prediction_for_numeric_target_is_an_error_not_a_crash() -> None:
    numeric_task = task("D14", "numeric", "change_point_year")
    row = {
        "answer_type": "numeric",
        "answer_value": 2018.0,
        "rendered_target_value": "2018.00",
    }
    result = extract_answer("This result is not available for this area.", numeric_task)
    assert not extraction_is_correct(row, numeric_task, result)


def test_numeric_summary_keeps_targets_and_predictions_paired() -> None:
    numeric_task = task("D14", "numeric", "change_point_year")
    rows = [
        {
            "answer_type": "numeric",
            "answer_value": 1.0,
            "rendered_target_value": "1.0",
            "pred": 1.1,
            "ok": False,
            "extraction_status": PARSED,
        },
        {
            "answer_type": "numeric",
            "answer_value": 2.0,
            "rendered_target_value": "2.0",
            "pred": MISSING_VALUE,
            "ok": False,
            "extraction_status": PARSED,
        },
        {
            "answer_type": "numeric",
            "answer_value": 3.0,
            "rendered_target_value": "3.0",
            "pred": 2.5,
            "ok": False,
            "extraction_status": PARSED,
        },
    ]

    summary = summarize_task(numeric_task, rows)

    assert summary["n_numeric"] == 3
    assert summary["n_pred"] == 2
    assert summary["mae"] == 0.3
    assert summary["r2"] == 0.87


def test_explicit_refusal_paraphrase_is_recognized() -> None:
    result = extract_answer(
        "The safe answer is to decline. The tile-level evidence cannot establish a cause.",
        task("X11", "refusal"),
    )
    assert result.status == PARSED
    assert result.value == "refusal"


def test_explicit_missing_value_paraphrase_is_recognized() -> None:
    result = extract_answer(
        "This EGMS tile has no valid output for that readout.",
        task("D14", "categorical"),
        ["stable", "unstable"],
    )
    assert result.status == PARSED
    assert result.value == MISSING_VALUE


def test_plain_negative_answer_is_not_mistaken_for_refusal() -> None:
    result = extract_answer(
        "No, the building is safe.",
        task("X21", "refusal"),
    )
    assert result.status == UNPARSED
