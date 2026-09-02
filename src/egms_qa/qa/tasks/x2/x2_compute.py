"""Build X2 unsupported-data/scope boundary catalog."""
from __future__ import annotations

import csv
from pathlib import Path


OUT_DIR = Path("./outputs/tasks/x2")
OUT_PATH = OUT_DIR / "x2_final_table.csv"

ANSWER_POLICY = (
    "Refuse unsupported data/scope requests, name the missing scale/channel/time/reference universe, "
    "then redirect to supported tile-level, bin-level, or corpus-relative tasks."
)

TASKS = [
    {
        "task_id": "X21",
        "target_column": "X21_exact_asset_refusal",
        "target_value": "refusal",
        "trigger": "User asks for a specific address, building, road segment, parcel, or named asset conclusion.",
        "refusal_reason": "The task set is tile-level and does not support exact asset-level judgment.",
        "supported_redirect": "Redirect to tile-level tasks such as A51/B21/B22/C51/C52/D42/S11/S22, or supported 8x8 bin locations C13/C32/D35 where defined.",
        "supported_redirect_tasks": "A51|B21|B22|C13|C32|C51|C52|D35|D42|S11|S22",
        "answer_policy": ANSWER_POLICY,
        "response_template": "Cannot make an exact asset-level judgment from this tile-level task set. The supported unit is the tile, with limited 8x8 bin-level locations for specific tasks. I can answer the tile-level monitoring status instead.",
    },
    {
        "task_id": "X22",
        "target_column": "X22_point_subcell_exact_refusal",
        "target_value": "refusal",
        "trigger": "User asks for a single point, sub-cell, exact pixel, or exact coordinate-level conclusion.",
        "refusal_reason": "The supported spatial scale is tile-level, with limited 8x8 bin-level explanation for specific tasks.",
        "supported_redirect": "Redirect to tile-level status A51/B21/C21/D42, or supported bin-level locations C13/C32/D35.",
        "supported_redirect_tasks": "A51|B21|C21|C13|C32|D35|D42",
        "answer_policy": ANSWER_POLICY,
        "response_template": "Cannot provide a single-point or sub-cell exact conclusion. The supported scale is tile-level, with limited 8x8 bin-level explanations for specific location tasks. I can answer the tile-level result or a supported bin location.",
    },
    {
        "task_id": "X23",
        "target_column": "X23_unsupported_displacement_component_refusal",
        "target_value": "refusal",
        "trigger": "User asks for a displacement direction or component not present in the input, such as unsupported horizontal or LOS-outside conclusions.",
        "refusal_reason": "The task can only answer displacement components and derived quantities present in the provided EGMS inputs.",
        "supported_redirect": "Redirect to supported EGMS-derived components: B21/B31/B32/B33 for velocity, B41 for acceleration strength, B51 for seasonality strength, and D21/D22/D31 for temporal behavior.",
        "supported_redirect_tasks": "B21|B31|B32|B33|B41|B51|D21|D22|D31",
        "answer_policy": ANSWER_POLICY,
        "response_template": "Cannot answer a displacement component that is not present in the input data. The task can only use the supported EGMS components and derived targets. I can answer supported velocity, acceleration, seasonal, spatial, or temporal quantities.",
    },
    {
        "task_id": "X24",
        "target_column": "X24_external_context_refusal",
        "target_value": "refusal",
        "trigger": "User asks for land use, imagery interpretation, building type, geology, infrastructure inventory, or other external context.",
        "refusal_reason": "The task set does not include external imagery, land-use, geology, or asset inventory layers.",
        "supported_redirect": "Redirect to EGMS/encoder evidence: B21/B22 for motion, C21/C31 for spatial organization, D11/D21 for temporal behavior, and S11/S22/S41 for representation context.",
        "supported_redirect_tasks": "B21|B22|C21|C31|D11|D21|S11|S22|S41",
        "answer_policy": ANSWER_POLICY,
        "response_template": "Cannot infer external context such as land use, building type, geology, or imagery content from this task set. Those layers are not included. I can describe EGMS-derived deformation and encoder representation evidence.",
    },
    {
        "task_id": "X25",
        "target_column": "X25_live_status_refusal",
        "target_value": "refusal",
        "trigger": "User asks what is happening now, live, today, or after the historical EGMS observation window.",
        "refusal_reason": "EGMS inputs are historical products, not a live feed.",
        "supported_redirect": "Redirect to historical-window evidence: D11-D14 for trend/regime, D31-D35 for acceleration/intensification, and D42 for temporal archetype.",
        "supported_redirect_tasks": "D11|D12|D13|D14|D31|D32|D33|D34|D35|D42",
        "answer_policy": ANSWER_POLICY,
        "response_template": "Cannot answer live or current status beyond the historical observation window. EGMS inputs here are not a live feed. I can describe the status within the available historical data.",
    },
    {
        "task_id": "X26",
        "target_column": "X26_unsupported_ranking_superlative_refusal",
        "target_value": "refusal",
        "trigger": "User asks for open-world ranking or superlatives such as worst in Europe, rank number, most severe, or highest risk.",
        "refusal_reason": "Task-defined Europe/corpus-relative classes can be answered, but open-world ranks or superlatives require a defined ranking table and reference universe.",
        "supported_redirect": "Redirect to predefined relative classes only: B22/B35/B36/B42 for motion typicality, C12/C22/C33/C42 for spatial classes, and S22/S32 for corpus-relative representation labels.",
        "supported_redirect_tasks": "B22|B35|B36|B42|C12|C22|C33|C42|S22|S32",
        "answer_policy": ANSWER_POLICY,
        "response_template": "Cannot claim an undefined rank or superlative such as worst, rank number, most severe, or highest risk. I can answer predefined Europe-defined or corpus-relative classes, percentiles, or typicality labels when the task defines that reference distribution.",
    },
]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "task_id",
        "target_column",
        "target_value",
        "trigger",
        "refusal_reason",
        "supported_redirect",
        "supported_redirect_tasks",
        "answer_policy",
        "response_template",
    ]
    with OUT_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(TASKS)


if __name__ == "__main__":
    main()
