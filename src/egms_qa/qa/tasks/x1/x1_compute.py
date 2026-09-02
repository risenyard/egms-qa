"""Build X1 unsupported-inference boundary catalog."""
from __future__ import annotations

import csv
from pathlib import Path


OUT_DIR = Path("./outputs/tasks/x1")
OUT_PATH = OUT_DIR / "x1_final_table.csv"

ANSWER_POLICY = (
    "Refuse unsupported inference, state the missing causal/predictive/engineering/economic evidence, "
    "then redirect to supported historical monitoring facts."
)

TASKS = [
    {
        "task_id": "X11",
        "target_column": "X11_causal_attribution_refusal",
        "target_value": "refusal",
        "trigger": "User asks whether deformation is caused by groundwater, geology, construction, mining, or another external cause.",
        "refusal_reason": "EGMS deformation observations can describe motion, but they do not establish a causal mechanism by themselves.",
        "supported_redirect": "Redirect first to A51/A52 for monitoring usability, then to supported deformation facts: B21/B31/B33 for motion, C11/C21/C31 for spatial organization, D11/D21/D31 for temporal dynamics, and S11/S22 for representation context.",
        "supported_redirect_tasks": "A51|A52|B21|B31|B33|C11|C21|C31|D11|D21|D31|S11|S22",
        "answer_policy": ANSWER_POLICY,
        "response_template": "Cannot determine the cause from the current EGMS tile data alone. The data can describe deformation behavior, but causal attribution needs external evidence. I can answer supported deformation facts such as velocity, spatial pattern, trend, seasonality, or acceleration.",
    },
    {
        "task_id": "X12",
        "target_column": "X12_future_forecast_refusal",
        "target_value": "refusal",
        "trigger": "User asks whether deformation will continue, worsen, stop, or happen in the future.",
        "refusal_reason": "EGMS-QA summarizes historical EGMS and encoder evidence; it is not a forecasting model.",
        "supported_redirect": "Redirect to historical dynamics: D11-D14 for trend/regime evidence, D21-D24 for seasonal behavior, D31-D35 for acceleration/intensification, and D41-D42 for temporal composition.",
        "supported_redirect_tasks": "D11|D12|D13|D14|D21|D22|D23|D24|D31|D32|D33|D34|D35|D41|D42",
        "answer_policy": ANSWER_POLICY,
        "response_template": "Cannot forecast future deformation from this task set. EGMS-QA describes the historical observation window, not future outcomes. I can answer historical trend shape, changepoint evidence, acceleration, or seasonal behavior.",
    },
    {
        "task_id": "X13",
        "target_column": "X13_building_safety_refusal",
        "target_value": "refusal",
        "trigger": "User asks whether a building or infrastructure asset is safe, unsafe, collapsing, or habitable.",
        "refusal_reason": "Tile-level EGMS deformation cannot replace structural safety assessment or field inspection.",
        "supported_redirect": "Redirect to monitoring evidence: A51/A52 for usability, B12/B22/B35/B36/B61 for motion signal and trigger, C33/C51/C52 for spatial priority/context, and D42 for temporal archetype.",
        "supported_redirect_tasks": "A51|A52|B12|B22|B35|B36|B61|C33|C51|C52|D42",
        "answer_policy": ANSWER_POLICY,
        "response_template": "Cannot determine building or infrastructure safety from tile-level EGMS data. That requires structural assessment and field evidence. I can describe whether the tile shows supported deformation signals that may merit expert review.",
    },
    {
        "task_id": "X14",
        "target_column": "X14_economic_loss_refusal",
        "target_value": "refusal",
        "trigger": "User asks for monetary loss, insurance impact, asset value loss, or compensation.",
        "refusal_reason": "The task set has no exposure, asset-value, vulnerability, or economic loss model.",
        "supported_redirect": "Redirect to A51 for usability and deformation severity proxies, not loss: B22/B35/B36/B42 for motion typicality, C42/C51/C52 for spatial extent/priority, and D42 for temporal archetype.",
        "supported_redirect_tasks": "A51|B22|B35|B36|B42|C42|C51|C52|D42",
        "answer_policy": ANSWER_POLICY,
        "response_template": "Cannot estimate economic loss from the current EGMS/VQA task set. It has no exposure, asset-value, vulnerability, or loss model. I can describe deformation intensity, spatial extent, and monitoring priority where supported.",
    },
    {
        "task_id": "X15",
        "target_column": "X15_intervention_recommendation_refusal",
        "target_value": "refusal",
        "trigger": "User asks for engineering intervention, evacuation, repair, pumping, reinforcement, or operational action.",
        "refusal_reason": "The task set is not an intervention recommender and cannot prescribe engineering action.",
        "supported_redirect": "Redirect to monitoring evidence before any action discussion: A51/A52 for reliability, B61 for monitoring trigger, C51 for monitoring priority, C52 for hidden local risk, and D42 for temporal archetype.",
        "supported_redirect_tasks": "A51|A52|B61|C51|C52|D42",
        "answer_policy": ANSWER_POLICY,
        "response_template": "Cannot recommend engineering intervention or operational action from this task set. It is a monitoring description system, not an intervention recommender. I can state the supported monitoring evidence and note that expert evaluation is needed for action decisions.",
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
