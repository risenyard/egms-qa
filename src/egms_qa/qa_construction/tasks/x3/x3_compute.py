"""Build X3 representation-boundary catalog."""
from __future__ import annotations

import csv
from pathlib import Path


OUT_DIR = Path("./outputs/tasks/x3")
OUT_PATH = OUT_DIR / "x3_final_table.csv"

ANSWER_POLICY = (
    "Refuse over-interpretation of encoder representations, name the missing attribution/probe/causal evidence, "
    "then redirect to representation-level statements plus supported A/B/C/D monitoring facts."
)

TASKS = [
    {
        "task_id": "X31",
        "target_column": "X31_representation_as_ground_truth_refusal",
        "target_value": "refusal",
        "trigger": "User treats an encoder anchor, rarity score, or local-structure construct as a direct physical, geological, or engineering truth label.",
        "refusal_reason": "S-group outputs are encoder representation constructs, not external physical ground truth.",
        "supported_redirect": "Redirect to representation constructs S11/S12/S14/S21/S22/S31/S32/S41/S42/S43 and separately to monitoring facts A51/B21/C31/D11 where relevant.",
        "supported_redirect_tasks": "S11|S12|S14|S21|S22|S31|S32|S41|S42|S43|A51|B21|C31|D11",
        "answer_policy": ANSWER_POLICY,
        "response_template": "Cannot treat the S-group representation result as direct physical, geological, or engineering ground truth. It is an encoder-level construct. I can state the representation result and separately report supported A/B/C/D monitoring facts.",
    },
    {
        "task_id": "X32",
        "target_column": "X32_representation_causality_refusal",
        "target_value": "refusal",
        "trigger": "User asks whether an embedding pattern, anchor membership, or representation rarity proves a real-world cause.",
        "refusal_reason": "Embedding similarity, rarity, or anchor assignment does not establish causal explanation.",
        "supported_redirect": "Redirect to representation similarity/rarity tasks S11/S12/S14/S21/S22/S31/S32/S33 and to A-D monitoring facts without causal attribution.",
        "supported_redirect_tasks": "S11|S12|S14|S21|S22|S31|S32|S33|A51|B21|C21|D11",
        "answer_policy": ANSWER_POLICY,
        "response_template": "Cannot use embedding similarity, anchor membership, or representation rarity as proof of a real-world cause. These are representation-level signals, not causal evidence. I can describe the representation pattern without converting it into causal attribution.",
    },
    {
        "task_id": "X33",
        "target_column": "X33_model_mechanism_certainty_refusal",
        "target_value": "refusal",
        "trigger": "User asks for a certain semantic meaning of a specific token dimension, latent coordinate, or model mechanism without attribution evidence.",
        "refusal_reason": "A token dimension or latent coordinate cannot be assigned a fixed physical meaning without attribution, probe, or validation evidence.",
        "supported_redirect": "Redirect to interpreted task-level constructs S11-S15, S21-S22, S31-S33, and S41-S43 rather than assigning semantics to raw dimensions.",
        "supported_redirect_tasks": "S11|S12|S13|S14|S15|S21|S22|S31|S32|S33|S41|S42|S43",
        "answer_policy": ANSWER_POLICY,
        "response_template": "Cannot assign a fixed physical meaning to a token dimension, latent coordinate, or model mechanism without attribution, probe, or validation evidence. I can use available task-level constructs or state that the dimension is not interpreted.",
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
