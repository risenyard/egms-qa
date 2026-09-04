"""Shared EGMS-QA QA rendering utilities over the EGMS-QA task catalog.

The logical dataset is:
  10,000 tiles x 78 delivered tasks x 20 phrase variants.

Probe-applicable tasks use the labels parquet produced by build_labels.py.
X tasks are static refusal policies from the task reference tables and are
sampled so refusal examples teach boundaries without dominating token-dependent
monitoring questions.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from egms_qa.paths import (
    LABELS as DEFAULT_LABELS,
    LABELS_META as DEFAULT_META,
    TASKS_DIR as DEFAULT_TASKS_DIR,
    QA_AUDIT as DEFAULT_QA_AUDIT,
)

N_PHRASES = 20
MISSING_VALUE = "__not_available__"
TARGET_FORMATS = ("natural",)
NUMERIC_ANSWER_STYLES = ("standard", "concise")
QA_SYSTEM_VERSION = "EGMS-QA"
ANSWER_PROTOCOL = "natural_only"


USER_QUESTION_TEMPLATES = [
    "{query}?",
    "Could you tell me: {query_lc}?",
    "Based on the available deformation measurements, {query_lc}?",
    "What does the deformation record indicate about {topic}?",
    "How would you assess {topic} in this area?",
    "Please summarize {topic} for this area.",
    "What should I know about {topic} here?",
    "For monitoring purposes, {query_lc}?",
    "Looking at the observed ground motion, {query_lc}?",
    "What is the result for {topic} in this area?",
    "How should {topic} be described here?",
    "Please report {topic} for this area.",
    "What do the historical measurements show about {topic}?",
    "Can you assess {topic} from this area's deformation record?",
    "For a ground-motion screening report, {query_lc}?",
    "What conclusion can be drawn about {topic} here?",
    "Using the available observations, {query_lc}?",
    "How does this area look in terms of {topic}?",
    "Please give the area-level result for {topic}.",
    "From a monitoring perspective, {query_lc}?",
]

MISSING_ANSWER = (
    "This result is not available for this area because the measurements do not "
    "meet the requirements for this calculation."
)

TASK_USER_TEXT: dict[str, tuple[str, str, str]] = {
    "A11": ("how sensitive the area summary is to missing observations", "How sensitive is the area summary if some radar points are missing", "The representation drift"),
    "A12": ("representation stability", "Is the area summary stable enough to trust", "The representation stability"),
    "A21": ("temporal reconstruction quality", "How well can the historical deformation time series be reconstructed", "The reconstruction error"),
    "A22": ("temporal reconstruction reliability", "Is the reconstructed deformation history reliable", "The reconstruction reliability"),
    "A31": ("spatial observation coverage", "How well do the radar points cover the area", "The spatial coverage"),
    "A32": ("spatial coverage quality", "Is the radar point coverage dense enough for monitoring", "The coverage status"),
    "A41": ("measurement noise", "How noisy are the deformation measurements", "The median measurement noise"),
    "A42": ("noise level", "Is the measurement noise low or high", "The noise level"),
    "A51": ("monitoring usability", "Can this tile be used for monitoring without caution", "The monitoring usability"),
    "A52": ("the reason for the monitoring usability decision", "What is the main reason behind the monitoring usability decision", "The usability reason"),
    "B11": ("average subsidence signal strength", "How strong is the average subsidence signal relative to noise", "The subsidence signal-to-noise ratio"),
    "B12": ("clear average subsidence", "Does this area show a clear average subsidence signal", "The clear-subsidence status"),
    "B21": ("mean vertical velocity", "What is the average vertical ground velocity", "The mean vertical velocity"),
    "B22": ("average subsidence severity", "How severe is the average subsidence in this area", "The average subsidence severity"),
    "B31": ("sinking tail velocity", "How fast are the more strongly sinking points moving", "The sinking tail velocity"),
    "B32": ("upper-tail vertical velocity", "What is the upper-tail vertical velocity in this area", "The upper-tail velocity"),
    "B33": ("strongest local motion magnitude", "How large is the strongest local ground-motion tail", "The strongest local motion magnitude"),
    "B34": ("uplift versus non-uplift direction", "Is this area mainly uplift or non-uplift", "The motion direction status"),
    "B35": ("worst-point significance", "How significant is the worst local motion point", "The worst-point significance"),
    "B36": ("European velocity anomaly level", "How unusual is this area's velocity compared with the European background", "The European velocity anomaly level"),
    "B41": ("acceleration strength", "How strong is the recent acceleration signal", "The acceleration strength"),
    "B42": ("European acceleration anomaly level", "How unusual is this area's acceleration compared with the European background", "The European acceleration anomaly level"),
    "B51": ("seasonal motion strength", "How strong is the seasonal deformation component", "The seasonal motion strength"),
    "B61": ("monitoring trigger", "Should this area trigger monitoring attention", "The monitoring trigger"),
    "C11": ("moving-point extent", "What fraction of the observed points are moving beyond noise", "The moving-point fraction"),
    "C12": ("spatial extent of motion", "Is the motion localized or spread across the area", "The motion extent"),
    "C13": ("location of the strongest moving bin", "Where is the strongest moving part of the tile", "The strongest moving-bin location"),
    "C21": ("spatial concentration of motion", "How concentrated is the deformation pattern", "The spatial concentration"),
    "C22": ("motion concentration class", "Is the deformation diffuse or concentrated", "The concentration class"),
    "C31": ("deformation front strength", "How strong is the sharpest deformation front", "The deformation-front strength"),
    "C32": ("deformation front location", "Where is the strongest deformation front", "The deformation-front location"),
    "C33": ("deformation front sharpness", "How sharp is the deformation front", "The deformation-front sharpness"),
    "C41": ("fast-tail spatial extent", "How much of the area contains fast local motion", "The fast-tail spatial extent"),
    "C42": ("fast-motion extent class", "Is the fast local motion isolated or widespread", "The fast-motion extent"),
    "C51": ("monitoring priority", "What monitoring priority should this area receive", "The monitoring priority"),
    "C52": ("hidden local risk", "Is there a local risk that the average motion would hide", "The hidden-local-risk status"),
    "D11": ("long-term trend shape", "What kind of long-term deformation trend does this area show", "The long-term trend shape"),
    "D12": ("trend curvature strength", "How nonlinear is the long-term deformation trend", "The trend curvature strength"),
    "D13": ("change-point confidence", "How confident is the strongest detected change point", "The change-point confidence"),
    "D14": ("timing of the strongest change point", "When does the strongest deformation change point occur", "The strongest change-point time"),
    "D21": ("seasonal peak timing", "In which season does the deformation cycle peak", "The seasonal peak timing"),
    "D22": ("seasonal phase coherence", "How synchronized is the seasonal deformation phase across the area", "The seasonal phase coherence"),
    "D23": ("seasonal phase spread", "How spread out are the seasonal phases across the area", "The seasonal phase spread"),
    "D24": ("seasonal amplitude change", "Is the seasonal deformation amplitude changing over time", "The seasonal amplitude change"),
    "D31": ("motion intensification", "Is the ground motion intensifying recently", "The motion intensification"),
    "D32": ("spatial support for acceleration", "How much of the area supports the acceleration signal", "The acceleration support"),
    "D33": ("spread of intensification", "How broadly is intensification distributed in the area", "The intensification spread"),
    "D34": ("intensification hotspot strength", "How strong is the main acceleration hotspot", "The acceleration hotspot strength"),
    "D35": ("intensification hotspot location", "Where is the main acceleration hotspot", "The acceleration hotspot location"),
    "D41": ("dominant temporal process", "What temporal process best describes the deformation", "The dominant temporal process"),
    "D42": ("temporal evolution pattern", "What overall time-evolution pattern does this area show", "The temporal evolution pattern"),
    "S11": ("similar reference profile", "Which known deformation profile does this area most resemble", "The nearest reference profile"),
    "S12": ("distance to the nearest reference profile", "How close is this area to its nearest reference profile", "The reference-profile distance"),
    "S13": ("separation from the next reference profile", "How clearly is this area separated from the next reference profile", "The reference-profile margin"),
    "S14": ("reference-profile assignment confidence", "Is the reference-profile assignment clear or borderline", "The reference assignment status"),
    "S15": ("plain-language reference profile", "What plain-language reference profile best describes this area", "The reference profile description"),
    "S21": ("representation isolation", "How isolated is this area in the learned deformation representation", "The representation isolation score"),
    "S22": ("representation rarity", "How rare is this area in the learned deformation representation", "The representation rarity"),
    "S31": ("gap between representation rarity and monitoring severity", "Does the learned representation find this area more unusual than the monitoring metrics do", "The representation-monitoring rarity gap"),
    "S32": ("relationship between representation rarity and monitoring severity", "How do representation rarity and monitoring severity compare", "The representation-monitoring relation"),
    "S33": ("most distinctive monitoring dimension", "Which monitoring dimension most distinguishes this area", "The distinctive monitoring dimension"),
    "S41": ("learned local-structure strength", "How much local deformation structure stands out in this area", "The local-structure strength"),
    "S42": ("learned local-structure class", "Is the learned local deformation structure weak or strong", "The local-structure class"),
    "S43": ("concentration of learned local structure", "How concentrated is the learned local structure", "The local-structure concentration"),
}


# Questions vary across the 20-phrasing pool, but answers use one stable,
# task-specific sentence. This keeps the supervision user-facing and avoids
# teaching the model arbitrary changes in answer wording.
NATURAL_NUMERIC_ANSWERS: dict[str, str] = {
    "A11": "The normalized angular change in the area summary after removing most observations is {value}.",
    "A21": "The reconstructed deformation history has a normalized mean-squared error of {value} z^2.",
    "A31": "The spatial coverage fraction is {value} across the 8-by-8 monitoring grid.",
    "A41": "The median measurement uncertainty is {value} mm.",
    "B11": "The average subsidence signal-to-noise ratio is {value}.",
    "B21": "The mean vertical ground velocity is {value} mm/yr.",
    "B31": "The sinking-tail velocity is {value} mm/yr at the 10th percentile.",
    "B32": "The upper-tail vertical velocity is {value} mm/yr.",
    "B33": "The absolute vertical velocity is {value} mm/yr at the 90th percentile.",
    "B41": "The acceleration magnitude is {value} mm/yr^2 at the 90th percentile.",
    "B51": "The seasonal deformation amplitude is {value} mm at the 90th percentile.",
    "C11": "The fraction of observed points moving beyond their noise level is {value}.",
    "C21": "The spatial concentration score is {value}, with larger values indicating more localized motion.",
    "C31": "The strongest deformation front has a velocity contrast of {value} mm/yr.",
    "C41": "Fast local motion appears in {value} of the spatial bins.",
    "D12": "The long-term trend curvature strength is {value}.",
    "D13": "The strongest detected change point has a confidence score of {value}.",
    "D14": "The strongest detected change point occurs around the year {value}.",
    "D22": "The seasonal phase coherence is {value}, where higher values mean more synchronized seasonal motion.",
    "D23": "The seasonal phases have a spread of {value} days.",
    "D24": "The change in seasonal deformation amplitude over the observation period is {value} mm.",
    "D31": "Recent motion intensification is {value} mm/yr^2.",
    "D32": "The fraction of spatial bins supporting the acceleration signal is {value}.",
    "D33": "The spatial spread of intensification is {value} mm/yr^2.",
    "D34": "The main intensification hotspot has a strength of {value} mm/yr^2.",
    "S12": "The distance to the nearest reference profile is {value}.",
    "S13": "The separation margin from the next reference profile is {value}.",
    "S21": "The representation isolation score is {value}.",
    "S31": "The signed percentile gap between representation rarity and monitoring severity is {value}.",
    "S41": "The learned local-structure strength is {value}.",
    "S43": "The learned local structure has a concentration score of {value}.",
}


CATEGORICAL_LABEL_DESCRIPTIONS: dict[str, dict[str, str]] = {
    "A12": {
        "stable": "stable when observations are removed",
        "mildly_sensitive": "mildly sensitive to missing observations",
        "highly_sensitive": "highly sensitive to missing observations",
        "extreme": "extremely sensitive to missing observations",
    },
    "A22": {
        "reconstructable": "reliable enough to reconstruct",
        "mildly_hard": "somewhat difficult to reconstruct",
        "high_error": "reconstructed with high error",
        "unreliable": "too unreliable to reconstruct",
    },
    "A32": {
        "well_spread": "well spread",
        "moderate_gaps": "affected by moderate gaps",
        "sparse": "sparse",
        "highly_fragmented": "highly fragmented",
    },
    "A42": {
        "low_noise": "low",
        "moderate_noise": "moderate",
        "high_noise": "high",
        "very_high_noise": "very high",
    },
    "A51": {
        "usable": "usable",
        "caution": "usable with caution",
        "unreliable": "not reliable enough",
    },
    "A52": {
        "stable_inputs": "stable input observations",
        "high_noise": "high measurement noise",
        "multiple_quality_issues": "several substantial data-quality issues",
        "multiple_minor_issues": "several minor data-quality issues",
        "sparse_coverage": "sparse spatial coverage",
        "sensitive_representation": "an area summary that is sensitive to missing observations",
        "fragmented_coverage": "fragmented spatial coverage",
        "very_high_noise": "very high measurement noise",
        "unstable_representation": "an unstable area summary",
        "poor_reconstruction": "poor reconstruction of the deformation history",
        "high_reconstruction_error": "high reconstruction error",
    },
    "B12": {
        "clear_subsidence": "shows a clear average subsidence signal",
        "no_clear_subsidence": "does not show a clear average subsidence signal",
    },
    "B22": {
        "uplift": "dominated by uplift rather than subsidence",
        "low": "in the low severity band",
        "low_mid": "in the low-to-middle severity band",
        "mid": "in the middle severity band",
        "high_mid": "in the middle-to-high severity band",
        "high": "in the high severity band",
    },
    "B34": {
        "uplift": "uplift",
        "non_uplift": "not uplift",
    },
    "B35": {
        "very_low": "very low",
        "low": "low",
        "moderate": "moderate",
        "high": "high",
        "very_high": "very high",
    },
    "B36": {
        "typ_low": "within the lower part of the typical European range",
        "typ_high": "within the upper part of the typical European range",
        "low": "lower than the typical European range",
        "high": "higher than the typical European range",
        "extreme": "extreme relative to the European background",
    },
    "B42": {
        "typ_low": "within the lower part of the typical European range",
        "typ_high": "within the upper part of the typical European range",
        "low": "lower than the typical European range",
        "high": "higher than the typical European range",
        "extreme": "extreme relative to the European background",
    },
    "B61": {
        "yes": "should be flagged for monitoring attention",
        "no": "does not currently warrant additional monitoring attention",
    },
    "C12": {
        "limited": "limited to a small part of the area",
        "partial": "present in part of the area",
        "broad": "spread broadly across the area",
        "widespread": "widespread across the area",
    },
    "C22": {
        "diffuse": "diffuse",
        "mildly_concentrated": "mildly concentrated",
        "concentrated": "concentrated",
        "highly_concentrated": "highly concentrated",
    },
    "C33": {
        "weak": "weak",
        "moderate": "moderately sharp",
        "strong": "sharp",
        "very_sharp": "very sharp",
    },
    "C42": {
        "none": "not detected",
        "sparse": "present in only a few spatial bins",
        "localized": "localized to part of the area",
        "extensive": "extensive across the area",
    },
    "C51": {
        "none": "no additional",
        "standard": "standard",
        "high": "high",
    },
    "C52": {
        "no": "does not appear to hide a localized high-motion area",
        "yes": "may hide a localized high-motion area",
    },
    "D11": {
        "linear_trend": "an approximately linear trend",
        "curved_trend": "a curved trend",
        "stage_change": "a stage or regime change",
        "complex_trend": "a complex trend",
    },
    "D21": {
        "no_clear_seasonal_peak": "has no clear seasonal peak",
        "winter_peak": "peaks in winter",
        "spring_peak": "peaks in spring",
        "summer_peak": "peaks in summer",
        "autumn_peak": "peaks in autumn",
    },
    "D41": {
        "low_activity": "low activity",
        "mixed": "a mixture of processes",
        "seasonal_dominant": "primarily seasonal motion",
        "trend_dominant": "primarily a long-term trend",
        "acceleration_dominant": "primarily acceleration",
    },
    "D42": {
        "low_activity": "characterized by low activity",
        "linear_trend_dominated": "dominated by a linear trend",
        "curved_trend_dominated": "dominated by a curved trend",
        "regime_change_trend_dominated": "dominated by a trend with a regime change",
        "coherent_seasonal_dominated": "dominated by coherent seasonal motion",
        "incoherent_seasonal_dominated": "dominated by seasonal motion with inconsistent phase",
        "intensifying_acceleration_dominated": "dominated by intensifying acceleration",
        "weakening_acceleration_dominated": "dominated by weakening acceleration",
        "uncertain_direction_acceleration_dominated": "dominated by acceleration with uncertain direction",
        "trend_acceleration_mixed": "a mixture of trend and acceleration",
        "seasonal_acceleration_mixed": "a mixture of seasonal motion and acceleration",
        "trend_seasonal_mixed": "a mixture of trend and seasonal motion",
    },
    "S11": {
        "stable_low_activity_background_reference": "a stable, low-activity background profile",
        "mixed_acceleration_complex_trend_reference": "a mixed acceleration and complex-trend profile",
        "spring_trend_acceleration_reference": "a spring-associated trend and acceleration profile",
        "extreme_localized_deformation_front_reference": "an extreme, localized deformation-front profile",
        "summer_trend_seasonal_mixed_reference": "a summer-associated trend and seasonal profile",
        "coherent_autumn_seasonal_reference": "a coherent autumn-seasonal profile",
    },
    "S14": {
        "strongly_anchored": "strongly matched to one reference profile",
        "transition_or_weakly_anchored": "transitional or only weakly matched",
        "far_or_ambiguous_from_reference_anchors": "far from, or ambiguous between, the reference profiles",
    },
    "S15": {
        "large low-activity stable background reference with low velocity and acceleration": "a large, stable, low-activity background profile with low velocity and acceleration",
        "large mixed dynamic reference with elevated acceleration and complex trend behavior": "a large, dynamically mixed profile with elevated acceleration and complex trend behavior",
        "spring-associated trend and acceleration mixed reference": "a spring-associated profile combining trend and acceleration",
        "small extreme reference with strong localized deformation, front strength, and fast-tail extent": "a small, extreme profile with strong localized deformation, a sharp front, and substantial fast-motion extent",
        "summer-associated trend-seasonal mixed reference with relatively diffuse spatial structure": "a summer-associated profile combining trend and seasonal motion with relatively diffuse spatial structure",
        "compact autumn-seasonal reference with high phase coherence": "a compact autumn-seasonal profile with highly coherent phase",
    },
    "S22": {
        "common": "common",
        "unusual": "unusual",
        "rare": "rare",
        "extreme": "extremely rare",
    },
    "S32": {
        "aligned": "closely aligned",
        "moderate_encoder_excess": "different, with representation rarity moderately higher",
        "strong_encoder_excess": "different, with representation rarity much higher",
        "moderate_monitoring_excess": "different, with monitoring severity moderately higher",
        "strong_monitoring_excess": "different, with monitoring severity much higher",
    },
    "S33": {
        "temporal": "temporal",
        "spatial": "spatial",
        "motion": "ground-motion",
        "quality": "data-quality",
    },
    "S42": {
        "weak_local_structure": "weak local structure",
        "clear_local_structure": "clear local structure",
        "strong_local_structure": "strong local structure",
        "spatially_coherent": "spatially coherent local structure",
    },
}


CATEGORICAL_ANSWER_PATTERNS: dict[str, str] = {
    "A12": "The area summary is {description}.",
    "A22": "The historical deformation signal is {description}.",
    "A32": "Radar observations are {description} across the area.",
    "A42": "Measurement noise is {description}.",
    "A51": "This tile is {description} for monitoring.",
    "A52": "The monitoring decision is driven by {description}.",
    "B12": "This area {description}.",
    "B22": "Average vertical motion is {description}.",
    "B34": "The dominant vertical-motion direction is {description}.",
    "B35": "The worst local motion has {description} statistical significance.",
    "B36": "The area's vertical velocity is {description}.",
    "B42": "The area's acceleration is {description}.",
    "B61": "This area {description}.",
    "C12": "Ground motion is {description}.",
    "C13": "The strongest moving part is in {description}.",
    "C22": "The deformation pattern is {description}.",
    "C32": "The strongest deformation front lies {description}.",
    "C33": "The strongest deformation front is {description}.",
    "C42": "Fast local motion is {description}.",
    "C51": "This area has {description} monitoring priority.",
    "C52": "The average motion {description}.",
    "D11": "The long-term deformation follows {description}.",
    "D21": "The seasonal deformation {description}.",
    "D35": "The main intensification hotspot is {description}.",
    "D41": "The dominant temporal behavior is {description}.",
    "D42": "The temporal evolution is {description}.",
    "S11": "This area most closely matches {description}.",
    "S14": "The reference-profile assignment is {description}.",
    "S15": "This area most closely resembles {description}.",
    "S22": "In the learned representation, this area is {description}.",
    "S32": "Representation rarity and monitoring severity are {description}.",
    "S33": "The area is most distinctive in its {description} behavior.",
    "S42": "The learned local deformation pattern shows {description}.",
}


NATURAL_X_ANSWERS: dict[str, str] = {
    "X11": (
        "I cannot determine the cause from deformation measurements alone. "
        "They describe how the ground moved, while causal attribution requires external evidence such as geology, groundwater, construction, or mining records."
    ),
    "X12": (
        "I cannot forecast what will happen after the observation period from these historical measurements alone. "
        "I can describe the observed trend, seasonal behavior, change points, and recent acceleration."
    ),
    "X13": (
        "I cannot determine whether a building or infrastructure asset is safe from tile-level ground-motion data. "
        "That requires an asset-specific structural assessment and field inspection."
    ),
    "X14": (
        "I cannot estimate economic or insurance loss from deformation measurements alone. "
        "That calculation also requires information about exposed assets, their value, and their vulnerability."
    ),
    "X15": (
        "I cannot recommend an engineering intervention or operational action from these monitoring results alone. "
        "An appropriate response requires site-specific investigation and professional engineering judgment."
    ),
    "X21": (
        "I cannot make an exact judgment about a named building, road, or parcel from a tile-level result. "
        "I can report the monitoring status of the surrounding area."
    ),
    "X22": (
        "I cannot provide an exact single-point or pixel-level conclusion from this area summary. "
        "Only tile-level results and a few predefined coarse-grid locations are supported."
    ),
    "X23": (
        "I cannot report a displacement component that is not present in the supplied measurements. "
        "I can only describe the available vertical-motion, acceleration, seasonal, spatial, and temporal quantities."
    ),
    "X24": (
        "I cannot infer land use, building type, geology, or infrastructure from these deformation measurements. "
        "Those conclusions require external imagery or contextual data."
    ),
    "X25": (
        "I cannot report live conditions beyond the historical observation period because these measurements are not a real-time feed. "
        "I can describe what was observed during the available period."
    ),
    "X26": (
        "I cannot assign an open-ended rank such as the worst site in Europe without a defined comparison set and ranking method. "
        "I can report only the predefined relative severity and rarity classes."
    ),
    "X31": (
        "I cannot treat similarity or rarity in the learned representation as physical, geological, or engineering ground truth. "
        "I can report that representation result separately from the measured deformation evidence."
    ),
    "X32": (
        "I cannot use representation similarity, profile membership, or rarity as proof of a real-world cause. "
        "Those quantities describe learned patterns, not causal mechanisms."
    ),
    "X33": (
        "I cannot assign a fixed physical meaning to an individual hidden feature without dedicated interpretation and validation evidence. "
        "Only validated monitoring outputs should be interpreted."
    ),
}


X_USER_QUESTIONS: dict[str, list[str]] = {
    "X11": [
        "Is the subsidence here caused by groundwater pumping?",
        "Can you tell whether construction caused this deformation?",
        "Is geology the reason this area is moving?",
        "Does the tile prove mining activity is causing the motion?",
        "What is the cause of the deformation in this area?",
    ],
    "X12": [
        "Will this subsidence continue next year?",
        "Is the deformation going to get worse in the future?",
        "Can you forecast whether this area will keep moving?",
        "Will the motion stop or accelerate after the observation period?",
        "What will happen to this area in the coming years?",
    ],
    "X13": [
        "Is this building safe?",
        "Can people safely occupy structures in this tile?",
        "Does this ground motion mean the infrastructure is unsafe?",
        "Should this building be evacuated?",
        "Can you certify structural safety from this tile?",
    ],
    "X14": [
        "How much economic damage is this deformation causing?",
        "What is the expected insurance loss for this area?",
        "Can you estimate the repair cost from this tile?",
        "How much money will be lost because of this ground motion?",
        "What compensation value does this deformation imply?",
    ],
    "X15": [
        "What engineering intervention should be done here?",
        "Should pumps be installed to stop the subsidence?",
        "Do you recommend reinforcement or repair work for this tile?",
        "What construction action should authorities take?",
        "Should operations be stopped because of this deformation?",
    ],
    "X21": [
        "Is the specific building at this address affected?",
        "Can you assess this road segment exactly?",
        "Does this named asset have a deformation problem?",
        "Can you judge this parcel from the tile data?",
        "What is the condition of this exact facility?",
    ],
    "X22": [
        "What is happening at this exact coordinate?",
        "Can you give the motion for a single scatterer point?",
        "Which sub-cell has the precise maximum deformation?",
        "Can you make a pixel-level conclusion here?",
        "What is the exact point-level risk?",
    ],
    "X23": [
        "What is the horizontal displacement in this area?",
        "Can you report north-south motion from this tile?",
        "What is the full 3D displacement vector here?",
        "Can you describe a displacement component not included in the data?",
        "What is the unsupported motion direction for this area?",
    ],
    "X24": [
        "What land use is visible in this tile?",
        "Is this deformation under a railway or a building?",
        "Can you identify the geology from imagery?",
        "What infrastructure is present in this area?",
        "Can you interpret satellite imagery or land cover here?",
    ],
    "X25": [
        "What is happening in this area today?",
        "Is this tile currently moving right now?",
        "Can you give live deformation status?",
        "What changed after the EGMS observation window?",
        "Is there an active emergency at this location now?",
    ],
    "X26": [
        "Is this the worst subsidence site in Europe?",
        "Which tile is the most dangerous overall?",
        "Can you rank this exact area against every other possible location?",
        "Is this the single highest-risk site?",
        "Where does this tile rank globally among all assets?",
    ],
    "X31": [
        "Does the learned representation prove a physical hazard type?",
        "Can I treat the representation profile as ground truth?",
        "Does the embedding rarity directly mean geological danger?",
        "Is the reference profile a confirmed engineering diagnosis?",
        "Can the representation result replace measured monitoring evidence?",
    ],
    "X32": [
        "Does embedding similarity prove the real-world cause?",
        "Can representation rarity explain why the ground is moving?",
        "Does the anchor assignment prove a physical mechanism?",
        "Can the learned neighborhood establish causality?",
        "Does the representation show what caused the deformation?",
    ],
    "X33": [
        "What exact physical meaning does this hidden model feature have?",
        "Can you assign a certain geological meaning to this latent coordinate?",
        "Which model neuron proves the deformation mechanism?",
        "Can you explain every hidden model feature as a fixed monitoring variable?",
        "Does a specific embedding coordinate have a guaranteed physical meaning?",
    ],
}


@dataclass(frozen=True)
class TaskRecord:
    task_id: str
    family: str
    name: str
    target_column: str
    label_type: str
    probe_applicable: bool
    construct: bool = False
    trigger: str = ""
    refusal_reason: str = ""
    supported_redirect: str = ""
    response_template: str = ""


def load_meta(path: Path = DEFAULT_META) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def load_labels(path: Path = DEFAULT_LABELS) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["tile_id"] = df["tile_id"].astype(str)
    df["split"] = df["split"].astype(str)
    return df


def _read_x_rows(root: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for family in ("x1", "x2", "x3"):
        path = root / family / f"{family}_final_table.csv"
        df = pd.read_csv(path)
        for row in df.to_dict("records"):
            out[str(row["task_id"])] = row
    return out


def load_task_records(meta_path: Path = DEFAULT_META, tasks_root: Path = DEFAULT_TASKS_DIR) -> list[TaskRecord]:
    meta = load_meta(meta_path)
    x_rows = _read_x_rows(Path(tasks_root))
    records: list[TaskRecord] = []
    for task in meta["tasks"]:
        task_id = str(task["id"])
        if task.get("probe_applicable", False):
            records.append(TaskRecord(
                task_id=task_id,
                family=str(task.get("family", task_id[0])),
                name=str(task.get("name", task_id)),
                target_column=str(task.get("target_column", task_id)),
                label_type=str(task.get("label_type", "categorical")),
                probe_applicable=True,
                construct=bool(task.get("construct", False)),
            ))
            continue
        x = x_rows[task_id]
        records.append(TaskRecord(
            task_id=task_id,
            family="X",
            name=str(task.get("name", x.get("target_column", task_id))).replace("_", " "),
            target_column=str(x["target_column"]),
            label_type="refusal",
            probe_applicable=False,
            trigger=str(x.get("trigger", "")),
            refusal_reason=str(x.get("refusal_reason", "")),
            supported_redirect=str(x.get("supported_redirect", "")),
            response_template=str(x.get("response_template", "")),
        ))
    return records


def load_task_config(path: Path | str) -> dict[str, Any]:
    """Load a task manifest shared by training and generation evaluation."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    maintain = [str(t) for t in data.get("maintain", [])]
    focus = [str(t) for t in data.get("focus", [])]
    numeric_auxiliary = [str(t) for t in data.get("numeric_auxiliary", focus)]
    numeric_balance = [str(t) for t in data.get("numeric_balance", focus)]
    availability_balance = [str(t) for t in data.get("availability_balance", [])]
    task_ids = maintain + focus
    if not task_ids:
        raise ValueError(f"task config has no maintain/focus tasks: {path}")
    duplicates = sorted({t for t in task_ids if task_ids.count(t) > 1})
    if duplicates:
        raise ValueError(f"task config contains duplicate task IDs: {duplicates}")
    invalid_availability = sorted(set(availability_balance) - set(focus))
    if invalid_availability:
        raise ValueError(
            "availability-balanced tasks must also be focus tasks: "
            f"{invalid_availability}"
        )
    invalid_numeric_auxiliary = sorted(set(numeric_auxiliary) - set(focus))
    if invalid_numeric_auxiliary:
        raise ValueError(
            "numeric-auxiliary tasks must also be focus tasks: "
            f"{invalid_numeric_auxiliary}"
        )
    invalid_numeric_balance = sorted(set(numeric_balance) - set(focus))
    if invalid_numeric_balance:
        raise ValueError(
            "numeric-balanced tasks must also be focus tasks: "
            f"{invalid_numeric_balance}"
        )
    invalid_availability_auxiliary = sorted(
        set(availability_balance) - set(numeric_auxiliary)
    )
    if invalid_availability_auxiliary:
        raise ValueError(
            "availability-balanced tasks must also be numeric-auxiliary tasks: "
            f"{invalid_availability_auxiliary}"
        )
    data["maintain"] = maintain
    data["focus"] = focus
    data["numeric_auxiliary"] = numeric_auxiliary
    data["numeric_balance"] = numeric_balance
    data["availability_balance"] = availability_balance
    data["task_ids"] = task_ids
    return data


def load_qa_audit(path: Path | str, required_phrases: int = N_PHRASES) -> dict[str, Any]:
    """Load the passed QA gate and return approved phrase IDs by task."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("status") != "pass":
        raise ValueError(f"QA audit has not passed: {path}")
    if data.get("answer_protocol") != ANSWER_PROTOCOL:
        raise ValueError(
            f"QA audit uses {data.get('answer_protocol')!r}, expected {ANSWER_PROTOCOL!r}: {path}"
        )
    raw = data.get("allowed_phrase_ids", {})
    approved: dict[str, list[int]] = {}
    for task_id, phrase_ids in raw.items():
        ids = [int(value) for value in phrase_ids]
        if len(ids) < required_phrases:
            raise ValueError(
                f"QA audit approves only {len(ids)} phrasings for {task_id}; "
                f"{required_phrases} required"
            )
        approved[str(task_id)] = ids[:required_phrases]
    if not approved:
        raise ValueError(f"QA audit contains no approved phrase pools: {path}")
    data["approved_phrase_ids"] = approved
    return data


def task_lookup(records: Iterable[TaskRecord]) -> dict[str, TaskRecord]:
    return {r.task_id: r for r in records}


def humanize_label(value: Any) -> str:
    if value is None:
        return MISSING_VALUE
    s = str(value)
    if s in ("", "nan", "None", "<NA>", "NaN"):
        return MISSING_VALUE
    out = s.replace("_", " ").replace("|", " / ")
    out = out.replace("encoder", "representation")
    out = out.replace("non uplift", "non-uplift")
    out = out.replace("high mid", "high-mid").replace("low mid", "low-mid")
    return out


def _grid_point_text(token: str) -> str | None:
    match = re.fullmatch(r"r(\d+)c(\d+)", token)
    if not match:
        return None
    row = int(match.group(1))
    col_text = match.group(2)
    col = int(col_text)
    # The released C32 vocabulary can append a direction code to the first grid
    # token, for example r4c11 for grid column 1. Coordinates remain 0..7.
    if col > 7 and col_text.endswith("1") and int(col_text[:-1]) <= 7:
        col = int(col_text[:-1])
    if not (0 <= row <= 7 and 0 <= col <= 7):
        return None
    return f"row {row + 1}, column {col + 1}"


def categorical_label_description(task_id: str, label: Any) -> str:
    raw = str(label)
    if task_id in ("C13", "D35"):
        if raw == "none":
            return "not detected"
        point = _grid_point_text(raw)
        if point:
            return f"{point} of the 8-by-8 monitoring grid"
    if task_id == "C32":
        parts = raw.split("-", 1)
        if len(parts) == 2:
            first = _grid_point_text(parts[0])
            second = _grid_point_text(parts[1])
            if first and second:
                return f"between {first} and {second} of the 8-by-8 monitoring grid"
    return CATEGORICAL_LABEL_DESCRIPTIONS.get(task_id, {}).get(raw, humanize_label(raw))


def render_categorical_answer(task: TaskRecord, label: Any) -> str:
    description = categorical_label_description(task.task_id, label)
    pattern = CATEGORICAL_ANSWER_PATTERNS.get(task.task_id)
    if pattern:
        return pattern.format(description=description)
    return f"For this area, {sentence_lc(task.name.replace('_', ' '))} is {description}."


def categorical_label_aliases(task: TaskRecord, label: Any) -> list[str]:
    """Return canonical and user-facing phrases accepted for a class label."""
    raw = str(label)
    aliases = {
        raw,
        humanize_label(raw),
        categorical_label_description(task.task_id, raw),
        render_categorical_answer(task, raw),
    }
    return sorted(a for a in aliases if a and a != MISSING_VALUE)


def user_text(task: TaskRecord) -> tuple[str, str, str]:
    if task.task_id in TASK_USER_TEXT:
        return TASK_USER_TEXT[task.task_id]
    topic = task.name.replace("_", " ")
    query = f"what does this area show about {topic}"
    answer_subject = topic[:1].upper() + topic[1:]
    return topic, query, answer_subject


def sentence_lc(text: str) -> str:
    return text[:1].lower() + text[1:] if text else text


def clean_user_text(text: str) -> str:
    out = str(text)
    out = out.replace("EGMS-QA", "This monitoring product")
    out = out.replace("EGMS-QA", "This monitoring product")
    out = out.replace("the task set", "this monitoring product")
    out = out.replace("The task set", "This monitoring product")
    out = out.replace("encoder evidence", "learned representation evidence")
    out = out.replace("A token dimension or latent coordinate", "A hidden model feature or latent coordinate")
    out = out.replace("a token dimension or latent coordinate", "a hidden model feature or latent coordinate")
    out = out.replace("attribution, probe, or validation evidence", "external validation evidence")
    out = out.replace("S-group outputs", "representation outputs")
    out = out.replace("S-group", "representation")
    return out


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except Exception:
        pass
    if isinstance(value, float):
        return math.isnan(value)
    return str(value) in ("", "nan", "None", "<NA>", "NaN")


def infer_unit(task: TaskRecord) -> str:
    t = task.target_column.lower()
    if "fraction" in t or "coherence" in t or "probability" in t or "score" in t:
        return ""
    if "distance" in t or "margin" in t or "gap_p" in t:
        return ""
    if "mm_yr2" in t or "yr2" in t or "acc_abs" in t or "acceleration" in t or "intensification" in t:
        return "mm/yr^2"
    if "mm_yr" in t or "velocity" in t:
        return "mm/yr"
    if t.endswith("_mm") or "_mm_" in t or "amplitude_change_mm" in t or "seasonality" in t:
        return "mm"
    if "days" in t:
        return "days"
    if "year" in t:
        return "year"
    if "mse_z" in t:
        return "z^2"
    return ""


def format_number(value: Any, task: TaskRecord) -> str:
    v = float(value)
    t = task.target_column.lower()
    if "year" in t:
        return f"{v:.2f}"
    if "angular_drift" in t:
        return f"{v:.5f}"
    if "fraction" in t or "coherence" in t or "probability" in t or "score" in t:
        return f"{v:.3f}"
    if "distance" in t or "margin" in t or "gap_p" in t:
        return f"{v:.3f}"
    if abs(v) < 1:
        return f"{v:.3f}"
    if abs(v) < 10:
        return f"{v:.2f}"
    return f"{v:.1f}"


def supervision_target(row: dict[str, Any], target_format: str = "natural") -> str:
    """Return the visible natural answer used as the complete LM target."""
    if target_format not in TARGET_FORMATS:
        raise ValueError(f"unknown target format: {target_format}")
    return str(row["answer"])


def response_format_instruction(target_format: str = "natural") -> str:
    """Natural-only EGMS-QA uses no hidden response-format instruction."""
    if target_format not in TARGET_FORMATS:
        raise ValueError(f"unknown target format: {target_format}")
    return ""


def render_probe_row(
    tile_id: str,
    split: str,
    task: TaskRecord,
    value: Any,
    phrase_idx: int,
    numeric_answer_style: str = "standard",
) -> dict[str, Any]:
    if numeric_answer_style not in NUMERIC_ANSWER_STYLES:
        raise ValueError(f"unknown numeric answer style: {numeric_answer_style}")
    topic, query, answer_subject = user_text(task)
    qtmpl = USER_QUESTION_TEMPLATES[phrase_idx % len(USER_QUESTION_TEMPLATES)]
    question = qtmpl.format(
        topic=topic,
        query=query,
        query_lc=sentence_lc(query),
    )
    if _is_missing(value):
        answer = MISSING_ANSWER
        answer_value: Any = MISSING_VALUE
        answer_type = "missing"
        rendered_target_value = None
        decision_value = "not_available"
    elif task.label_type == "numeric":
        unit = infer_unit(task)
        unit_text = f" {unit}" if unit else ""
        vstr = format_number(value, task)
        if numeric_answer_style == "concise":
            if task.task_id == "D14":
                answer = f"The strongest change point is around the year {vstr}."
            else:
                answer = f"{answer_subject} is {vstr}{unit_text}."
        else:
            natural_template = NATURAL_NUMERIC_ANSWERS.get(task.task_id)
            if natural_template:
                answer = natural_template.format(value=vstr)
            else:
                answer = f"{answer_subject} is {vstr}{unit_text}."
        answer_value = float(value)
        answer_type = "numeric"
        rendered_target_value = vstr
        decision_value = None
    else:
        answer = render_categorical_answer(task, value)
        answer_value = str(value)
        answer_type = "categorical"
        rendered_target_value = None
        decision_value = str(value)
    return {
        "id": f"{task.task_id}:{tile_id}:p{phrase_idx:02d}",
        "tile_id": tile_id,
        "task": task.task_id,
        "question": question,
        "answer": answer,
        "answer_value": answer_value,
        "answer_type": answer_type,
        "rendered_target_value": rendered_target_value,
        "decision_value": decision_value,
        "target_column": task.target_column,
        "phrase_id": phrase_idx,
        "split": split,
    }


def render_x_row(tile_id: str, split: str, task: TaskRecord, phrase_idx: int) -> dict[str, Any]:
    examples = X_USER_QUESTIONS.get(task.task_id) or [task.trigger.rstrip(".")]
    base = examples[phrase_idx % len(examples)].rstrip("?.! ")
    wrappers = [
        "{base}?",
        "For this area, {base_lc}?",
        "Can you answer this from the EGMS record: {base_lc}?",
        "A monitoring user asks, \"{base}?\" What should you say?",
    ]
    question = wrappers[(phrase_idx // max(len(examples), 1)) % len(wrappers)].format(
        base=base,
        base_lc=sentence_lc(base),
    )
    response = NATURAL_X_ANSWERS.get(task.task_id, clean_user_text(task.response_template))
    answer = response.strip()
    return {
        "id": f"{task.task_id}:{tile_id}:p{phrase_idx:02d}",
        "tile_id": tile_id,
        "task": task.task_id,
        "question": question,
        "answer": answer,
        "answer_value": "refusal",
        "answer_type": "refusal",
        "rendered_target_value": None,
        "decision_value": "refusal",
        "target_column": task.target_column,
        "phrase_id": phrase_idx,
        "split": split,
    }


def render_row(
    label_row: pd.Series,
    task: TaskRecord,
    phrase_idx: int,
    numeric_answer_style: str = "standard",
) -> dict[str, Any]:
    tile_id = str(label_row["tile_id"])
    split = str(label_row["split"])
    if task.probe_applicable:
        return render_probe_row(
            tile_id,
            split,
            task,
            label_row.get(task.task_id),
            phrase_idx,
            numeric_answer_style=numeric_answer_style,
        )
    return render_x_row(tile_id, split, task, phrase_idx)


def logical_row_count(n_tiles: int, n_tasks: int, n_phrases: int) -> int:
    return int(n_tiles) * int(n_tasks) * int(n_phrases)
