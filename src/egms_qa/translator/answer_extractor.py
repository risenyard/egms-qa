"""Deterministic canonical-value extraction from visible EGMS-QA answers.

The extractor never sees the gold value. It receives the task schema, the
allowed categorical labels, and the model's visible natural-language answer.
Numeric quantities are discovered with quantulum3; task rules only select the
quantity with the expected unit or reject ambiguous output.
"""
from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any, Iterable

from quantulum3 import classifier as quantity_classifier
from quantulum3 import parser as quantity_parser

from egms_qa.qa_construction.qa_lib import (
    MISSING_VALUE,
    TaskRecord,
    categorical_label_aliases,
    format_number,
    infer_unit,
)


# The bundled classifier is unnecessary for this closed unit inventory and can
# be incompatible with the environment's scikit-learn version. Quantulum's
# deterministic unit disambiguator remains active.
quantity_classifier.USE_CLF = False

PARSED = "parsed"
UNPARSED = "unparsed"
AMBIGUOUS = "ambiguous"

REFUSAL_CUES = (
    "cannot",
    "unable",
    "not supported",
    "outside the supported",
    "insufficient for that request",
    "do not answer directly",
    "answer is to decline",
    "response is to decline",
    "response should be refusal",
    "scoped answer is refusal",
    "refuse the unsupported",
    "should refuse the unsupported",
    "inference is unsupported",
    "does not support that request",
    "should not make that claim",
    "does not include external imagery",
    "does not establish a causal mechanism",
    "do not establish a causal mechanism",
    "does not establish causal explanation",
    "has no exposure asset value vulnerability or economic loss model",
    "is not a forecasting model",
    "not a live feed",
    "does not support exact asset level judgment",
    "cannot replace structural safety assessment",
    "supported spatial scale is tile level",
    "can only answer displacement components",
    "can only use the supported egms components",
    "encoder representation constructs not external physical ground truth",
    "requires a defined ranking table",
    "require a defined ranking table",
    "not an intervention recommender",
    "do not have access",
    "would require external",
    "would require evidence outside",
    "requires external",
    "requires site specific",
    "requires an asset specific",
    "only report",
    "only describe",
    "only give supported historical monitoring facts",
)
MISSING_CUES = (
    "not available",
    "unavailable",
    "no valid result",
    "no valid output",
    "no valid value",
    "no valid tile level value",
    "no supported result",
    "no supported value",
    "cannot provide this result",
    "does not meet the requirements",
    "does not meet the validity condition",
    "not recorded",
    "gated off",
)

CONTRACTIONS = {
    "can't": "cannot",
    "cannot": "cannot",
    "isn't": "is not",
    "aren't": "are not",
    "doesn't": "does not",
    "don't": "do not",
    "won't": "will not",
}


@dataclass(frozen=True)
class ExtractionResult:
    status: str
    value: Any | None
    method: str
    evidence: tuple[str, ...] = ()
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["evidence"] = list(self.evidence)
        return data


@dataclass(frozen=True)
class QuantityCandidate:
    value: float
    surface: str
    start: int
    end: int
    unit_name: str


def normalize_text(value: Any) -> str:
    text = str(value).lower().replace("\u2019", "'").replace("\u2018", "'")
    for contraction, expanded in CONTRACTIONS.items():
        text = text.replace(contraction, expanded)
    text = text.replace("_", " ").replace("/", " per ").replace("-", " ")
    text = re.sub(r"[^a-z0-9+\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _contains_phrase(text: str, phrase: str) -> bool:
    padded_text = f" {text} "
    needle = f" {phrase} "
    phrase_tokens = set(phrase.split())
    phrase_is_negated = bool(phrase_tokens & {"no", "not", "without", "cannot"})
    start = 0
    while True:
        index = padded_text.find(needle, start)
        if index < 0:
            return False
        prefix_tokens = padded_text[:index].split()
        nearby_negation = bool(set(prefix_tokens[-3:]) & {"no", "not", "without"})
        if phrase_is_negated or not nearby_negation:
            return True
        start = index + 1


@lru_cache(maxsize=None)
def _normalized_label_aliases(task: TaskRecord, canonical: str) -> tuple[tuple[int, str, str], ...]:
    aliases = []
    for alias in categorical_label_aliases(task, canonical):
        normalized = normalize_text(alias)
        if normalized:
            aliases.append((len(normalized), alias, normalized))
    return tuple(aliases)


def _cue_result(text: str, cues: Iterable[str], value: str, method: str) -> ExtractionResult:
    normalized = normalize_text(text)
    padded_text = f" {normalized} "
    matches = tuple(
        cue for cue in cues
        if f" {normalize_text(cue)} " in padded_text
    )
    if matches:
        return ExtractionResult(PARSED, value, method, matches)
    return ExtractionResult(UNPARSED, None, method, reason=f"no {method} cue")


def extract_categorical(text: str, task: TaskRecord, labels: Iterable[Any]) -> ExtractionResult:
    normalized = normalize_text(text)
    matches: dict[str, tuple[int, str, str]] = {}
    for label in labels:
        canonical = str(label)
        best: tuple[int, str, str] | None = None
        for alias_length, alias, normalized_alias in _normalized_label_aliases(task, canonical):
            if not _contains_phrase(normalized, normalized_alias):
                continue
            candidate = (alias_length, alias, normalized_alias)
            if best is None or candidate[0] > best[0]:
                best = candidate
        if best is not None:
            matches[canonical] = best

    if not matches:
        return ExtractionResult(
            UNPARSED,
            None,
            "task_alias",
            reason="no task label alias matched the visible answer",
        )

    longest = max(length for length, _, _ in matches.values())
    winners = sorted(label for label, (length, _, _) in matches.items() if length == longest)
    evidence = tuple(matches[label][1] for label in winners)
    if len(winners) != 1:
        return ExtractionResult(
            AMBIGUOUS,
            None,
            "task_alias",
            evidence,
            reason=f"equally specific aliases matched labels: {', '.join(winners)}",
        )
    winner = winners[0]
    winner_alias = matches[winner][2]
    unrelated = sorted(
        label for label, (_, _, alias) in matches.items()
        if label != winner and alias not in winner_alias
    )
    if unrelated:
        all_labels = [winner, *unrelated]
        return ExtractionResult(
            AMBIGUOUS,
            None,
            "task_alias",
            tuple(matches[label][1] for label in all_labels),
            reason=f"independent aliases matched labels: {', '.join(all_labels)}",
        )
    return ExtractionResult(PARSED, winner, "task_alias", evidence)


def _is_structural_number(text: str, candidate: QuantityCandidate) -> bool:
    left = text[max(0, candidate.start - 8):candidate.start].lower()
    right = text[candidate.end:candidate.end + 12].lower()
    surface = candidate.surface.lower()
    if re.fullmatch(r"\s*\d+(?:\.\d+)?\s*[- ]?by[- ]?\s*\d+(?:\.\d+)?\s*", surface):
        return True
    if re.fullmatch(
        r"\s*[-+]?(?:\d+(?:\.\d*)?|\.\d+)\s*[-–]\s*"
        r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)\s*",
        surface,
    ):
        return True
    if re.search(r"(?:\^|\bz\s*\^\s*)$", left):
        return True
    if re.search(r"\b(?:row|column)\s*$", left):
        return True
    if re.match(r"\s*[- ]?by[- ]?\s*\d", right):
        return True
    return False


def _quantulum_candidates(text: str) -> tuple[list[QuantityCandidate], str]:
    try:
        quantities = quantity_parser.parse(str(text))
    except Exception as exc:
        return _regex_candidates(text), f"regex_fallback:{type(exc).__name__}"
    candidates = [
        QuantityCandidate(
            value=float(quantity.value),
            surface=str(quantity.surface),
            start=int(quantity.span[0]),
            end=int(quantity.span[1]),
            unit_name=str(quantity.unit.name),
        )
        for quantity in quantities
        if math.isfinite(float(quantity.value))
    ]
    return _merge_spoken_decimals(str(text), candidates), "quantulum3"


def _merge_spoken_decimals(text: str, candidates: list[QuantityCandidate]) -> list[QuantityCandidate]:
    """Repair quantulum splits such as 'minus one point four'."""
    merged: list[QuantityCandidate] = []
    index = 0
    while index < len(candidates):
        left = candidates[index]
        if index + 1 < len(candidates):
            right = candidates[index + 1]
            between = text[left.end:right.start]
            if re.fullmatch(r"\s*point\s*", between, re.IGNORECASE):
                right_integer = str(abs(int(right.value)))
                decimal = abs(right.value) / (10 ** len(right_integer))
                sign = -1.0 if left.value < 0 else 1.0
                merged.append(QuantityCandidate(
                    value=sign * (abs(left.value) + decimal),
                    surface=text[left.start:right.end],
                    start=left.start,
                    end=right.end,
                    unit_name=right.unit_name,
                ))
                index += 2
                continue
        merged.append(left)
        index += 1
    return merged


def _regex_candidates(text: str) -> list[QuantityCandidate]:
    pattern = r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?"
    return [
        QuantityCandidate(float(match.group()), match.group(), match.start(), match.end(), "unknown")
        for match in re.finditer(pattern, str(text))
    ]


def _unit_pattern(unit: str) -> re.Pattern[str] | None:
    millimetres = r"(?:mm|millimet(?:er|re)s?)"
    years = r"(?:yr|year)s?"
    if unit == "mm/yr^2":
        return re.compile(
            rf"{millimetres}\s*(?:/|per\s+){years}\s*(?:\^?\s*2|squared)\b",
            re.IGNORECASE,
        )
    if unit == "mm/yr":
        return re.compile(
            rf"{millimetres}\s*(?:/|per\s+){years}(?!\s*(?:\^?\s*2|squared))\b",
            re.IGNORECASE,
        )
    if unit == "mm":
        return re.compile(rf"{millimetres}(?!\s*(?:/|per\s+){years})\b", re.IGNORECASE)
    if unit == "days":
        return re.compile(r"\bdays?\b", re.IGNORECASE)
    if unit == "z^2":
        return re.compile(r"\bz\s*\^?\s*2\b", re.IGNORECASE)
    return None


def _has_expected_unit(text: str, candidate: QuantityCandidate, unit: str) -> bool:
    if unit == "year":
        left = text[max(0, candidate.start - 20):candidate.start]
        right = text[candidate.end:candidate.end + 12]
        return bool(re.search(r"\byear\s*$", left, re.IGNORECASE) or re.match(r"\s*years?\b", right, re.IGNORECASE))
    pattern = _unit_pattern(unit)
    if pattern is None:
        return False
    unit_window = f"{candidate.surface} {text[candidate.end:candidate.end + 20]}"
    return pattern.search(unit_window) is not None


def _is_fraction_task(task: TaskRecord) -> bool:
    column = task.target_column.lower()
    return any(key in column for key in ("fraction", "coherence", "probability"))


def _percent_value(text: str, candidate: QuantityCandidate, task: TaskRecord) -> float:
    if not _is_fraction_task(task):
        return candidate.value
    unit_window = f"{candidate.surface} {text[candidate.end:candidate.end + 20]}"
    if re.search(r"(?:%|\bpercent(?:age)?\b)", unit_window, re.IGNORECASE):
        return candidate.value / 100.0
    return candidate.value


def _has_percent_unit(text: str, candidate: QuantityCandidate) -> bool:
    unit_window = f"{candidate.surface} {text[candidate.end:candidate.end + 20]}"
    return bool(re.search(r"(?:%|\bpercent(?:age)?\b)", unit_window, re.IGNORECASE))


def _s12_literal_value(candidate: QuantityCandidate) -> float | None:
    """Read an explicit S12 decimal without Quantulum magnitude scaling."""
    if not re.search(r"\b(?:thousand|million|billion)\b", candidate.surface, re.IGNORECASE):
        return candidate.value
    match = re.search(
        r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?",
        candidate.surface,
    )
    return None if match is None else float(match.group())


def extract_numeric(text: str, task: TaskRecord) -> ExtractionResult:
    raw = str(text)
    candidates, method = _quantulum_candidates(raw)
    candidates = [candidate for candidate in candidates if not _is_structural_number(raw, candidate)]
    if not candidates:
        return ExtractionResult(UNPARSED, None, method, reason="no answer quantity found")

    expected_unit = infer_unit(task)
    selected = candidates
    selection_method = method
    if expected_unit:
        unit_matches = [candidate for candidate in candidates if _has_expected_unit(raw, candidate, expected_unit)]
        if not unit_matches:
            evidence = tuple(candidate.surface for candidate in candidates)
            return ExtractionResult(
                UNPARSED,
                None,
                method,
                evidence,
                reason=f"expected unit {expected_unit!r} not found",
            )
        selected = unit_matches
        selection_method = f"{method}+{expected_unit}"

    if len(selected) != 1:
        evidence = tuple(candidate.surface for candidate in selected)
        return ExtractionResult(
            AMBIGUOUS,
            None,
            selection_method,
            evidence,
            reason=f"{len(selected)} plausible answer quantities remain",
        )

    candidate = selected[0]
    if not expected_unit and candidate.unit_name not in {"dimensionless", "unknown"}:
        if not _has_percent_unit(raw, candidate):
            return ExtractionResult(
                UNPARSED,
                None,
                f"{selection_method}+dimensionless",
                (candidate.surface,),
                reason=f"physical unit {candidate.unit_name!r} is invalid for a dimensionless task",
            )
    if task.task_id == "S12":
        value = _s12_literal_value(candidate)
        if value is None:
            return ExtractionResult(
                UNPARSED,
                None,
                f"{selection_method}+s12_literal",
                (candidate.surface,),
                reason="S12 magnitude phrase contains no explicit numeric literal",
            )
        return ExtractionResult(
            PARSED,
            value,
            f"{selection_method}+s12_literal",
            (candidate.surface,),
        )
    value = _percent_value(raw, candidate, task)
    if _is_fraction_task(task) and not 0.0 <= value <= 1.0:
        return ExtractionResult(
            UNPARSED,
            None,
            f"{selection_method}+fraction_range",
            (candidate.surface,),
            reason="dimensionless fraction lies outside [0, 1]",
        )
    return ExtractionResult(PARSED, value, selection_method, (candidate.surface,))


def extract_answer(text: str, task: TaskRecord, labels: Iterable[Any] = ()) -> ExtractionResult:
    if task.label_type == "refusal" or not task.probe_applicable:
        return _cue_result(text, REFUSAL_CUES, "refusal", "refusal")
    missing = _cue_result(text, MISSING_CUES, MISSING_VALUE, "missing")
    if missing.status == PARSED:
        return missing
    if task.label_type == "numeric":
        return extract_numeric(text, task)
    return extract_categorical(text, task, labels)


def rendered_precision_tolerance(row: dict[str, Any], task: TaskRecord) -> float:
    rendered = str(row.get("rendered_target_value") or format_number(row["answer_value"], task))
    mantissa = rendered.lower().split("e", 1)[0]
    decimals = len(mantissa.rsplit(".", 1)[1]) if "." in mantissa else 0
    exponent = int(rendered.lower().split("e", 1)[1]) if "e" in rendered.lower() else 0
    return 0.5 * (10.0 ** (exponent - decimals))


def extraction_is_correct(row: dict[str, Any], task: TaskRecord, result: ExtractionResult) -> bool:
    if result.status != PARSED:
        return False
    # A missing target can belong to a nominally numeric task. It is a
    # categorical availability outcome, so it must be checked before numeric
    # conversion and renderer-precision scoring.
    if row.get("answer_type") == "missing":
        return str(result.value) == MISSING_VALUE
    if row.get("answer_type") == "numeric":
        try:
            predicted = float(result.value)
            expected = float(row.get("rendered_target_value") or format_number(row["answer_value"], task))
        except (TypeError, ValueError):
            return False
        tolerance = rendered_precision_tolerance(row, task)
        return math.isclose(predicted, expected, rel_tol=0.0, abs_tol=tolerance + 1e-12)
    return str(result.value) == str(row.get("answer_value"))
