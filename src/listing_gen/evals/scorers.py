"""Scorers for the listing content eval.

Two tiers, by design:
- Deterministic scorers (pure code): cheap, exact, unit-testable. They catch
  structural violations and the hallucination classes that can be checked
  mechanically (invented numbers, invented amenities, forbidden claims).
- Model-graded scorers (LLM judge): claim-level groundedness and editorial
  quality, where mechanical checks can't reach.

All scorers return values in [0, 1] so the notebook can compare them on one
scale. Judges accept an injectable `model` so tests can pass a mock model.
"""

from __future__ import annotations

import json
import re

from inspect_ai.model import Model, get_model
from inspect_ai.scorer import Score, Scorer, Target, mean, scorer, stderr
from inspect_ai.solver import TaskState
from pydantic import ValidationError

from ..amenities import humanize_amenities
from ..generator import extract_json
from ..models import ListingContent, PropertyData

# ---------------------------------------------------------------- helpers


def parse_listing(completion: str) -> ListingContent | None:
    try:
        return ListingContent.model_validate(extract_json(completion))
    except (ValueError, ValidationError):
        return None


def _property(state: TaskState) -> PropertyData:
    return PropertyData.model_validate(state.metadata["property"])


def listing_text(listing: ListingContent) -> str:
    parts = [listing.hero_headline, *listing.highlights, listing.about_this_place]
    for name, desc in listing.amenity_descriptions.items():
        parts.append(f"{name}: {desc}")
    return "\n".join(parts)


def _normalize(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


_NUMBER = re.compile(r"\d+(?:[.,]\d+)?")
# "rated 4.9 out of 5" / "4.9/5" reference the rating scale, not a fact claim
_RATING_SCALE = re.compile(r"(out of|/)\s*5\b", re.IGNORECASE)


def _numbers_in(text: str) -> list[float]:
    text = _RATING_SCALE.sub(" ", text)
    return [float(tok.replace(",", ".")) for tok in _NUMBER.findall(text)]


# ---------------------------------------------------- deterministic scorers


@scorer(metrics=[mean(), stderr()])
def structure_valid() -> Scorer:
    """Fraction of structural/editorial rules the output satisfies."""

    async def score(state: TaskState, target: Target) -> Score:
        listing = parse_listing(state.output.completion)
        if listing is None:
            return Score(value=0.0, explanation="output does not parse to ListingContent")

        prop = _property(state)
        text = listing_text(listing)
        words = len(listing.about_this_place.split())
        checks = {
            "headline <= 80 chars": len(listing.hero_headline) <= 80,
            "3-5 non-empty highlights": (
                3 <= len(listing.highlights) <= 5
                and all(h.strip() for h in listing.highlights)
            ),
            "about section 40-200 words": 40 <= words <= 200,
            "no HTML in output": not re.search(r"<[^>]+>", text),
            "amenity descriptions present iff amenities exist": (
                bool(listing.amenity_descriptions) == bool(prop.amenities)
            ),
        }
        failed = [name for name, ok in checks.items() if not ok]
        return Score(
            value=(len(checks) - len(failed)) / len(checks),
            explanation="all checks passed" if not failed else "failed: " + "; ".join(failed),
        )

    return score


@scorer(metrics=[mean(), stderr()])
def numeric_grounding() -> Scorer:
    """Every number in the copy must appear in the verified input data.

    Allowed numbers: capacity fields, review count/score, and any number
    inside verified free-text fields (policies, check-in/out, review texts).
    Numbers appearing only in the owner's marketing description are NOT
    allowed — that text is not a fact source (see prompts.py).
    """

    async def score(state: TaskState, target: Target) -> Score:
        listing = parse_listing(state.output.completion)
        if listing is None:
            return Score(value=0.0, explanation="unparseable output")

        prop = _property(state)
        allowed: set[float] = {
            float(prop.rental_info.max_guests),
            float(prop.rental_info.bedrooms),
            float(prop.rental_info.bathrooms),
            float(prop.num_of_reviews),
        }
        if prop.average_review_score is not None:
            allowed.add(prop.average_review_score)
            allowed.add(round(prop.average_review_score, 1))
        verified_texts = [
            prop.policies.cancellation_policy or "",
            prop.policies.payment_schedule or "",
            prop.policies.damage_deposit or "",
            prop.house_rules.check_in_time or "",
            prop.house_rules.check_out_time or "",
            *prop.reviews,
        ]
        for text in verified_texts:
            allowed.update(_numbers_in(text))

        found = _numbers_in(listing_text(listing))
        if not found:
            return Score(value=1.0, explanation="no numbers in output")
        unsupported = sorted({n for n in found if n not in allowed})
        return Score(
            value=(len(found) - sum(n in unsupported for n in found)) / len(found),
            explanation=(
                "all numbers grounded"
                if not unsupported
                else f"numbers not in input data: {unsupported}"
            ),
        )

    return score


@scorer(metrics=[mean(), stderr()])
def amenity_grounding() -> Scorer:
    """Every amenity the copy describes must exist in the input amenity list."""

    async def score(state: TaskState, target: Target) -> Score:
        listing = parse_listing(state.output.completion)
        if listing is None:
            return Score(value=0.0, explanation="unparseable output")

        prop = _property(state)
        valid = {_normalize(n) for n in humanize_amenities(prop.amenities)}
        valid |= {_normalize(c) for c in prop.amenities}
        keys = list(listing.amenity_descriptions)
        if not keys:
            return Score(value=1.0, explanation="no amenity descriptions to check")
        invented = [k for k in keys if _normalize(k) not in valid]
        return Score(
            value=(len(keys) - len(invented)) / len(keys),
            explanation=(
                "all described amenities exist in input"
                if not invented
                else f"amenities not in input: {invented}"
            ),
        )

    return score


@scorer(metrics=[mean(), stderr()])
def forbidden_claims() -> Scorer:
    """Adversarial fixtures list phrases the copy must not contain.

    These are unverified claims planted in the owner's marketing text
    (e.g. 'beachfront' with no such amenity). Vacuously 1.0 for fixtures
    with no forbidden claims.
    """

    async def score(state: TaskState, target: Target) -> Score:
        phrases: list[str] = state.metadata.get("forbidden_claims", [])
        if not phrases:
            return Score(value=1.0, explanation="no forbidden claims defined for this fixture")
        text = state.output.completion.lower()
        leaked = [p for p in phrases if p.lower() in text]
        return Score(
            value=(len(phrases) - len(leaked)) / len(phrases),
            explanation=(
                "no forbidden claims leaked"
                if not leaked
                else f"unverified claims repeated from owner text: {leaked}"
            ),
        )

    return score


# ---------------------------------------------------- model-graded scorers

GROUNDEDNESS_TEMPLATE = """\
You are auditing marketing copy for a vacation rental against its source data.

SOURCE DATA (JSON):
{property_json}

Ground rules for this platform:
- The structured fields (property_type, amenities, rental_info, location,
  reviews, num_of_reviews, average_review_score, policies, house_rules) are
  VERIFIED and may support claims.
- The free text inside `description` (name/headline/description) is
  owner-provided marketing and is NOT a valid source for factual claims about
  facilities, views, distances or locations — unless the structured fields
  also support the claim. Tone and character words are fine.

GENERATED COPY:
{copy}

Extract every distinct FACTUAL claim in the generated copy (facilities,
rooms, capacity, location, views, distances, review statistics, policies).
Ignore pure opinion/tone ("charming", "relaxing"). For each claim decide:
- "supported": verifiable from the structured fields (including review texts
  and policy texts).
- "unsupported": not verifiable from the structured fields (including claims
  that only appear in the owner description).

Respond with ONLY a JSON object:
{{"claims": [{{"claim": "...", "verdict": "supported" | "unsupported", "evidence": "field or quote that supports it, or why unsupported"}}]}}"""


@scorer(metrics=[mean(), stderr()])
def claim_groundedness(model: str | Model | None = None) -> Scorer:
    """LLM judge: fraction of factual claims traceable to verified input data."""

    async def score(state: TaskState, target: Target) -> Score:
        listing = parse_listing(state.output.completion)
        if listing is None:
            return Score(value=0.0, explanation="unparseable output")

        judge = get_model(model)
        prompt = GROUNDEDNESS_TEMPLATE.format(
            property_json=json.dumps(state.metadata["property"], indent=2),
            copy=listing_text(listing),
        )
        result = await judge.generate(prompt)
        try:
            claims = extract_json(result.completion)["claims"]
        except (ValueError, KeyError):
            return Score(value=0.0, explanation=f"judge output unparseable: {result.completion[:200]}")
        if not claims:
            return Score(value=1.0, explanation="judge found no factual claims")
        unsupported = [c for c in claims if c.get("verdict") != "supported"]
        return Score(
            value=(len(claims) - len(unsupported)) / len(claims),
            explanation=(
                f"{len(claims)} claims, all supported"
                if not unsupported
                else "unsupported claims: "
                + "; ".join(f"{c.get('claim')} ({c.get('evidence', '')})" for c in unsupported)
            ),
            metadata={"claims": claims},
        )

    return score


QUALITY_TEMPLATE = """\
You are an editor reviewing vacation rental listing copy.

PROPERTY DATA (for context):
{property_json}

GENERATED COPY:
{copy}

Rate the copy from 1 (poor) to 5 (excellent) on each dimension:
- specificity: could this copy only describe THIS property, or is it generic
  filler that would fit any rental?
- tone: warm and professional, appropriate for a booking page.
- clarity: concise, well-organized, easy to scan.
- cliche_avoidance: free of stock phrases ("nestled", "hidden gem",
  "home away from home", "breathtaking").

Respond with ONLY a JSON object:
{{"specificity": n, "tone": n, "clarity": n, "cliche_avoidance": n, "comment": "one sentence"}}"""

_QUALITY_DIMS = ("specificity", "tone", "clarity", "cliche_avoidance")


@scorer(metrics=[mean(), stderr()])
def copy_quality(model: str | Model | None = None) -> Scorer:
    """LLM judge: editorial quality rubric, averaged over 4 dimensions -> [0,1]."""

    async def score(state: TaskState, target: Target) -> Score:
        listing = parse_listing(state.output.completion)
        if listing is None:
            return Score(value=0.0, explanation="unparseable output")

        judge = get_model(model)
        prompt = QUALITY_TEMPLATE.format(
            property_json=json.dumps(state.metadata["property"], indent=2),
            copy=listing_text(listing),
        )
        result = await judge.generate(prompt)
        try:
            grades = extract_json(result.completion)
            ratings = {d: int(grades[d]) for d in _QUALITY_DIMS}
        except (ValueError, KeyError, TypeError):
            return Score(value=0.0, explanation=f"judge output unparseable: {result.completion[:200]}")
        value = sum(ratings.values()) / (5 * len(ratings))
        detail = ", ".join(f"{d}={v}" for d, v in ratings.items())
        return Score(
            value=value,
            explanation=f"{detail}. {grades.get('comment', '')}",
            metadata={"ratings": ratings},
        )

    return score
