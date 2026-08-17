"""Scorer tests — deterministic scorers on crafted outputs, judges on a mock model."""

import json

import pytest
from inspect_ai.model import ModelOutput, get_model
from inspect_ai.scorer import Target

from listing_gen.evals.scorers import (
    amenity_grounding,
    claim_groundedness,
    copy_quality,
    forbidden_claims,
    numeric_grounding,
    structure_valid,
)
from listing_gen.models import ListingContent

from .conftest import make_state

TARGET = Target("")


def mock_judge(response: str):
    """An Inspect model that replays a canned judge response, offline."""
    return get_model(
        "mockllm/model",
        custom_outputs=[ModelOutput.from_content(model="mockllm", content=response)],
    )


# ---------------------------------------------------------------- structure


@pytest.mark.asyncio
async def test_structure_valid_passes_good_listing(good_listing, villa_raw) -> None:
    state = make_state(good_listing.model_dump_json(), villa_raw)
    score = await structure_valid()(state, TARGET)
    assert score.value == 1.0


@pytest.mark.asyncio
async def test_structure_valid_catches_violations(good_listing, villa_raw) -> None:
    bad = good_listing.model_copy(
        update={
            "hero_headline": "x" * 100,
            "highlights": ["only one"],
            "about_this_place": "<b>Too short</b> and has HTML.",
        }
    )
    score = await structure_valid()(make_state(bad.model_dump_json(), villa_raw), TARGET)
    assert score.value == pytest.approx(1 / 5)
    assert "headline" in score.explanation
    assert "HTML" in score.explanation


@pytest.mark.asyncio
async def test_structure_valid_enforces_prompt_word_bounds(good_listing, villa_raw) -> None:
    # the prompt contract says 60-140 words; the scorer must enforce exactly that
    for words, ok in [(59, False), (60, True), (140, True), (141, False)]:
        sized = good_listing.model_copy(update={"about_this_place": "word " * words})
        score = await structure_valid()(make_state(sized.model_dump_json(), villa_raw), TARGET)
        assert (score.value == 1.0) == ok, f"{words} words should {'pass' if ok else 'fail'}"


@pytest.mark.asyncio
async def test_structure_valid_zero_on_unparseable(villa_raw) -> None:
    score = await structure_valid()(make_state("not json", villa_raw), TARGET)
    assert score.value == 0.0


# ---------------------------------------------------------------- numbers


@pytest.mark.asyncio
async def test_numeric_grounding_accepts_input_numbers(good_listing, villa_raw) -> None:
    score = await numeric_grounding()(make_state(good_listing.model_dump_json(), villa_raw), TARGET)
    assert score.value == 1.0


@pytest.mark.asyncio
async def test_numeric_grounding_ignores_rating_scale(good_listing, villa_raw) -> None:
    scaled = good_listing.model_copy(
        update={"about_this_place": "Guests rate the villa 4.96 out of 5 across 47 reviews. " * 5}
    )
    score = await numeric_grounding()(make_state(scaled.model_dump_json(), villa_raw), TARGET)
    assert score.value == 1.0


@pytest.mark.asyncio
async def test_numeric_grounding_flags_invented_numbers(good_listing, villa_raw) -> None:
    bad = good_listing.model_copy(
        update={"about_this_place": "Just 500 m from the beach, sleeps 6 guests. " * 10}
    )
    score = await numeric_grounding()(make_state(bad.model_dump_json(), villa_raw), TARGET)
    assert score.value < 1.0
    assert "500" in score.explanation


# ---------------------------------------------------------------- amenities


@pytest.mark.asyncio
async def test_amenity_grounding_accepts_real_amenities(good_listing, villa_raw) -> None:
    score = await amenity_grounding()(make_state(good_listing.model_dump_json(), villa_raw), TARGET)
    assert score.value == 1.0


@pytest.mark.asyncio
async def test_amenity_grounding_flags_invented_amenity(good_listing, villa_raw) -> None:
    bad = good_listing.model_copy(
        update={
            "amenity_descriptions": {
                "Private pool": "A pool.",
                "Hot tub": "Invented — the villa has no hot tub.",
            }
        }
    )
    score = await amenity_grounding()(make_state(bad.model_dump_json(), villa_raw), TARGET)
    assert score.value == pytest.approx(1 / 2)
    assert "Hot tub" in score.explanation


# ---------------------------------------------------------------- claims


@pytest.mark.asyncio
async def test_forbidden_claims_vacuous_without_expectations(good_listing, villa_raw) -> None:
    score = await forbidden_claims()(make_state(good_listing.model_dump_json(), villa_raw), TARGET)
    assert score.value == 1.0


@pytest.mark.asyncio
async def test_forbidden_claims_catches_leak(good_listing, villa_raw) -> None:
    state = make_state(
        good_listing.model_dump_json(), villa_raw, forbidden_claims=["beach", "hot tub"]
    )
    # good_listing mentions no beach/hot tub -> passes
    assert (await forbidden_claims()(state, TARGET)).value == 1.0

    leaky = good_listing.model_copy(update={"hero_headline": "Beachfront villa with pool"})
    state = make_state(leaky.model_dump_json(), villa_raw, forbidden_claims=["beach", "hot tub"])
    score = await forbidden_claims()(state, TARGET)
    assert score.value == pytest.approx(1 / 2)
    assert "beach" in score.explanation


@pytest.mark.asyncio
async def test_forbidden_claims_positive_assertions_spare_honest_paraphrase(
    good_listing, villa_raw
) -> None:
    # fixture phrases are positive assertions, so copy that honestly relays a
    # verified review ("not near a beach") must not be scored as a leak
    honest = good_listing.model_copy(
        update={"about_this_place": "One guest notes it is nowhere near a beach. " * 15}
    )
    state = make_state(
        honest.model_dump_json(), villa_raw, forbidden_claims=["beachfront", "beach access"]
    )
    assert (await forbidden_claims()(state, TARGET)).value == 1.0


# ---------------------------------------------------------------- judges


@pytest.mark.asyncio
async def test_claim_groundedness_scores_fraction_supported(good_listing, villa_raw) -> None:
    judge_response = json.dumps(
        {
            "claims": [
                {"claim": "private pool", "verdict": "supported", "evidence": "amenities"},
                {"claim": "sleeps 6", "verdict": "supported", "evidence": "rental_info"},
                {"claim": "5 min from beach", "verdict": "unsupported", "evidence": "not in data"},
            ]
        }
    )
    scorer_fn = claim_groundedness(model=mock_judge(judge_response))
    score = await scorer_fn(make_state(good_listing.model_dump_json(), villa_raw), TARGET)
    assert score.value == pytest.approx(2 / 3)
    assert "5 min from beach" in score.explanation


@pytest.mark.asyncio
async def test_claim_groundedness_handles_bad_judge_output(good_listing, villa_raw) -> None:
    scorer_fn = claim_groundedness(model=mock_judge("I refuse to answer in JSON"))
    score = await scorer_fn(make_state(good_listing.model_dump_json(), villa_raw), TARGET)
    assert score.value == 0.0
    assert "unparseable" in score.explanation


@pytest.mark.asyncio
async def test_copy_quality_averages_rubric(good_listing, villa_raw) -> None:
    judge_response = json.dumps(
        {"specificity": 5, "tone": 4, "clarity": 4, "cliche_avoidance": 5, "comment": "Solid."}
    )
    scorer_fn = copy_quality(model=mock_judge(judge_response))
    score = await scorer_fn(make_state(good_listing.model_dump_json(), villa_raw), TARGET)
    assert score.value == pytest.approx(18 / 20)
    assert score.metadata["ratings"]["tone"] == 4


@pytest.mark.asyncio
async def test_copy_quality_recovers_ratings_from_malformed_judge_json(
    good_listing, villa_raw
) -> None:
    # real failure observed in a live run: unescaped quotes inside the comment
    # break json.loads, but the integer ratings are still recoverable
    malformed = (
        '{"specificity": 3, "tone": 4, "clarity": 4, "cliche_avoidance": 4, '
        '"comment": "calls it both a "studio" and "1-bedroom", a contradiction"}'
    )
    scorer_fn = copy_quality(model=mock_judge(malformed))
    score = await scorer_fn(make_state(good_listing.model_dump_json(), villa_raw), TARGET)
    assert score.value == pytest.approx(15 / 20)
    assert "recovered" in score.explanation


@pytest.mark.asyncio
async def test_claim_groundedness_recovers_verdicts_from_malformed_judge_json(
    good_listing, villa_raw
) -> None:
    malformed = (
        '{"claims": [{"claim": "has a "studio" layout", "verdict": "unsupported", "evidence": "x"}, '
        '{"claim": "sleeps 6", "verdict": "supported", "evidence": "rental_info"}]}'
    )
    scorer_fn = claim_groundedness(model=mock_judge(malformed))
    score = await scorer_fn(make_state(good_listing.model_dump_json(), villa_raw), TARGET)
    assert score.value == pytest.approx(1 / 2)


@pytest.mark.asyncio
async def test_judges_zero_on_unparseable_listing(villa_raw) -> None:
    scorer_fn = copy_quality(model=mock_judge("irrelevant"))
    score = await scorer_fn(make_state("garbage output", villa_raw), TARGET)
    assert score.value == 0.0
