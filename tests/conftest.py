"""Shared test fixtures. All tests run offline — no network, no API keys."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from listing_gen.models import ListingContent, PropertyData

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES_DIR = REPO_ROOT / "fixtures"


class FakeTaskState:
    """Minimal stand-in for inspect_ai's TaskState.

    Scorers only touch `state.output.completion` and `state.metadata`, so a
    small fake keeps scorer tests independent of Inspect's TaskState
    constructor churn.
    """

    class _Output:
        def __init__(self, completion: str) -> None:
            self.completion = completion

    def __init__(self, completion: str, metadata: dict) -> None:
        self.output = self._Output(completion)
        self.metadata = metadata


@pytest.fixture
def villa() -> PropertyData:
    raw = json.loads((FIXTURES_DIR / "properties" / "101_villa_rich.json").read_text())
    return PropertyData.model_validate(raw)


@pytest.fixture
def villa_raw() -> dict:
    return json.loads((FIXTURES_DIR / "properties" / "101_villa_rich.json").read_text())


@pytest.fixture
def good_listing(villa: PropertyData) -> ListingContent:
    """A listing that satisfies every rule for the villa fixture."""
    return ListingContent(
        hero_headline="Family villa with private pool and sea views in Moraira",
        highlights=[
            "Private pool and covered terrace with BBQ",
            "Sleeps 6 across 3 bedrooms and 2 bathrooms",
            "Rated 4.96 by 47 guests",
        ],
        about_this_place=(
            "Villa Mar Blava is a light-filled Mediterranean villa in Moraira, Spain, "
            "sleeping six guests across three bedrooms and two bathrooms. Days revolve "
            "around the private pool and the covered terrace, where guests fire up the "
            "BBQ for long dinners. Inside you'll find air conditioning, a fully equipped "
            "kitchen with dishwasher, and broadband internet for the odd remote-work "
            "morning. Guests rate the villa 4.96 across 47 reviews, praising the pool "
            "area, the sea views from the terrace and the easy free parking."
        ),
        amenity_descriptions={
            "Private pool": "A private pool at the heart of the garden.",
            "Sea view": "Sea views from the terrace.",
            "Broadband internet (WiFi)": "Reliable WiFi throughout the villa.",
        },
    )


def make_state(listing_json: str, property_raw: dict, **metadata) -> FakeTaskState:
    return FakeTaskState(listing_json, {"property": property_raw, **metadata})
