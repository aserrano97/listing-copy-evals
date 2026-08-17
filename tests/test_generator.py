import json

import pytest

from listing_gen.generator import (
    GenerationError,
    ListingGenerator,
    StaticLLMClient,
    extract_json,
)
from listing_gen.models import ListingContent, PropertyData
from listing_gen.prompts import SYSTEM_PROMPT

VALID_JSON = json.dumps(
    {
        "hero_headline": "Family villa with private pool in Moraira",
        "highlights": ["Private pool", "Sleeps 6", "Sea views"],
        "about_this_place": "A lovely villa. " * 20,
        "amenity_descriptions": {"Private pool": "A pool."},
    }
)


def test_extract_json_tolerates_code_fences() -> None:
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Here you go:\n{"a": 1}') == {"a": 1}
    with pytest.raises(ValueError):
        extract_json("no json here")


@pytest.mark.asyncio
async def test_generate_parses_valid_output(villa: PropertyData) -> None:
    client = StaticLLMClient([VALID_JSON])
    result = await ListingGenerator(client).generate(villa)
    assert isinstance(result, ListingContent)
    assert result.highlights == ["Private pool", "Sleeps 6", "Sea views"]
    system, user = client.calls[0]
    assert system == SYSTEM_PROMPT
    assert "Villa Mar Blava" in user


@pytest.mark.asyncio
async def test_generate_retries_on_malformed_output(villa: PropertyData) -> None:
    client = StaticLLMClient(["not json at all", VALID_JSON])
    result = await ListingGenerator(client, max_retries=1).generate(villa)
    assert isinstance(result, ListingContent)
    assert len(client.calls) == 2
    retry_prompt = client.calls[1][1]
    assert "previous response was not valid" in retry_prompt


@pytest.mark.asyncio
async def test_generate_raises_after_exhausting_retries(villa: PropertyData) -> None:
    client = StaticLLMClient(["garbage", "more garbage"])
    with pytest.raises(GenerationError) as exc_info:
        await ListingGenerator(client, max_retries=1).generate(villa)
    assert exc_info.value.raw_output == "more garbage"


@pytest.mark.asyncio
async def test_generate_rejects_schema_violations(villa: PropertyData) -> None:
    missing_field = json.dumps({"hero_headline": "x", "highlights": []})
    client = StaticLLMClient([missing_field])
    with pytest.raises(GenerationError):
        await ListingGenerator(client, max_retries=0).generate(villa)
