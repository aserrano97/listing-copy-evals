from listing_gen.models import PropertyData
from listing_gen.prompts import build_user_prompt, strip_html


def test_strip_html_removes_tags_and_entities() -> None:
    html_text = "<p>Bright <strong>loft</strong> &amp; garden</p><br/>"
    assert strip_html(html_text) == "Bright loft & garden"


def test_prompt_contains_verified_facts(villa: PropertyData) -> None:
    prompt = build_user_prompt(villa)
    assert "sleeps 6, 3 bedroom(s), 2 bathroom(s)" in prompt
    assert "Moraira, Spain" in prompt
    assert "Private pool" in prompt
    assert "47 review(s), average score 4.96" in prompt
    assert "Check-in from 4 PM" in prompt


def test_owner_text_is_quarantined_from_facts(villa: PropertyData) -> None:
    prompt = build_user_prompt(villa)
    facts, _, owner = prompt.partition("OWNER-PROVIDED TEXT")
    assert owner, "prompt must contain the owner text section"
    assert villa.description.description.split(".")[0] not in facts


def test_sparse_property_prompt_omits_missing_data() -> None:
    prop = PropertyData.model_validate(
        {
            "property_id": 1,
            "property_name": "Bare Studio",
            "property_type": "NormalApartment",
            "rental_info": {"max_guests": 2, "bedrooms": 1, "bathrooms": 1},
            "location": {"city": "Porto", "country": "Portugal"},
        }
    )
    prompt = build_user_prompt(prop)
    assert "Guest reviews: none yet" in prompt
    assert "Cancellation policy" not in prompt
    assert "(none provided)" in prompt


def test_html_in_reviews_is_stripped(villa: PropertyData) -> None:
    villa.reviews = ["<b>Great</b> stay &amp; lovely pool"]
    prompt = build_user_prompt(villa)
    assert "<b>" not in prompt
    assert "Great stay & lovely pool" in prompt
