import json

import pytest
from pydantic import ValidationError

from listing_gen.models import PropertyData

from .conftest import FIXTURES_DIR


def test_all_fixtures_validate() -> None:
    paths = sorted((FIXTURES_DIR / "properties").glob("*.json"))
    assert len(paths) >= 6
    for path in paths:
        prop = PropertyData.model_validate(json.loads(path.read_text()))
        assert prop.property_id > 0


def test_sparse_property_gets_safe_defaults() -> None:
    prop = PropertyData.model_validate(
        {
            "property_id": 1,
            "property_name": "Bare",
            "property_type": "NormalApartment",
            "rental_info": {"max_guests": 2, "bedrooms": 1, "bathrooms": 1},
            "location": {"city": "X", "country": "Y"},
        }
    )
    assert prop.amenities == []
    assert prop.reviews == []
    assert prop.average_review_score is None
    assert prop.policies.cancellation_policy is None


def test_missing_required_fields_rejected() -> None:
    with pytest.raises(ValidationError):
        PropertyData.model_validate({"property_id": 1, "property_name": "No rental info"})
