"""Fixture loading: property JSON files -> Inspect MemoryDataset.

Fixture files under fixtures/properties/ conform exactly to the platform's
property data spec. Eval-side expectations (e.g. claims a listing must NOT
make) live separately in fixtures/expectations.json, keyed by property_id,
so the input data stays byte-for-byte what the platform would send.
"""

from __future__ import annotations

import json
from pathlib import Path

from inspect_ai.dataset import MemoryDataset, Sample

from ..models import PropertyData

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURES_DIR = REPO_ROOT / "fixtures"


def load_dataset(fixtures_dir: Path = FIXTURES_DIR) -> MemoryDataset:
    expectations_path = fixtures_dir / "expectations.json"
    expectations: dict[str, dict] = (
        json.loads(expectations_path.read_text()) if expectations_path.exists() else {}
    )

    samples: list[Sample] = []
    for path in sorted((fixtures_dir / "properties").glob("*.json")):
        raw = json.loads(path.read_text())
        prop = PropertyData.model_validate(raw)  # fail fast on a broken fixture
        expect = expectations.get(str(prop.property_id), {})
        samples.append(
            Sample(
                id=f"{prop.property_id}-{path.stem}",
                input=f"Generate listing content for: {prop.property_name} "
                f"({prop.property_type}, {prop.location.city})",
                metadata={
                    "property": raw,
                    "forbidden_claims": expect.get("forbidden_claims", []),
                },
            )
        )
    if not samples:
        raise FileNotFoundError(f"no property fixtures found under {fixtures_dir}")
    return MemoryDataset(samples, name="property-fixtures")
