"""Mapping from internal amenity codes to guest-facing names.

Unknown codes fall back to de-camel-casing so a new platform code never
crashes the pipeline — but the fallback is flagged so evals can catch
codes that deserve a curated name.
"""

from __future__ import annotations

import re

KNOWN_AMENITIES: dict[str, str] = {
    "AirConditioning": "Air conditioning",
    "BalconyTerrace": "Balcony or terrace",
    "BathroomAndLaundry": "Bathroom & laundry",
    "BbqGrill": "BBQ grill",
    "CoffeeMachine": "Coffee machine",
    "DishWasher": "Dishwasher",
    "Fireplace": "Fireplace",
    "FreeParking": "Free parking",
    "GardenView": "Garden view",
    "Heating": "Heating",
    "HotTub": "Hot tub",
    "InternetBroadband": "Broadband internet (WiFi)",
    "KitchenFullyEquipped": "Fully equipped kitchen",
    "MicrowaveOven": "Microwave oven",
    "PetsAllowed": "Pets allowed",
    "PrivatePool": "Private pool",
    "SeaView": "Sea view",
    "SmartTv": "Smart TV",
    "TowelsAndLinen": "Towels & linen provided",
    "WashingMachine": "Washing machine",
}

_CAMEL_SPLIT = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


def humanize_amenity(code: str) -> str:
    """Guest-facing name for an amenity code (curated, or de-camel-cased)."""
    if code in KNOWN_AMENITIES:
        return KNOWN_AMENITIES[code]
    words = _CAMEL_SPLIT.sub(" ", code.strip()).split()
    return " ".join(words).capitalize() if words else code


def humanize_amenities(codes: list[str]) -> list[str]:
    return [humanize_amenity(c) for c in codes]


def unknown_codes(codes: list[str]) -> list[str]:
    """Codes without a curated name — surfaced so evals can flag them."""
    return [c for c in codes if c not in KNOWN_AMENITIES]
