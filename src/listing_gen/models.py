"""Data models for property input and generated listing content."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Description(BaseModel):
    name: str = ""
    headline: str = ""
    description: str = ""  # owner-provided free text, may contain HTML


class RentalInfo(BaseModel):
    max_guests: int
    bedrooms: int
    bathrooms: int


class Location(BaseModel):
    city: str
    country: str
    latitude: float | None = None
    longitude: float | None = None


class Policies(BaseModel):
    cancellation_policy: str | None = None
    payment_schedule: str | None = None
    damage_deposit: str | None = None


class HouseRules(BaseModel):
    check_in_time: str | None = None
    check_out_time: str | None = None


class PropertyData(BaseModel):
    """Input property object, per the platform data spec.

    Tolerant of sparse data: everything an owner might leave empty defaults
    to empty/None so the pipeline never crashes on a thin listing.
    """

    property_id: int
    property_name: str
    property_type: str
    description: Description = Description()
    amenities: list[str] = Field(default_factory=list)
    image_urls: list[str] = Field(default_factory=list)
    reviews: list[str] = Field(default_factory=list)
    num_of_reviews: int = 0
    average_review_score: float | None = None
    rental_info: RentalInfo
    location: Location
    policies: Policies = Policies()
    house_rules: HouseRules = HouseRules()


class ListingContent(BaseModel):
    """Structured marketing copy for a listing page.

    Deliberately permissive (types only): editorial constraints such as
    headline length or highlight count are checked by the eval scorers,
    so a rule violation becomes a *score*, not a crash.
    """

    hero_headline: str
    highlights: list[str]
    about_this_place: str
    amenity_descriptions: dict[str, str]
