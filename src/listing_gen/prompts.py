"""Prompt construction for the listing content generator.

Grounding strategy: the prompt draws a hard line between VERIFIED FACTS
(structured platform fields — the only permitted source for factual claims)
and OWNER-PROVIDED TEXT (free-form description, which may contain HTML and
unverified marketing claims — usable for tone and character only). This is
what the groundedness evals test.
"""

from __future__ import annotations

import html
import re

from .amenities import humanize_amenities
from .models import PropertyData

SYSTEM_PROMPT = """\
You write marketing copy for vacation rental listing pages.

You will receive a property fact sheet with two sections:
- VERIFIED FACTS: structured platform data. This is the ONLY permitted source
  for factual claims (rooms, capacity, amenities, location, reviews, policies).
- OWNER-PROVIDED TEXT: the owner's own description. Use it ONLY to pick up the
  property's character and tone. If it claims a feature that is not in
  VERIFIED FACTS, you must NOT repeat that claim.

Rules:
1. Every factual claim in your copy must be traceable to VERIFIED FACTS.
   This includes location claims: do not call a place "central", "quiet" or
   "close to X" unless VERIFIED FACTS say so.
2. Never invent or embellish: no numbers, amenities, views, distances or
   nearby attractions beyond VERIFIED FACTS, and no qualifiers the data does
   not state ("free", "covered", "newly renovated", "spacious").
3. Do not generalize from reviews: you may paraphrase what guests said, but
   never turn one mention into a pattern ("guests always...", "a favourite").
4. If data is missing (no reviews, no policy), simply omit the topic.
5. Plain text only — no HTML, no markdown.
6. Warm, specific, concise. Avoid generic filler ("nestled", "hidden gem",
   "home away from home").
7. Each section has its own job — do not repeat the same selling point across
   headline, highlights, about and amenity descriptions.

Respond with a single JSON object, no code fences, with exactly these keys:
{
  "hero_headline": string,        // max 80 characters, no ending period
  "highlights": [string],         // 3 to 5 short selling points, each grounded
  "about_this_place": string,     // one paragraph, 60-140 words
  "amenity_descriptions": {       // key: amenity name EXACTLY as listed in
    "<amenity name>": string      // VERIFIED FACTS; value: one short sentence.
  }                               // Include only amenities from VERIFIED FACTS.
}"""


def strip_html(text: str) -> str:
    """Remove tags and unescape entities; collapse whitespace."""
    no_tags = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", html.unescape(no_tags)).strip()


def build_user_prompt(prop: PropertyData) -> str:
    lines: list[str] = ["VERIFIED FACTS"]
    lines.append(f"- Property name: {prop.property_name}")
    lines.append(f"- Property type: {prop.property_type}")
    lines.append(f"- Location: {prop.location.city}, {prop.location.country}")
    ri = prop.rental_info
    lines.append(
        f"- Capacity: sleeps {ri.max_guests}, {ri.bedrooms} bedroom(s), "
        f"{ri.bathrooms} bathroom(s)"
    )
    if prop.amenities:
        names = humanize_amenities(prop.amenities)
        lines.append("- Amenities: " + "; ".join(names))
    else:
        lines.append("- Amenities: none listed")
    if prop.num_of_reviews > 0 and prop.average_review_score is not None:
        lines.append(
            f"- Guest reviews: {prop.num_of_reviews} review(s), "
            f"average score {prop.average_review_score}"
        )
        for review in prop.reviews[:5]:
            lines.append(f'  - Guest said: "{strip_html(review)}"')
    else:
        lines.append("- Guest reviews: none yet")
    pol = prop.policies
    if pol.cancellation_policy:
        lines.append(f"- Cancellation policy: {pol.cancellation_policy}")
    if pol.payment_schedule:
        lines.append(f"- Payment schedule: {pol.payment_schedule}")
    if pol.damage_deposit:
        lines.append(f"- Damage deposit: {pol.damage_deposit}")
    hr = prop.house_rules
    if hr.check_in_time and hr.check_out_time:
        lines.append(f"- Check-in from {hr.check_in_time}, check-out by {hr.check_out_time}")

    lines.append("")
    lines.append("OWNER-PROVIDED TEXT (tone/character only — not a source of facts)")
    if prop.description.headline:
        lines.append(f"- Owner headline: {strip_html(prop.description.headline)}")
    if prop.description.description:
        lines.append(f"- Owner description: {strip_html(prop.description.description)}")
    if not prop.description.headline and not prop.description.description:
        lines.append("- (none provided)")

    lines.append("")
    lines.append("Write the listing content JSON for this property.")
    return "\n".join(lines)
