"""The Inspect task tying dataset, solver and scorers together."""

from __future__ import annotations

from inspect_ai import Task, task

from .dataset import load_dataset
from .scorers import (
    amenity_grounding,
    claim_groundedness,
    copy_quality,
    forbidden_claims,
    numeric_grounding,
    structure_valid,
)
from .solver import listing_solver


@task
def listing_eval() -> Task:
    return Task(
        dataset=load_dataset(),
        solver=listing_solver(),
        scorer=[
            structure_valid(),
            numeric_grounding(),
            amenity_grounding(),
            forbidden_claims(),
            claim_groundedness(),
            copy_quality(),
        ],
    )
