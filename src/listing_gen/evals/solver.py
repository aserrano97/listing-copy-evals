"""Inspect solver that runs the listing generation pipeline on a sample."""

from __future__ import annotations

from inspect_ai.model import ModelOutput
from inspect_ai.solver import Generate, Solver, TaskState, solver

from ..generator import GenerationError, InspectLLMClient, ListingGenerator
from ..models import PropertyData


@solver
def listing_solver(max_retries: int = 1) -> Solver:
    """Generate listing content from the sample's property metadata.

    Uses InspectLLMClient with no explicit model, so generation runs on the
    eval's active model and is captured in the .eval log. A generation that
    stays unparseable after retries is stored raw instead of raised, so it
    shows up as a scored failure rather than a sample error.
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        prop = PropertyData.model_validate(state.metadata["property"])
        gen = ListingGenerator(InspectLLMClient(), max_retries=max_retries)
        try:
            content = await gen.generate(prop)
            completion = content.model_dump_json(indent=2)
        except GenerationError as exc:
            completion = exc.raw_output
        state.output = ModelOutput.from_content(
            model="listing_generator", content=completion
        )
        return state

    return solve
