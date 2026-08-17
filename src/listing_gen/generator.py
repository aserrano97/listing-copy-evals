"""Listing content generator with a pluggable LLM client.

`ListingGenerator` depends on the `LLMClient` protocol, not on a concrete
provider. In production/evals the client is `InspectLLMClient`, which routes
through Inspect AI's model API so every generation is captured in the .eval
log; in tests it is a stub. That one seam gives dependency injection,
mocking, and reproducible logs at the same time.
"""

from __future__ import annotations

import json
import re
from typing import Protocol

from pydantic import ValidationError

from .models import ListingContent, PropertyData
from .prompts import SYSTEM_PROMPT, build_user_prompt


class LLMClient(Protocol):
    async def complete(self, system: str, user: str) -> str:
        """Return the model's text completion for a system+user prompt."""
        ...


class InspectLLMClient:
    """LLMClient backed by Inspect AI's model API.

    With `model=None` inside a running eval, `get_model()` resolves to the
    eval's active model, so generations land in the eval transcript/log.
    """

    def __init__(self, model: object | None = None) -> None:
        self._model = model

    async def complete(self, system: str, user: str) -> str:
        from inspect_ai.model import ChatMessageSystem, ChatMessageUser, get_model

        model = get_model(self._model)
        result = await model.generate(
            [ChatMessageSystem(content=system), ChatMessageUser(content=user)]
        )
        return result.completion


class StaticLLMClient:
    """LLMClient that replays canned responses. Used in tests and offline demos."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls: list[tuple[str, str]] = []

    async def complete(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        if not self._responses:
            raise RuntimeError("StaticLLMClient ran out of canned responses")
        return self._responses.pop(0)


class GenerationError(Exception):
    """Model output could not be parsed into ListingContent after retries."""

    def __init__(self, message: str, raw_output: str) -> None:
        super().__init__(message)
        self.raw_output = raw_output


def extract_json(text: str) -> dict:
    """Parse the first JSON object in `text`, tolerating code fences."""
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object found in model output")
    return json.loads(cleaned[start : end + 1])


class ListingGenerator:
    def __init__(self, client: LLMClient, max_retries: int = 1) -> None:
        self._client = client
        self._max_retries = max_retries

    async def generate(self, prop: PropertyData) -> ListingContent:
        user_prompt = build_user_prompt(prop)
        last_error = ""
        raw = ""
        for attempt in range(self._max_retries + 1):
            prompt = user_prompt
            if attempt > 0:
                prompt += (
                    "\n\nYour previous response was not valid "
                    f"({last_error}). Respond again with ONLY the JSON object."
                )
            raw = await self._client.complete(SYSTEM_PROMPT, prompt)
            try:
                return ListingContent.model_validate(extract_json(raw))
            except (ValueError, ValidationError) as exc:
                last_error = str(exc).splitlines()[0]
        raise GenerationError(
            f"unparseable output after {self._max_retries + 1} attempt(s): {last_error}",
            raw_output=raw,
        )
