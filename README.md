# listing-copy-evals

A property listing content generator built **evaluation-first**: the pipeline takes vacation
rental property data and produces structured, grounded marketing copy — and the primary
deliverable is the [Inspect AI](https://inspect.aisi.org.uk/) evaluation suite that measures
whether that copy is grounded, well-structured and useful.

## Quick start

```bash
uv sync
uv run jupyter lab evals.ipynb   # or: open evals.ipynb in your editor
```

**No API key is needed.** The notebook detects the absence of `ANTHROPIC_API_KEY` and replays
the committed `.eval` logs under `logs/` — real results from real runs, fully reproducible
offline. The committed notebook was itself executed in this replay mode.

To re-run the eval live instead:

```bash
cp .env.example .env             # put your Anthropic key in .env
uv run --env-file .env jupyter lab evals.ipynb
```

Tests (fully offline — stub clients and Inspect's `mockllm`, no network):

```bash
uv run pytest
```

## Approach

**Grounding contract.** The core design decision: structured platform fields (amenities,
rental_info, reviews, policies…) are *verified facts* — the only permitted source of factual
claims. The owner's free-text description is *tone only*: it may contain HTML and unverified
marketing ("beachfront paradise!"), so the generator is prompted — and the evals verify — that
no factual claim is sourced from it.

**Two-tier evaluation**, because different failures need different instruments:

| Scorer | Tier | Checks |
|---|---|---|
| `structure_valid` | deterministic | parses to schema, headline length, 3–5 highlights, word counts, no HTML |
| `numeric_grounding` | deterministic | every number in the copy exists in the input data (rating-scale mentions excluded) |
| `amenity_grounding` | deterministic | every amenity described exists in the input amenity list |
| `forbidden_claims` | deterministic | adversarial fixtures: planted owner-text overclaims must not leak into the copy |
| `claim_groundedness` | LLM judge | extracts each factual claim, verdicts it against the source JSON; unsupported claims are named in the explanation |
| `copy_quality` | LLM judge | 1–5 rubric: specificity, tone, clarity, cliché avoidance |

All scorers emit values in [0, 1] plus a human-readable explanation — the explanations are the
debugging surface that drove every iteration (see notebook §7).

**Fixtures** (`fixtures/properties/`) are designed to cover the failure space: rich happy path,
minimal data (does it invent filler?), HTML-heavy input, adversarial owner overclaims, mixed
reviews, unknown amenity codes, non-English property. Eval-side expectations live separately in
`fixtures/expectations.json` so inputs stay spec-realistic.

**Engineering.** `ListingGenerator` depends on an `LLMClient` protocol. In evals it's
`InspectLLMClient` (Inspect's model API — so every generation lands in the `.eval` log); in
tests it's a stub. One seam gives dependency injection, mocking and reproducible logs at once.
Judge scorers take an injectable model, tested offline with `mockllm` and canned outputs.

## Reading the eval logs

Committed logs are the evidence trail of the eval-driven loop (v1 smoke → prompt fix →
v2 → scorer fix → v3 final; the story with numbers is in notebook §7):

```
logs/smoke/…WZAi7….eval   # v1 baseline (2 samples): groundedness 0.78, owner-text leaks
logs/…UHYAfL5….eval       # v2 full run: groundedness 0.96, numeric-scorer false positive found
logs/…kNwBdn….eval        # v3 final: deterministic scorers 1.00, groundedness 0.96, quality 0.85
```

Browse them interactively:

```bash
uv run inspect view --log-dir logs/
```

Each sample shows the full generation transcript, every scorer's value and explanation
(e.g. exactly which claims the judge found unsupported), and token usage. Programmatic access:
`inspect_ai.log.read_eval_log(path)` — this is what the notebook does in replay mode.

## Project layout

```
evals.ipynb                  # the deliverable: pipeline + eval + analysis
src/listing_gen/
  models.py                  # PropertyData (input), ListingContent (output)
  amenities.py               # amenity code → guest-facing name (+ fallback)
  prompts.py                 # grounding-contract prompt construction
  generator.py               # LLMClient protocol, Inspect/stub clients, ListingGenerator
  evals/                     # dataset, solver, scorers, task
fixtures/                    # property fixtures + eval expectations
tests/                       # 32 offline tests
logs/                        # committed inspect-ai run logs
```

## How I used AI

Built pair-programming with Claude Code (Claude Fable 5): I set the architecture, the grounding
contract, the scorer design and the fixture failure-space, and reviewed/steered every file; the
assistant wrote the bulk of the code and ran the eval iterations under my direction. The
iteration decisions in notebook §7 (what each eval round meant and what to change) were the
human-in-the-loop part — which is rather the point of the exercise: the evals made those
decisions cheap.

## What I'd do next

See notebook §8: judge calibration against a human-labeled claim set, regression thresholds in
CI, a property-type × language × sparsity fixture grid, image inputs as multimodal grounding,
and cost/latency as first-class eval metrics.
