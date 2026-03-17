# mellea-partial

Streaming LLM output with chunk-by-chunk validation using [Mellea](https://github.com/generative-computing/mellea).

Validates LLM responses _as they stream_ — splitting output into sentences, words, or paragraphs and running quick checks on each chunk. Fails fast on the first bad chunk instead of waiting for the full response.

## How it works

Two-phase validation:

1. **Quick checks (per-chunk):** As tokens stream in, output is split by a chunking pattern (sentence/word/paragraph). Each completed chunk is validated against quick-check requirements. Streaming stops immediately on failure.
2. **Full validation (post-stream):** Once streaming completes successfully, the full output is validated against the instruction's own requirements.

Core implementation is in `src/mellea_partial/chunking.py` and `src/mellea_partial/instruct.py`.

## Setup

Requires Python 3.12+ and a local LLM server ([LM Studio](https://lmstudio.ai/) or [Ollama](https://ollama.com/)).

```bash
uv sync
```

Start your local model server, then:

```bash
# Basic streaming demo
uv run python examples/stream_demo.py "Your prompt here"

# Chunked validation demo (happy path + fail-fast)
uv run python examples/stream_chunking_demo.py
```

## Usage

### `stream_with_chunking` — low-level streaming with per-chunk validation

```python
import asyncio
from mellea.core.requirement import Requirement
from mellea.stdlib.components.instruction import Instruction
from mellea.stdlib.requirements.requirement import simple_validate
from mellea_partial import ChunkingMode, stream_with_chunking
from mellea.stdlib.context import SimpleContext
from mellea_partial.extras import LMStudioBackend

async def main():
    backend = LMStudioBackend("granite-4.0-micro")
    instruction = Instruction(
        description="Explain neural networks in 4 sentences.",
    )
    quick_checks = [
        Requirement(
            "Each sentence must be under 200 characters.",
            simple_validate(lambda x: len(x.strip()) < 200),
            check_only=True,
        ),
    ]

    result = await stream_with_chunking(
        instruction, backend, SimpleContext(),
        quick_check_requirements=quick_checks,
        chunking=ChunkingMode.SENTENCE,
    )

    async for chunk in result.astream():
        print(chunk)          # validated chunks arrive here as they stream

    await result.acomplete()
    print("completed:", result.completed)

asyncio.run(main())
```

### `stream_instruct` — session-level streaming with retry strategies

```python
import asyncio
import re
from mellea.core.requirement import Requirement
from mellea.stdlib.requirements.requirement import simple_validate
from mellea.stdlib.sampling import RejectionSamplingStrategy
from mellea.stdlib.session import MelleaSession
import mellea_partial  # registers stream_instruct on MelleaSession
from mellea_partial import ChunkEvent, ChunkRepairedEvent, CompletedEvent
from mellea_partial.extras import LMStudioBackend

async def main():
    backend = LMStudioBackend("granite-4.0-micro")
    session = MelleaSession(backend)

    # repair: strip leading list numbers from chunks that contain them
    async def strip_numbering(chunk, ctx, qc_reqs, results):
        return (True, re.sub(r"^\s*\d+[\.\)]\s*", "", chunk))

    result = await session.stream_instruct(
        "Write a haiku about the ocean.",
        quick_check_requirements=[
            Requirement(
                "Chunk must not start with a number.",
                simple_validate(lambda x: not re.match(r"\s*\d", x)),
                check_only=True,
            ),
        ],
        quick_repair=strip_numbering,
        requirements=[
            Requirement(
                "Must be exactly 3 lines.",
                simple_validate(lambda x: len(x.strip().splitlines()) == 3),
                check_only=True,
            ),
        ],
        strategy=RejectionSamplingStrategy(loop_budget=3),
    )

    async for event in result.astream_events():
        if isinstance(event, ChunkRepairedEvent):
            print(f"[repaired] {event.original.strip()!r} → {event.repaired.strip()!r}")
        elif isinstance(event, ChunkEvent):
            print(event.text, end="", flush=True)
        elif isinstance(event, CompletedEvent):
            print(f"\nsuccess={event.success}, attempts={event.attempts_used}")

asyncio.run(main())
```

## Project structure

| Path | Description |
|---|---|
| `src/mellea_partial/chunking.py` | Core streaming + chunking + validation logic |
| `src/mellea_partial/instruct.py` | `stream_instruct()` powerup for `MelleaSession` |
| `src/mellea_partial/extras/` | `LMStudioBackend`, `FixedDocument` |
| `tests/` | Test suite (12 tests) |
| `examples/stream_chunking_demo.py` | Demo: happy-path sentence validation and fail-fast on bad chunks |
| `examples/stream_demo.py` | Minimal streaming example |
| `examples/stream_instruct_demo.py` | Demo: sampling strategies with `stream_instruct` |
| `examples/intrinsic_chunking_demo.py` | Demo: hallucination detection intrinsic as a streaming quick check |

## Example output

```
Demo 1: Happy path — sentence chunking with code-based quick checks

  Chunk 1: PASS | A neural network is a computational system modeled after the human brain...
  Chunk 2: PASS | It consists of interconnected nodes, or neurons, organized in layers.
  Chunk 3: PASS | Each neuron receives input, processes it, and passes the output...
  ...
  Status: Completed

Demo 2: Fail-fast — code-based check catches violation mid-stream

  Failed chunk: 1.
  Status: Stopped early
```
