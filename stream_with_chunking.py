"""Stream LLM output with chunk-by-chunk validation.

Two-phase validation:
1. Quick checks: validated per-chunk during streaming (stops on failure)
2. Full output checks: validated once after complete output (instruction requirements)
"""

import enum
import re
from dataclasses import dataclass, field

from mellea.core.backend import Backend
from mellea.core.base import CBlock, ModelOutputThunk
from mellea.core.requirement import Requirement, ValidationResult
from mellea.stdlib.components.instruction import Instruction
from mellea.stdlib.context import SimpleContext
from mellea.stdlib.functional import avalidate


class ChunkingMode(enum.Enum):
    SENTENCE = "sentence"
    WORD = "word"
    PARAGRAPH = "paragraph"


_SPLIT_PATTERNS: dict[ChunkingMode, re.Pattern[str]] = {
    ChunkingMode.SENTENCE: re.compile(r"(?<=[.!?])\s+"),
    ChunkingMode.WORD: re.compile(r"\s+"),
    ChunkingMode.PARAGRAPH: re.compile(r"\n\n+"),
}


@dataclass
class StreamChunkingResult:
    validated_chunks: list[str] = field(default_factory=list)
    failed_chunk: str | None = None
    quick_check_results: list[list[ValidationResult]] = field(default_factory=list)
    full_validation_results: list[ValidationResult] | None = None
    completed: bool = False
    full_text: str = ""


def _cancel_thunk(thunk: ModelOutputThunk) -> None:
    """Cancel a thunk's underlying generate task to avoid async generator warnings."""
    if thunk._generate is not None:
        thunk._generate.cancel()


async def stream_with_chunking(
    instruction: Instruction,
    backend: Backend,
    *,
    quick_check_requirements: list[Requirement | str] | None = None,
    chunking_mode: ChunkingMode = ChunkingMode.SENTENCE,
    model_options: dict | None = None,
) -> StreamChunkingResult:
    """Stream LLM output, validating chunks against quick-check requirements.

    Streams the response for the given instruction, splitting output into chunks
    (sentences, words, or paragraphs) and validating each completed chunk against
    quick_check_requirements. Stops immediately on failure. If streaming completes
    successfully, validates the full output against the instruction's own requirements.
    """
    ctx = SimpleContext()
    thunk, ctx = await backend.generate_from_context(
        instruction, ctx, model_options=model_options
    )

    # Normalize quick-check requirements
    qc_reqs: list[Requirement | str] = quick_check_requirements or []

    result = StreamChunkingResult()
    pattern = _SPLIT_PATTERNS[chunking_mode]
    validated_up_to = 0  # index into chunks we've already validated

    while not thunk.is_computed():
        delta = await thunk.astream()
        result.full_text += delta

        if not qc_reqs:
            continue

        parts = pattern.split(result.full_text)
        # Last element is the incomplete buffer (unless stream is done)
        completed_chunks = parts[:-1]

        # Validate any new completed chunks
        for i in range(validated_up_to, len(completed_chunks)):
            chunk = completed_chunks[i]
            if not chunk.strip():
                result.validated_chunks.append(chunk)
                result.quick_check_results.append([])
                continue

            validation_ctx = SimpleContext()
            chunk_results = await avalidate(
                qc_reqs, validation_ctx, backend, output=ModelOutputThunk(chunk)
            )
            result.quick_check_results.append(chunk_results)

            if not all(chunk_results):
                result.failed_chunk = chunk
                _cancel_thunk(thunk)
                return result

            result.validated_chunks.append(chunk)

        validated_up_to = len(completed_chunks)

    # Stream finished — get final text
    result.full_text = await thunk.avalue()

    # Validate remaining chunks from final text
    if qc_reqs:
        parts = pattern.split(result.full_text)
        for i in range(validated_up_to, len(parts)):
            chunk = parts[i]
            if not chunk.strip():
                result.validated_chunks.append(chunk)
                result.quick_check_results.append([])
                continue

            validation_ctx = SimpleContext()
            chunk_results = await avalidate(
                qc_reqs, validation_ctx, backend, output=ModelOutputThunk(chunk)
            )
            result.quick_check_results.append(chunk_results)

            if not all(chunk_results):
                result.failed_chunk = chunk
                _cancel_thunk(thunk)
                return result

            result.validated_chunks.append(chunk)

    result.completed = True

    # Phase 2: full-output validation against instruction requirements
    if instruction.requirements:
        full_ctx = SimpleContext()
        result.full_validation_results = await avalidate(
            instruction.requirements,
            full_ctx,
            backend,
            output=ModelOutputThunk(result.full_text),
        )

    return result
