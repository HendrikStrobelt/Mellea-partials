# mellea-partial

Streaming LLM output with chunk-by-chunk validation using [Mellea](https://github.com/generative-computing/mellea).

Validates LLM responses _as they stream_ — splitting output into sentences, words, or paragraphs and running quick checks on each chunk. Fails fast on the first bad chunk instead of waiting for the full response.

## How it works

Two-phase validation:

1. **Quick checks (per-chunk):** As tokens stream in, output is split by a chunking pattern (sentence/word/paragraph). Each completed chunk is validated against quick-check requirements. Streaming stops immediately on failure.
2. **Full validation (post-stream):** Once streaming completes successfully, the full output is validated against the instruction's own requirements.

Core implementation is in `stream_with_chunking.py`.

## Setup

Requires Python 3.12+ and a local LLM server ([LM Studio](https://lmstudio.ai/) or [Ollama](https://ollama.com/)).

```bash
uv sync
```

Start your local model server, then:

```bash
# Basic streaming demo
uv run python stream_demo.py "Your prompt here"

# Chunked validation demo (happy path + fail-fast)
uv run python stream_chunking_demo.py
```

## Project structure

| File | Description |
|---|---|
| `stream_with_chunking.py` | Core streaming + chunking + validation logic |
| `stream_chunking_demo.py` | Demo: happy-path sentence validation and fail-fast on bad chunks |
| `stream_demo.py` | Minimal streaming example |
| `mellea_extra/` | `LMStudioBackend` — OpenAI-compatible backend for LM Studio |
| `fun/intrinsic_101.py` | RAG intrinsics example (context relevance checking) |

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
