import asyncio
import sys

from mellea.backends import ModelOption
from mellea.backends.ollama import OllamaModelBackend
from mellea.core import CBlock
from mellea.stdlib.components.instruction import Instruction
from mellea.stdlib.context import SimpleContext

from mellea_partial.extras import LMStudioBackend


async def main():
    prompt = " ".join(sys.argv[1:]) or "Explain what a neural network is in 13 sentences."

    backend = LMStudioBackend("granite-4.0-micro@q8_0", model_options={ModelOption.STREAM: True})
    # backend = OllamaModelBackend(model_options={ModelOption.STREAM: True})
    instruction = CBlock(prompt)
    ctx = SimpleContext()

    thunk, ctx = await backend.generate_from_context(instruction, ctx)

    while not thunk.is_computed():
        text = await thunk.astream()
        print(f"\r{text}\n=========\n", end="", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
