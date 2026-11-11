"""Interactive demo for the LLM project (Assignment 1B).

Run from the project root::

    python demo.py

The first run downloads the model weights into the HuggingFace cache, so it is
slowest the very first time. Type ``quit`` to leave the loop.
"""

import argparse

from src.model import get_response, load_model


def main():
    parser = argparse.ArgumentParser(
        description="Chat with a quantized open-weight LLM."
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=150,
        help="Maximum tokens per reply (default: %(default)s).",
    )
    args = parser.parse_args()

    print("Loading model (first run downloads the weights) ...")
    model, tokenizer = load_model()
    print(f"Model ready: {model.name_or_path}")
    print("Ask anything, or type 'quit' to exit.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"quit", "exit", "q"}:
            break
        if not user_input:
            continue
        response = get_response(
            user_input, model, tokenizer, max_length=args.max_new_tokens
        )
        print(f"Model: {response}\n")


if __name__ == "__main__":
    main()
