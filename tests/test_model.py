"""Unit tests for the LLM project (Assignment 1B).

Run from the project root with either command::

    python -m unittest discover tests
    python -m pytest -v

The model is loaded once for the whole suite (``setUpClass``) so the tests
only pay the download/load cost a single time.
"""

import unittest

from src.model import get_response, load_model


class TestModel(unittest.TestCase):
    """Basic end-to-end checks for the quantized chat model."""

    @classmethod
    def setUpClass(cls):
        """Load the model once for all tests."""
        cls.model, cls.tokenizer = load_model()

    def test_factual_query(self):
        """A simple factual question should name the correct capital."""
        response = get_response(
            "What is the capital of France?",
            self.model,
            self.tokenizer,
            max_length=50,
        )
        self.assertIn("Paris", response)

    def test_code_generation(self):
        """A coding request should produce function-shaped output."""
        response = get_response(
            "Write a Python function to calculate the factorial of a number.",
            self.model,
            self.tokenizer,
            max_length=200,
        )
        self.assertIn("def", response)
        self.assertIn("factorial", response)
        self.assertTrue("for" in response or "range" in response)

    def test_reasoning(self):
        """A one-step arithmetic puzzle should yield the right number."""
        response = get_response(
            "If I have 3 apples and give 1 to Mary, how many do I have left?",
            self.model,
            self.tokenizer,
            max_length=50,
        )
        # Accept either the digit or the written word.
        self.assertTrue("2" in response or "two" in response)

    def test_response_is_text(self):
        """The reply must be a non-empty string, never a special token."""
        response = get_response(
            "Say hello.", self.model, self.tokenizer, max_length=30
        )
        self.assertIsInstance(response, str)
        self.assertGreater(len(response.strip()), 0)


if __name__ == "__main__":
    unittest.main()
