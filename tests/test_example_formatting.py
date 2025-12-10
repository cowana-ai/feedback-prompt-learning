"""
Unit tests for example formatting in MCTS optimizer.
Ensures that the template-based formatting produces the same output as the original string concatenation.
"""

import unittest
from typing import Dict, List

from feedback_prompt_learning.search_algo.mcts import format_examples


class TestExampleFormatting(unittest.TestCase):
    """Test example formatting matches original implementation"""

    def setUp(self):
        """Set up test data"""
        self.sample_evaluations = [
            {
                'input': 'What is 2+2?',
                'output': '4',
                'target': '4',
                'score': 1.0,
                'feedback': 'Correct answer'
            },
            {
                'input': 'What is the capital of France?',
                'output': 'London',
                'target': 'Paris',
                'score': 0.0,
                'feedback': 'Incorrect city, this is the capital of UK'
            },
            {
                'input': 'Solve: 3 * 7',
                'output': '20',
                'target': '21',
                'score': 0.5,
                'feedback': 'Close but incorrect calculation'
            }
        ]

        self.total_evaluations = 10
        self.shown_examples = self.sample_evaluations

    def format_examples_original(self, examples: list[dict], total: int, include_feedback: bool) -> str:
        """Original implementation using string concatenation"""
        if include_feedback:
            example_string = f"Total evaluations: {total}\n"
            example_string += f"Showing {len(examples)} evaluated examples:\n"
            for i, item in enumerate(examples, 1):
                example_string += f"\nExample {i} (Score: {item['score']:.2f}):\n"
                example_string += f"Question: {item['input']}\n"
                example_string += f"LLM Output: {item['output']}\n"
                example_string += f"Expected: {item['target']}\n"
                example_string += f"Feedback: {item['feedback']}\n"
        else:
            example_string = f"Total evaluations: {total}\n"
            example_string += f"Showing {len(examples)} evaluated examples:\n"
            for i, item in enumerate(examples, 1):
                example_string += f"\nExample {i} (Score: {item['score']:.2f}):\n"
                example_string += f"Question: {item['input']}\n"
                example_string += f"LLM Output: {item['output']}\n"
                example_string += f"Expected: {item['target']}\n"

        return example_string

    def test_format_with_feedback(self):
        """Test that formatting with feedback matches original implementation"""
        original = self.format_examples_original(
            self.shown_examples,
            self.total_evaluations,
            include_feedback=True
        )
        new = format_examples(
            total=self.total_evaluations,
            examples=self.shown_examples,
            include_feedback=True
        )

        self.assertEqual(original, new, "Formatted output with feedback should match original")

    def test_format_without_feedback(self):
        """Test that formatting without feedback matches original implementation"""
        original = self.format_examples_original(
            self.shown_examples,
            self.total_evaluations,
            include_feedback=False
        )
        new = format_examples(
            total=self.total_evaluations,
            examples=self.shown_examples,
            include_feedback=False
        )

        self.assertEqual(original, new, "Formatted output without feedback should match original")

    def test_single_example_with_feedback(self):
        """Test formatting single example with feedback"""
        single_example = [self.sample_evaluations[0]]

        original = self.format_examples_original(single_example, 1, include_feedback=True)
        new = format_examples(total=1, examples=single_example, include_feedback=True)

        self.assertEqual(original, new, "Single example with feedback should match")

    def test_single_example_without_feedback(self):
        """Test formatting single example without feedback"""
        single_example = [self.sample_evaluations[0]]

        original = self.format_examples_original(single_example, 1, include_feedback=False)
        new = format_examples(total=1, examples=single_example, include_feedback=False)

        self.assertEqual(original, new, "Single example without feedback should match")

    def test_empty_examples(self):
        """Test formatting with empty examples list"""
        original = self.format_examples_original([], 0, include_feedback=True)
        new = format_examples(total=0, examples=[], include_feedback=True)

        self.assertEqual(original, new, "Empty examples should match")

    def test_score_formatting(self):
        """Test that scores are formatted with 2 decimal places"""
        examples_with_weird_scores = [
            {
                'input': 'Test',
                'output': 'Test',
                'target': 'Test',
                'score': 0.123456789,
                'feedback': 'Test feedback'
            }
        ]

        original = self.format_examples_original(examples_with_weird_scores, 1, include_feedback=True)
        new = format_examples(total=1, examples=examples_with_weird_scores, include_feedback=True)

        self.assertEqual(original, new)
        self.assertIn('0.12', original)  # Should be rounded to 2 decimals
        self.assertIn('0.12', new)

    def test_special_characters(self):
        """Test handling of special characters in input/output"""
        special_examples = [
            {
                'input': 'What is "quoted" text?',
                'output': 'It\'s text with quotes',
                'target': 'Text with "quotes"',
                'score': 0.75,
                'feedback': 'Missing some {special} characters'
            }
        ]

        original = self.format_examples_original(special_examples, 1, include_feedback=True)
        new = format_examples(total=1, examples=special_examples, include_feedback=True)

        self.assertEqual(original, new)

    def test_multiline_content(self):
        """Test handling of multiline input/output"""
        multiline_examples = [
            {
                'input': 'Line 1\nLine 2\nLine 3',
                'output': 'Output line 1\nOutput line 2',
                'target': 'Expected\nmultiline\noutput',
                'score': 0.33,
                'feedback': 'Multiline\nfeedback\nhere'
            }
        ]

        original = self.format_examples_original(multiline_examples, 1, include_feedback=True)
        new = format_examples(total=1, examples=multiline_examples, include_feedback=True)

        self.assertEqual(original, new)


class TestExampleFormattingIntegration(unittest.TestCase):
    """Integration tests for example formatting in actual usage scenarios"""

    def test_realistic_scenario(self):
        """Test with realistic data similar to actual MCTS usage"""
        realistic_examples = [
            {
                'input': 'Calculate the derivative of x^2 + 3x + 2',
                'output': '2x + 3',
                'target': '2x + 3',
                'score': 1.0,
                'feedback': 'Perfect! Correctly applied derivative rules.'
            },
            {
                'input': 'What is the integral of 2x?',
                'output': 'x^2 + C',
                'target': 'x^2 + C',
                'score': 1.0,
                'feedback': 'Correct integration with constant.'
            },
            {
                'input': 'Solve: sin(π/2)',
                'output': '0',
                'target': '1',
                'score': 0.0,
                'feedback': 'Incorrect value. sin(π/2) = 1, not 0.'
            }
        ]

        def format_original(examples, total, include_feedback):
            if include_feedback:
                s = f"Total evaluations: {total}\n"
                s += f"Showing {len(examples)} evaluated examples:\n"
                for i, item in enumerate(examples, 1):
                    s += f"\nExample {i} (Score: {item['score']:.2f}):\n"
                    s += f"Question: {item['input']}\n"
                    s += f"LLM Output: {item['output']}\n"
                    s += f"Expected: {item['target']}\n"
                    s += f"Feedback: {item['feedback']}\n"
            else:
                s = f"Total evaluations: {total}\n"
                s += f"Showing {len(examples)} evaluated examples:\n"
                for i, item in enumerate(examples, 1):
                    s += f"\nExample {i} (Score: {item['score']:.2f}):\n"
                    s += f"Question: {item['input']}\n"
                    s += f"LLM Output: {item['output']}\n"
                    s += f"Expected: {item['target']}\n"
            return s

        # Test both with and without feedback
        for include_feedback in [True, False]:
            original = format_original(realistic_examples, 50, include_feedback)
            new = format_examples(total=50, examples=realistic_examples, include_feedback=include_feedback)

            self.assertEqual(
                original,
                new,
                f"Realistic scenario (include_feedback={include_feedback}) should match"
            )


if __name__ == '__main__':
    unittest.main()
