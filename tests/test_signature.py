"""
Unit tests for Signature class.
"""

import unittest

from feedback_prompt_learning.core.signature import (FieldType, Signature,
                                                     SignatureField)


class TestSignature(unittest.TestCase):
    """Test Signature functionality"""

    def setUp(self):
        """Set up test data"""
        self.input_field = SignatureField(
            name="question",
            description="A question to answer",
            field_type=FieldType.INPUT
        )
        self.output_field = SignatureField(
            name="answer",
            description="The answer",
            field_type=FieldType.OUTPUT
        )
        self.intermediate_field = SignatureField(
            name="reasoning",
            description="Step-by-step reasoning",
            field_type=FieldType.INTERMEDIATE,
            required=False
        )

    def test_signature_creation(self):
        """Test creating a basic signature"""
        sig = Signature(
            name="QA",
            inputs=[self.input_field],
            outputs=[self.output_field]
        )
        self.assertEqual(sig.name, "QA")
        self.assertEqual(len(sig.inputs), 1)
        self.assertEqual(len(sig.outputs), 1)
        self.assertIsNone(sig.instructions)

    def test_signature_with_instructions(self):
        """Test signature with instructions"""
        sig = Signature(
            name="QA_with_Reasoning",
            inputs=[self.input_field],
            outputs=[self.output_field, self.intermediate_field],
            instructions="Answer with reasoning."
        )
        self.assertEqual(sig.instructions, "Answer with reasoning.")

    def test_signature_str(self):
        """Test string representation"""
        sig = Signature(
            name="QA",
            inputs=[self.input_field],
            outputs=[self.output_field]
        )
        expected = "QA: question → answer"
        self.assertEqual(str(sig), expected)

    def test_signature_field_str(self):
        """Test SignatureField string representation"""
        self.assertEqual(str(self.input_field), "→ question: A question to answer")
        self.assertEqual(str(self.output_field), "← answer: The answer")
        self.assertEqual(str(self.intermediate_field), "○ reasoning: Step-by-step reasoning")

    def test_signature_field_defaults(self):
        """Test SignatureField default values"""
        field = SignatureField(
            name="test",
            description="test desc",
            field_type=FieldType.INPUT
        )
        self.assertEqual(field.dtype, str)
        self.assertTrue(field.required)
        self.assertIsNone(field.default)


if __name__ == '__main__':
    unittest.main()
