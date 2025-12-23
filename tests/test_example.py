"""
Unit tests for Example class.
"""

import unittest

from feedback_prompt_learning.core.example import Example
from feedback_prompt_learning.core.signature import (FieldType, Signature,
                                                     SignatureField)


class TestExample(unittest.TestCase):
    """Test Example functionality"""

    def test_explicit_creation(self):
        """Test creating example with explicit inputs/outputs"""
        example = Example(
            inputs={"question": "What is 2+2?"},
            outputs={"answer": "4"}
        )
        self.assertEqual(example.inputs["question"], "What is 2+2?")
        self.assertEqual(example.outputs["answer"], "4")
        self.assertEqual(example.pending_fields, {})

    def test_builder_pattern_simple(self):
        """Test builder pattern for simple QA"""
        example = Example(
            question="What is 2+2?",
            answer="4"
        ).with_inputs("question")
        self.assertEqual(example.inputs["question"], "What is 2+2?")
        self.assertEqual(example.outputs["answer"], "4")

    def test_builder_pattern_multiple_inputs(self):
        """Test builder pattern with multiple inputs"""
        example = Example(
            text="Hello world",
            context="greeting",
            label="positive"
        ).with_inputs("text", "context")
        self.assertEqual(example.inputs["text"], "Hello world")
        self.assertEqual(example.inputs["context"], "greeting")
        self.assertEqual(example.outputs["label"], "positive")

    def test_with_outputs(self):
        """Test with_outputs method"""
        example = Example(
            question="What is 2+2?",
            reasoning="2 + 2 = 4",
            answer="4"
        ).with_inputs("question").with_outputs("reasoning", "answer")
        self.assertEqual(example.inputs["question"], "What is 2+2?")
        self.assertEqual(example.outputs["reasoning"], "2 + 2 = 4")
        self.assertEqual(example.outputs["answer"], "4")

    def test_with_feedback(self):
        """Test adding feedback"""
        example = Example(
            question="What is 2+2?",
            answer="4"
        ).with_inputs("question").with_feedback(
            accuracy_feedback="Correct",
            reasoning_feedback="Simple"
        )
        self.assertEqual(example.feedback["accuracy"], "Correct")
        self.assertEqual(example.feedback["reasoning"], "Simple")

    def test_with_metadata(self):
        """Test adding metadata"""
        example = Example(
            question="What is 2+2?",
            answer="4"
        ).with_metadata(source="test", difficulty="easy")
        self.assertEqual(example.metadata["source"], "test")
        self.assertEqual(example.metadata["difficulty"], "easy")

    def test_getitem(self):
        """Test dict-like access"""
        example = Example(
            question="What is 2+2?",
            answer="4"
        ).with_inputs("question")
        self.assertEqual(example["question"], "What is 2+2?")
        self.assertEqual(example["answer"], "4")
        with self.assertRaises(KeyError):
            _ = example["nonexistent"]

    def test_repr(self):
        """Test string representation"""
        example = Example(
            question="What is 2+2?",
            answer="4"
        ).with_inputs("question")
        repr_str = repr(example)
        self.assertIn("inputs=['question']", repr_str)
        self.assertIn("outputs=['answer']", repr_str)

    def test_validate_against_signature(self):
        """Test validation against signature"""
        sig = Signature(
            name="QA",
            inputs=[SignatureField("question", "A question", FieldType.INPUT)],
            outputs=[SignatureField("answer", "The answer", FieldType.OUTPUT)]
        )
        example = Example(
            question="What is 2+2?",
            answer="4"
        ).with_inputs("question")
        self.assertTrue(example.validate_against_signature(sig))

        # Missing input
        bad_example = Example(outputs={"answer": "4"})
        self.assertFalse(bad_example.validate_against_signature(sig))

    def test_get_missing_fields(self):
        """Test getting missing fields"""
        sig = Signature(
            name="QA",
            inputs=[SignatureField("question", "A question", FieldType.INPUT)],
            outputs=[SignatureField("answer", "The answer", FieldType.OUTPUT)]
        )
        bad_example = Example(outputs={"answer": "4"})
        missing = bad_example.get_missing_fields(sig)
        self.assertIn("question", missing["inputs"])
        self.assertEqual(missing["outputs"], [])

    def test_from_dict_auto(self):
        """Test from_dict with auto inference"""
        data = {"question": "What is 2+2?", "answer": "4", "source": "test"}
        example = Example.from_dict(data)
        self.assertEqual(example.inputs["question"], "What is 2+2?")
        self.assertEqual(example.outputs["answer"], "4")
        self.assertEqual(example.metadata["source"], "test")

    def test_from_dict_explicit_inputs(self):
        """Test from_dict with explicit input keys"""
        data = {"text": "Hello", "context": "greeting", "label": "positive"}
        example = Example.from_dict(data, input_keys=["text", "context"])
        self.assertEqual(example.inputs["text"], "Hello")
        self.assertEqual(example.inputs["context"], "greeting")
        self.assertEqual(example.outputs["label"], "positive")


if __name__ == '__main__':
    unittest.main()
