from collections import defaultdict

import pytest

from feedback_prompt_learning.core.example import Example
from feedback_prompt_learning.core.example_set import ExampleSet
from feedback_prompt_learning.core.signature import (FieldType, Signature,
                                                     SignatureField)


class TestExampleSet:
    """Unit tests for ExampleSet class."""

    @pytest.fixture
    def sample_examples(self):
        """Create sample examples for testing."""
        return [
            Example(question="What is 2+2?", answer="4").with_inputs("question"),
            Example(question="What is 3+3?", answer="6").with_inputs("question"),
            Example(question="What is 4+4?", answer="8").with_inputs("question"),
            Example(question="What is 5+5?", answer="10").with_inputs("question"),
        ]

    @pytest.fixture
    def sample_signature(self):
        """Create a sample signature for QA tasks."""
        return Signature(
            name="QA_Signature",
            inputs=[
                SignatureField("question", "A question to answer", FieldType.INPUT)
            ],
            outputs=[
                SignatureField("answer", "The answer", FieldType.OUTPUT)
            ]
        )

    @pytest.fixture
    def example_set(self, sample_examples):
        """Create a basic ExampleSet."""
        return ExampleSet(examples=sample_examples, name="test_dataset")

    @pytest.fixture
    def example_set_with_signature(self, sample_examples, sample_signature):
        """Create ExampleSet with signature."""
        return ExampleSet(examples=sample_examples, signature=sample_signature, name="test_dataset")

    def test_initialization(self, sample_examples):
        """Test basic initialization."""
        dataset = ExampleSet(examples=sample_examples, name="test")
        assert len(dataset) == 4
        assert dataset.name == "test"
        assert dataset.signature is None

    def test_initialization_with_signature(self, sample_examples, sample_signature):
        """Test initialization with signature validation."""
        dataset = ExampleSet(examples=sample_examples, signature=sample_signature)
        assert dataset.signature == sample_signature

    def test_initialization_invalid_signature(self, sample_examples):
        """Test initialization with invalid signature raises error."""
        invalid_signature = Signature(
            name="Invalid",
            inputs=[SignatureField("missing_field", "Missing", FieldType.INPUT)],
            outputs=[]
        )
        with pytest.raises(ValueError):
            ExampleSet(examples=sample_examples, signature=invalid_signature)

    def test_len(self, example_set):
        """Test __len__ method."""
        assert len(example_set) == 4

    def test_getitem(self, example_set):
        """Test __getitem__ method."""
        assert isinstance(example_set[0], Example)
        assert example_set[0]['question'] == "What is 2+2?"

        # Test slicing
        subset = example_set[1:3]
        assert isinstance(subset, ExampleSet)
        assert len(subset) == 2

    def test_iter(self, example_set):
        """Test iteration."""
        questions = [ex['question'] for ex in example_set]
        assert questions == ["What is 2+2?", "What is 3+3?", "What is 4+4?", "What is 5+5?"]

    def test_repr(self, example_set):
        """Test string representation."""
        assert "ExampleSet(name='test_dataset', n=4)" in repr(example_set)

    def test_validate_all_no_signature(self, example_set):
        """Test validate_all without signature raises error."""
        with pytest.raises(ValueError):
            example_set.validate_all()

    def test_validate_all_valid(self, example_set_with_signature):
        """Test validate_all with valid examples."""
        is_valid, invalid = example_set_with_signature.validate_all()
        assert is_valid
        assert invalid == []

    def test_add_signature(self, example_set, sample_signature):
        """Test adding signature."""
        example_set.add_signature(sample_signature)
        assert example_set.signature == sample_signature

    def test_get_valid_examples(self, example_set_with_signature):
        """Test getting valid examples."""
        valid = example_set_with_signature.get_valid_examples()
        assert len(valid) == 4  # All are valid

    def test_infer_signature(self, example_set):
        """Test signature inference."""
        signature = example_set.infer_signature(name="Inferred")
        assert signature.name == "Inferred"
        assert len(signature.inputs) == 1  # question
        assert len(signature.outputs) == 1  # answer

    def test_infer_signature_empty_dataset(self):
        """Test signature inference on empty dataset."""
        empty_dataset = ExampleSet(examples=[])
        with pytest.raises(ValueError):
            empty_dataset.infer_signature()

    def test_sample_random(self, example_set):
        """Test random sampling."""
        sample = example_set.sample(n=2, strategy="random", seed=42)
        assert len(sample) == 2
        assert isinstance(sample, ExampleSet)

    def test_sample_first(self, example_set):
        """Test first-n sampling."""
        sample = example_set.sample(n=2, strategy="first")
        assert len(sample) == 2
        assert sample[0]['question'] == "What is 2+2?"
        assert sample[1]['question'] == "What is 3+3?"

    def test_sample_last(self, example_set):
        """Test last-n sampling."""
        sample = example_set.sample(n=2, strategy="last")
        assert len(sample) == 2
        assert sample[0]['question'] == "What is 4+4?"
        assert sample[1]['question'] == "What is 5+5?"

    def test_sample_stratified(self, example_set):
        """Test stratified sampling."""
        # Add variety to answers for stratification
        examples = [
            Example(question="Q1", answer="A").with_inputs("question"),
            Example(question="Q2", answer="A").with_inputs("question"),
            Example(question="Q3", answer="B").with_inputs("question"),
            Example(question="Q4", answer="B").with_inputs("question"),
        ]
        dataset = ExampleSet(examples=examples)
        sample = dataset.sample(n=2, strategy="stratified", stratify_by="answer", seed=42)
        assert len(sample) == 2

    def test_split(self, example_set):
        """Test dataset splitting."""
        train, eval, test = example_set.split(train=0.5, eval=0.25, test=0.25, seed=42)
        assert len(train) == 2
        assert len(eval) == 1
        assert len(test) == 1

    def test_split_invalid_proportions(self, example_set):
        """Test split with invalid proportions."""
        with pytest.raises(ValueError):
            example_set.split(train=0.5, eval=0.5, test=0.5)

    def test_filter(self, example_set):
        """Test filtering with predicate."""
        filtered = example_set.filter(lambda ex: "2" in ex['question'])
        assert len(filtered) == 1
        assert filtered[0]['question'] == "What is 2+2?"

    def test_filter_by_feedback(self, sample_examples):
        """Test filtering by feedback."""
        examples_with_feedback = [
            ex.with_feedback(accuracy_feedback="Correct ✓") for ex in sample_examples[:2]
        ] + [
            ex.with_feedback(accuracy_feedback="Wrong ✗") for ex in sample_examples[2:]
        ]
        dataset = ExampleSet(examples=examples_with_feedback)
        correct = dataset.filter_by_feedback("accuracy", contains="Correct")
        assert len(correct) == 2

    def test_filter_by_field(self, example_set):
        """Test filtering by field values."""
        filtered = example_set.filter_by_field("answer", ["4", "6"])
        assert len(filtered) == 2

    def test_get_field_distribution(self, example_set):
        """Test field distribution."""
        dist = example_set.get_field_distribution("answer")
        assert dist["4"] == 1
        assert dist["6"] == 1
        assert dist["8"] == 1
        assert dist["10"] == 1

    def test_get_statistics(self, example_set_with_signature):
        """Test getting statistics."""
        stats = example_set_with_signature.get_statistics()
        assert stats["total_examples"] == 4
        assert stats["has_signature"] is True
        assert stats["signature_name"] == "QA_Signature"

    def test_to_list_and_from_list(self, example_set):
        """Test conversion to/from list."""
        data_list = example_set.to_list()
        assert len(data_list) == 4
        assert "inputs" in data_list[0]
        assert "outputs" in data_list[0]

        # Reconstruct
        reconstructed = ExampleSet.from_list(data_list, name="reconstructed")
        assert len(reconstructed) == 4
        assert reconstructed[0]['question'] == "What is 2+2?"
