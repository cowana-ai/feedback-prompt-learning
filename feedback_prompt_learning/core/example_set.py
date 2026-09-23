from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Callable
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field

from feedback_prompt_learning.core.example import Example
from feedback_prompt_learning.core.signature import (FieldType, Signature,
                                                     SignatureField)


class ExampleSet(BaseModel):
    """
    Collection of examples with dataset-level operations.

    Features:
    - Optional signature validation
    - Sampling strategies (random, stratified, balanced)
    - Filtering by feedback, performance, fields
    - Train/eval/test splitting
    - Signature inference and adaptation
    - Statistics and analysis

    Examples:
        # Basic creation
        dataset = ExampleSet(examples=[
            Example(question="Q1", answer="A1").with_inputs("question"),
            Example(question="Q2", answer="A2").with_inputs("question")
        ])

        # With signature validation
        dataset = ExampleSet(examples=examples, signature=signature)

        # Sampling
        train_sample = dataset.sample(n=10, strategy="random")

        # Filtering
        correct_examples = dataset.filter_by_feedback("accuracy", contains="Correct")
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    examples: list[Example] = Field(description="List of examples")
    signature: Signature | None = Field(None, description="Optional signature for validation")
    name: str = Field("dataset", description="Dataset name")
    description: str | None = Field(None, description="Dataset description")

    def __init__(self, examples: list[Example] = None, **data):
        """Initialize and optionally validate against signature."""
        if examples is not None:
            data['examples'] = examples

        super().__init__(**data)

        # Validate on creation if signature provided
        if self.signature and self.examples:
            self.validate_all(raise_on_error=True)

    def __len__(self) -> int:
        """Return number of examples."""
        return len(self.examples)

    def __getitem__(self, idx: int | slice) -> Example | ExampleSet:
        """Support indexing and slicing."""
        if isinstance(idx, slice):
            return ExampleSet(
                examples=self.examples[idx],
                signature=self.signature,
                name=f"{self.name}_slice"
            )
        return self.examples[idx]

    def __iter__(self):
        """Iterate over examples."""
        return iter(self.examples)

    def __repr__(self) -> str:
        """String representation."""
        sig_str = f", signature={self.signature.name}" if self.signature else ""
        return f"ExampleSet(name='{self.name}', n={len(self)}{sig_str})"

    def validate_all(self, raise_on_error: bool = True) -> tuple[bool, list[tuple[int, dict]]]:
        """
        Validate all examples against signature.

        Args:
            raise_on_error: If True, raise ValueError on validation failure

        Returns:
            (is_valid, invalid_examples) where invalid_examples is list of (index, missing_fields)

        Raises:
            ValueError: If raise_on_error=True and validation fails
        """
        if not self.signature:
            raise ValueError("Cannot validate without signature. Use add_signature() first.")

        invalid_examples = []
        for i, example in enumerate(self.examples):
            if not example.validate_against_signature(self.signature):
                missing = example.get_missing_fields(self.signature)
                invalid_examples.append((i, missing))

        if invalid_examples and raise_on_error:
            error_msg = f"Found {len(invalid_examples)} invalid examples:\n"
            for idx, missing in invalid_examples[:5]:  # Show first 5
                error_msg += f"  Example {idx}: missing {missing}\n"
            if len(invalid_examples) > 5:
                error_msg += f"  ... and {len(invalid_examples) - 5} more\n"
            raise ValueError(error_msg)

        is_valid = len(invalid_examples) == 0
        return is_valid, invalid_examples

    def add_signature(self, signature: Signature, validate: bool = True) -> None:
        """
        Add signature to dataset and optionally validate.

        Args:
            signature: The signature to add
            validate: If True, validate all examples

        Raises:
            ValueError: If validate=True and examples don't match signature
        """
        self.signature = signature
        if validate and self.examples:
            self.validate_all(raise_on_error=True)

    def get_valid_examples(self) -> ExampleSet:
        """Return subset of examples that match signature."""
        if not self.signature:
            return self

        valid = [
            ex for ex in self.examples
            if ex.validate_against_signature(self.signature)
        ]
        return ExampleSet(
            examples=valid,
            signature=self.signature,
            name=f"{self.name}_valid"
        )

    def infer_signature(
        self,
        name: str = "InferredSignature",
        instructions: str | None = None
    ) -> Signature:
        """
        Infer signature from examples by analyzing field names.

        Args:
            name: Name for the inferred signature
            instructions: Optional task instructions

        Returns:
            Inferred Signature

        Raises:
            ValueError: If dataset is empty

        Example:
            dataset = ExampleSet(examples=[...])
            signature = dataset.infer_signature()
            dataset.add_signature(signature)
        """
        if not self.examples:
            raise ValueError("Cannot infer signature from empty dataset")

        # Collect all field names
        all_input_fields = set()
        all_output_fields = set()

        for ex in self.examples:
            all_input_fields.update(ex.inputs.keys())
            all_output_fields.update(ex.outputs.keys())

        # Create signature fields
        inputs = [
            SignatureField(
                name=name,
                description=f"Input field: {name}",
                field_type=FieldType.INPUT
            )
            for name in sorted(all_input_fields)
        ]
        outputs = [
            SignatureField(
                name=name,
                description=f"Output field: {name}",
                field_type=FieldType.OUTPUT
            )
            for name in sorted(all_output_fields)
        ]

        return Signature(
            name=name,
            inputs=inputs,
            outputs=outputs,
            instructions=instructions
        )

    def sample(
        self,
        n: int,
        strategy: Literal["random", "first", "last", "stratified"] = "random",
        stratify_by: str | None = None,
        seed: int | None = None
    ) -> ExampleSet:
        """
        Sample n examples using specified strategy.

        Args:
            n: Number of examples to sample
            strategy: Sampling strategy
                - "random": Random sampling
                - "first": First n examples
                - "last": Last n examples
                - "stratified": Stratified sampling (requires stratify_by)
            stratify_by: Field name to stratify by (for strategy="stratified")
            seed: Random seed for reproducibility

        Returns:
            New ExampleSet with sampled examples
        """
        if seed is not None:
            random.seed(seed)

        n = min(n, len(self.examples))

        if strategy == "random":
            sampled = random.sample(self.examples, n)

        elif strategy == "first":
            sampled = self.examples[:n]

        elif strategy == "last":
            sampled = self.examples[-n:]

        elif strategy == "stratified":
            if not stratify_by:
                raise ValueError("stratify_by required for stratified sampling")
            sampled = self._stratified_sample(n, stratify_by)

        else:
            raise ValueError(f"Unknown sampling strategy: {strategy}")

        return ExampleSet(
            examples=sampled,
            signature=self.signature,
            name=f"{self.name}_sample_{n}"
        )

    def _stratified_sample(self, n: int, field_name: str) -> list[Example]:
        """Stratified sampling by field value."""
        # Group examples by field value
        groups = defaultdict(list)
        for ex in self.examples:
            value = ex.outputs.get(field_name, ex.inputs.get(field_name, "unknown"))
            groups[str(value)].append(ex)

        # Calculate samples per group
        samples_per_group = n // len(groups)
        remainder = n % len(groups)

        sampled = []
        for i, (value, group_examples) in enumerate(sorted(groups.items())):
            # Distribute remainder across first groups
            group_n = samples_per_group + (1 if i < remainder else 0)
            group_n = min(group_n, len(group_examples))
            sampled.extend(random.sample(group_examples, group_n))

        # If we didn't get enough, sample more from largest groups
        if len(sampled) < n:
            remaining = n - len(sampled)
            largest_groups = sorted(groups.items(), key=lambda x: len(x[1]), reverse=True)
            for value, group_examples in largest_groups:
                available = [ex for ex in group_examples if ex not in sampled]
                if available:
                    additional = min(remaining, len(available))
                    sampled.extend(random.sample(available, additional))
                    remaining -= additional
                    if remaining == 0:
                        break

        return sampled[:n]

    def split(
        self,
        train: float = 0.6,
        eval: float = 0.2,
        test: float = 0.2,
        seed: int | None = None,
        stratify_by: str | None = None
    ) -> tuple[ExampleSet, ExampleSet, ExampleSet]:
        """
        Split dataset into train/eval/test sets.

        Args:
            train: Proportion for training set
            eval: Proportion for evaluation set
            test: Proportion for test set
            seed: Random seed for reproducibility
            stratify_by: Optional field to stratify split by

        Returns:
            (train_set, eval_set, test_set)

        Example:
            train, eval, test = dataset.split(train=0.7, eval=0.15, test=0.15)
        """
        if not abs(train + eval + test - 1.0) < 0.01:
            raise ValueError(f"Splits must sum to 1.0, got {train + eval + test}")

        if seed is not None:
            random.seed(seed)

        n = len(self.examples)
        train_n = int(n * train)
        eval_n = int(n * eval)
        # test_n is the remainder

        if stratify_by:
            # Stratified split
            groups = defaultdict(list)
            for ex in self.examples:
                value = ex.outputs.get(stratify_by, ex.inputs.get(stratify_by, "unknown"))
                groups[str(value)].append(ex)

            train_examples, eval_examples, test_examples = [], [], []

            for group_examples in groups.values():
                random.shuffle(group_examples)
                group_n = len(group_examples)
                group_train_n = int(group_n * train)
                group_eval_n = int(group_n * eval)

                train_examples.extend(group_examples[:group_train_n])
                eval_examples.extend(group_examples[group_train_n:group_train_n + group_eval_n])
                test_examples.extend(group_examples[group_train_n + group_eval_n:])
        else:
            # Random split
            shuffled = self.examples.copy()
            random.shuffle(shuffled)

            train_examples = shuffled[:train_n]
            eval_examples = shuffled[train_n:train_n + eval_n]
            test_examples = shuffled[train_n + eval_n:]

        return (
            ExampleSet(examples=train_examples, signature=self.signature, name=f"{self.name}_train"),
            ExampleSet(examples=eval_examples, signature=self.signature, name=f"{self.name}_eval"),
            ExampleSet(examples=test_examples, signature=self.signature, name=f"{self.name}_test")
        )

    # ========================================================================
    # FILTERING
    # ========================================================================

    def filter(self, predicate: Callable[[Example], bool]) -> ExampleSet:
        """
        Filter examples using a predicate function.

        Args:
            predicate: Function that takes Example and returns bool

        Returns:
            New ExampleSet with filtered examples

        Example:
            # Filter examples with long questions
            long_questions = dataset.filter(lambda ex: len(ex['question']) > 100)
        """
        filtered = [ex for ex in self.examples if predicate(ex)]
        return ExampleSet(
            examples=filtered,
            signature=self.signature,
            name=f"{self.name}_filtered"
        )

    def filter_by_feedback(
        self,
        feedback_type: str,
        contains: str | None = None,
        equals: str | None = None
    ) -> ExampleSet:
        """
        Filter examples based on feedback content.

        Args:
            feedback_type: Type of feedback to check (e.g., "accuracy", "reasoning")
            contains: String that must be contained in feedback
            equals: String that feedback must equal exactly

        Returns:
            New ExampleSet with filtered examples

        Example:
            correct = dataset.filter_by_feedback("accuracy", contains="Correct")
            wrong = dataset.filter_by_feedback("accuracy", contains="Wrong")
        """
        def predicate(ex: Example) -> bool:
            if not ex.feedback or feedback_type not in ex.feedback:
                return False

            feedback_value = ex.feedback[feedback_type]

            if contains and contains not in feedback_value:
                return False
            if equals and feedback_value != equals:
                return False

            return True

        return self.filter(predicate)

    def filter_by_field(self, field_name: str, values: list[Any]) -> ExampleSet:
        """
        Filter examples where field has one of specified values.

        Args:
            field_name: Name of field to check (searches inputs then outputs)
            values: List of acceptable values

        Returns:
            New ExampleSet with filtered examples
        """
        def predicate(ex: Example) -> bool:
            value = ex.inputs.get(field_name, ex.outputs.get(field_name))
            return value in values

        return self.filter(predicate)

    def get_field_distribution(self, field_name: str) -> dict[Any, int]:
        """
        Get distribution of values for a field.

        Args:
            field_name: Field to analyze (searches inputs then outputs)

        Returns:
            Dictionary mapping values to counts
        """
        distribution = defaultdict(int)
        for ex in self.examples:
            value = ex.inputs.get(field_name, ex.outputs.get(field_name, "missing"))
            distribution[str(value)] += 1
        return dict(distribution)

    def get_statistics(self) -> dict[str, Any]:
        """
        Get dataset statistics.

        Returns:
            Dictionary with statistics
        """
        stats = {
            "total_examples": len(self.examples),
            "name": self.name,
            "has_signature": self.signature is not None,
        }

        if self.signature:
            stats["signature_name"] = self.signature.name
            stats["input_fields"] = [f.name for f in self.signature.inputs]
            stats["output_fields"] = [f.name for f in self.signature.outputs]

        # Analyze feedback presence
        feedback_counts = defaultdict(int)
        for ex in self.examples:
            if ex.feedback:
                for feedback_type in ex.feedback.keys():
                    feedback_counts[feedback_type] += 1

        if feedback_counts:
            stats["feedback_coverage"] = {
                feedback_type: count / len(self.examples)
                for feedback_type, count in feedback_counts.items()
            }

        return stats

    def print_summary(self) -> None:
        """Print a human-readable summary of the dataset."""
        stats = self.get_statistics()

        print(f"ExampleSet: {stats['name']}")
        print(f"  Total examples: {stats['total_examples']}")

        if stats['has_signature']:
            print(f"  Signature: {stats['signature_name']}")
            print(f"    Inputs: {', '.join(stats['input_fields'])}")
            print(f"    Outputs: {', '.join(stats['output_fields'])}")

        if 'feedback_coverage' in stats:
            print(f"  Feedback coverage:")
            for feedback_type, coverage in stats['feedback_coverage'].items():
                print(f"    {feedback_type}: {coverage:.1%}")

    def to_list(self) -> list[dict[str, Any]]:
        """
        Convert to list of dictionaries.

        Returns:
            List of dicts with 'inputs', 'outputs', 'feedback', 'metadata'
        """
        return [
            {
                "inputs": ex.inputs,
                "outputs": ex.outputs,
                "feedback": ex.feedback,
                "metadata": ex.metadata,
            }
            for ex in self.examples
        ]

    @classmethod
    def from_tuples(
        cls,
        tuples: list[tuple[str, str]],
        input_key: str = "input",
        output_key: str = "target",
        name: str = "dataset",
        signature: Signature | None = None
    ) -> ExampleSet:
        """
        Create ExampleSet from list of (input, target) tuples (legacy format).

        Args:
            tuples: List of (input_text, target_text) tuples
            input_key: Key to use for input field
            output_key: Key to use for output field
            name: Dataset name
            signature: Optional signature

        Returns:
            New ExampleSet
        """
        examples = [Example.from_tuple(t, input_key, output_key) for t in tuples]
        return cls(examples=examples, name=name, signature=signature)

    def to_tuples(self, input_key: str = "input", output_key: str = "target") -> list[tuple[str, str]]:
        """
        Convert to list of (input, target) tuples for backward compatibility.

        Args:
            input_key: Key of input field to use
            output_key: Key of output field to use

        Returns:
            List of (input_text, target_text) tuples
        """
        return [ex.to_tuple(input_key, output_key) for ex in self.examples]

    @classmethod
    def from_list(
        cls,
        data: list[dict[str, Any]],
        name: str = "dataset",
        signature: Signature | None = None
    ) -> ExampleSet:
        """
        Create ExampleSet from list of dictionaries.

        Args:
            data: List of dicts with 'inputs' and 'outputs' keys
            name: Dataset name
            signature: Optional signature

        Returns:
            New ExampleSet
        """
        examples = [
            Example(
                inputs=item.get("inputs", {}),
                outputs=item.get("outputs", {}),
                feedback=item.get("feedback"),
                metadata=item.get("metadata", {}),
                reasoning_trace=item.get("reasoning_trace")
            )
            for item in data
        ]
        return cls(examples=examples, name=name, signature=signature)
