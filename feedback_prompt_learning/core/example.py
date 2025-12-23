from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

from feedback_prompt_learning.core.signature import Signature


class Example(BaseModel):
    """
    A training/evaluation example with ergonomic builder pattern.
    Two ways to create examples:

    1. Explicit (verbose but clear):
        Example(
            inputs={"question": "What is 2+2?"},
            outputs={"answer": "4"}
        )

    2. Builder pattern (ergonomic):
        Example(question="What is 2+2?", answer="4").with_inputs("question")

        # Multiple inputs
        Example(
            text="Hello",
            context="greeting",
            label="positive"
        ).with_inputs("text", "context")

        # Explicitly set outputs too
        Example(
            question="...",
            reasoning="...",
            answer="..."
        ).with_inputs("question").with_outputs("reasoning", "answer")
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    # Core fields
    inputs: dict[str, Any] = Field(default_factory=dict, description="Input field values")
    outputs: dict[str, Any] = Field(default_factory=dict, description="Expected output values")

    # Optional enhancements
    feedback: dict[str, str] | None = Field(None, description="Structured feedback")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    reasoning_trace: list[str] | None = Field(None, description="Step-by-step reasoning")

    # Internal: Store kwargs until inputs/outputs are designated
    pending_fields: dict[str, Any] = Field(default_factory=dict, exclude=True)

    def __init__(self, **data):
        """
        Initialize example with flexible kwargs.

        Examples:
            # Explicit
            Example(inputs={"q": "..."}, outputs={"a": "..."})

            # Builder pattern - fields go to _pending_fields
            Example(question="...", answer="...")

            # Mixed (explicit takes precedence)
            Example(inputs={"q": "..."}, outputs={"a": "..."}, meta="value")
        """
        # Separate known fields from arbitrary kwargs
        inputs = data.pop("inputs", {})
        outputs = data.pop("outputs", {})
        feedback = data.pop("feedback", None)
        metadata = data.pop("metadata", {})
        reasoning_trace = data.pop("reasoning_trace", None)

        # Everything else goes to pending fields (for builder pattern)
        pending_fields = {k: v for k, v in data.items()
                if k not in ["pending_fields"]}

        super().__init__(
            inputs=inputs,
            outputs=outputs,
            feedback=feedback,
            metadata=metadata,
            reasoning_trace=reasoning_trace,
            pending_fields=pending_fields
        )

    def with_inputs(self, *field_names: str) -> Example:
        """
        Designate fields as inputs (builder pattern).

        Args:
            *field_names: Names of fields to treat as inputs

        Returns:
            New Example with fields moved to inputs

        Example:
            Example(question="Q1", answer="A1").with_inputs("question")
            # Results in: inputs={"question": "Q1"}, outputs={"answer": "A1"}

            Example(text="...", context="...", label="...").with_inputs("text", "context")
            # Results in: inputs={"text": "...", "context": "..."}, outputs={"label": "..."}
            """
        new_inputs = dict(self.inputs)
        new_outputs = dict(self.outputs)
        remaining_pending = dict(self.pending_fields)

        # Move specified fields from pending to inputs
        for field_name in field_names:
            if field_name in remaining_pending:
                new_inputs[field_name] = remaining_pending.pop(field_name)
            elif field_name in new_outputs:
                # Allow moving from outputs to inputs
                new_inputs[field_name] = new_outputs.pop(field_name)

        # Everything remaining in pending goes to outputs (default behavior)
        for field_name, value in remaining_pending.items():
            if field_name not in new_outputs:
                new_outputs[field_name] = value

        return Example(
            inputs=new_inputs,
            outputs=new_outputs,
            feedback=self.feedback,
            metadata=self.metadata,
            reasoning_trace=self.reasoning_trace
        )

    def with_outputs(self, *field_names: str) -> Example:
        """
        Explicitly designate fields as outputs (builder pattern).

        Args:
            *field_names: Names of fields to treat as outputs

        Returns:
            New Example with fields moved to outputs

        Example:
            Example(
                question="Q1",
                reasoning="Step 1...",
                answer="A1"
            ).with_inputs("question").with_outputs("reasoning", "answer")
            """
        new_inputs = dict(self.inputs)
        new_outputs = dict(self.outputs)
        remaining_pending = dict(self.pending_fields)

        # Move specified fields from pending to outputs
        for field_name in field_names:
            if field_name in remaining_pending:
                new_outputs[field_name] = remaining_pending.pop(field_name)
            elif field_name in new_inputs:
                # Allow moving from inputs to outputs
                new_outputs[field_name] = new_inputs.pop(field_name)

        # Remaining pending fields stay in pending
        return Example(
            inputs=new_inputs,
            outputs=new_outputs,
            feedback=self.feedback,
            metadata=self.metadata,
            reasoning_trace=self.reasoning_trace,
            pending_fields=remaining_pending
        )

    def with_feedback(
        self,
        accuracy_feedback: str | None = None,
        reasoning_feedback: str | None = None,
        prompt_feedback: str | None = None,
        **custom_feedback: str
    ) -> Example:
        """
        Add feedback to this example (builder pattern).

        Args:
            accuracy_feedback: Feedback on correctness
            reasoning_feedback: Feedback on reasoning quality
            prompt_feedback: Feedback on prompt effectiveness
            **custom_feedback: Additional custom feedback fields

        Returns:
            New Example with feedback added

        Example:
            example.with_feedback(
                accuracy_feedback="Correct answer ✓",
                reasoning_feedback="Good step-by-step approach"
            )
            """
        feedback = {}
        if accuracy_feedback:
            feedback["accuracy"] = accuracy_feedback
        if reasoning_feedback:
            feedback["reasoning"] = reasoning_feedback
        if prompt_feedback:
            feedback["prompt"] = prompt_feedback
        feedback.update(custom_feedback)

        return Example(
            inputs=self.inputs,
            outputs=self.outputs,
            feedback=feedback,
            metadata=self.metadata,
            reasoning_trace=self.reasoning_trace
        )

    def with_metadata(self, **metadata: Any) -> Example:
        """
        Add metadata to this example (builder pattern).

        Example:
            example.with_metadata(source="dataset_v1", difficulty="hard")
        """
        new_metadata = {**self.metadata, **metadata}
        return Example(
            inputs=self.inputs,
            outputs=self.outputs,
            feedback=self.feedback,
            metadata=new_metadata,
            reasoning_trace=self.reasoning_trace
        )

    def __getitem__(self, key: str) -> Any:
        """
        Allow dict-like access: example['question']

        Searches in order: inputs → outputs → metadata → pending
        """
        if key in self.inputs:
            return self.inputs[key]
        if key in self.outputs:
            return self.outputs[key]
        if key in self.metadata:
            return self.metadata[key]
        if key in self.pending_fields:
            return self.pending_fields[key]
        raise KeyError(f"Field '{key}' not found in example")

    def __repr__(self) -> str:
        """Clean representation."""
        parts = []
        if self.inputs:
            parts.append(f"inputs={list(self.inputs.keys())}")
        if self.outputs:
            parts.append(f"outputs={list(self.outputs.keys())}")
        if self.pending_fields:
            parts.append(f"pending={list(self.pending_fields.keys())}")
        return f"Example({', '.join(parts)})"

    # Validation methods (from previous design)
    def validate_against_signature(self, signature: Signature) -> bool:
        """Check if this example matches the signature."""
        from feedback_prompt_learning.core.signature import Signature

        for field in signature.inputs:
            if field.required and field.name not in self.inputs:
                return False
        for field in signature.outputs:
            if field.required and field.name not in self.outputs:
                return False
        return True

    def get_missing_fields(self, signature: Signature) -> dict[str, list[str]]:
        """Return missing fields for debugging."""
        from feedback_prompt_learning.core.signature import Signature

        missing_inputs = [
            f.name for f in signature.inputs
            if f.required and f.name not in self.inputs
        ]
        missing_outputs = [
            f.name for f in signature.outputs
            if f.required and f.name not in self.outputs
        ]
        return {"inputs": missing_inputs, "outputs": missing_outputs}

    @classmethod
    def from_dict(cls, data: dict[str, Any], input_keys: list[str] | None = None) -> Example:
        """
        Create from a flat dictionary with optional input key specification.

        Args:
            data: Dictionary with all fields
            input_keys: Optional list of keys to treat as inputs (rest are outputs)

        Example:
            # Infer automatically
            Example.from_dict({"question": "Q1", "answer": "A1"})

            # Explicit inputs
            Example.from_dict(
                {"text": "...", "context": "...", "label": "..."},
                input_keys=["text", "context"]
            )
        """
        if input_keys:
            inputs = {k: data[k] for k in input_keys if k in data}
            outputs = {k: v for k, v in data.items() if k not in input_keys}
            return cls(inputs=inputs, outputs=outputs)
        else:
            # Use heuristic (common input/output field names)
            input_names = {"question", "context", "query", "text", "input", "prompt"}
            output_names = {"answer", "output", "label", "prediction", "response"}

            inputs = {k: v for k, v in data.items() if k in input_names}
            outputs = {k: v for k, v in data.items() if k in output_names}
            metadata = {k: v for k, v in data.items()
                       if k not in inputs and k not in outputs}

            return cls(inputs=inputs, outputs=outputs, metadata=metadata)
