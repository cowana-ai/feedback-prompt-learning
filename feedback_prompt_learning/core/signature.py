from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class FieldType(str, Enum):
    """Field types for signature fields."""
    INPUT = "input"
    OUTPUT = "output"
    INTERMEDIATE = "intermediate"


@dataclass
class SignatureField:
    """A single field in a signature."""
    name: str
    description: str
    field_type: FieldType
    dtype: type = str
    required: bool = True
    default: Any = None

    def __str__(self) -> str:
        prefix = {
            FieldType.INPUT: "→",
            FieldType.OUTPUT: "←",
            FieldType.INTERMEDIATE: "○"
        }[self.field_type]
        return f"{prefix} {self.name}: {self.description}"


class Signature(BaseModel):
    """
    Declarative specification of a prompt's input/output structure.

    Examples:
        # Simple QA
        Signature(
            inputs=[SignatureField("question", "A question to answer", FieldType.INPUT)],
            outputs=[SignatureField("answer", "The answer", FieldType.OUTPUT)]
        )

        # Chain-of-thought reasoning
        Signature(
            inputs=[SignatureField("question", "A math problem", FieldType.INPUT)],
            outputs=[
                SignatureField("reasoning", "Step-by-step solution", FieldType.INTERMEDIATE),
                SignatureField("answer", "Final numeric answer", FieldType.OUTPUT)
            ]
        )
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = Field(description="Name of this signature")
    inputs: list[SignatureField] = Field(description="Input fields")
    outputs: list[SignatureField] = Field(description="Output fields")
    instructions: str | None = Field(None, description="High-level task description")

    def __str__(self) -> str:
        input_names = ", ".join(f.name for f in self.inputs)
        output_names = ", ".join(f.name for f in self.outputs)
        return f"{self.name}: {input_names} → {output_names}"
