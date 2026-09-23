from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, computed_field

from feedback_prompt_learning.core.signature import Signature


class PromptVersion(BaseModel):
    """
    A single prompt with performance tracking.

    Features:
    - Prompt text and signature
    - Performance scores (train/eval/test)
    - Per-example results with feedback
    - Parent-child relationships for tree/graph structure
    - Metadata for optimization methods

    Example:
        # Create prompt
        prompt = PromptVersion(
            prompt_text="Answer the question concisely.",
            signature=qa_signature,
            version=0
        )

        # After evaluation
        prompt.train_score = 0.75
        prompt.eval_score = 0.72
        prompt.example_results = [result1, result2, ...]

        # Create child prompt
        child = PromptVersion(
            prompt_text="Answer with step-by-step reasoning.",
            signature=qa_signature,
            version=1,
            parent_version=0,
            improvement_note="Add reasoning steps"
        )
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Core content
    prompt_text: str = Field(description="The actual prompt text")
    signature: Signature = Field(description="Input/output specification")

    # Version tracking
    version: int = Field(description="Version number in optimization")
    parent_version: int | None = Field(None, description="Parent prompt version (if any)")

    # Performance metrics
    train_score: float | None = Field(None, description="Score on training set")
    eval_score: float | None = Field(None, description="Score on eval set")
    test_score: float | None = Field(None, description="Score on test set")

    # Optimization metadata
    improvement_note: str | None = Field(
        None,
        description="Note about what was improved from parent"
    )
    creation_method: str = Field(
        "manual",
        description="How created (manual, mcts, beam_search, evolutionary, etc.)"
    )

    # Metadata
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    @computed_field
    @property
    def char_count(self) -> int:
        """Character count of prompt."""
        return len(self.prompt_text)

    def __repr__(self) -> str:
        """String representation."""
        score_str = f"eval={self.eval_score:.3f}" if self.eval_score else "eval=None"
        return f"PromptV{self.version}({score_str})"

    def __str__(self) -> str:
        """Human-readable string."""
        lines = [
            f"PromptVersion {self.version}:",
            f"  Method: {self.creation_method}",
            f"  Train: {self.train_score:.3f if self.train_score else 'N/A'}",
            f"  Eval:  {self.eval_score:.3f if self.eval_score else 'N/A'}",
            f"  Length: {self.word_count} words ({self.char_count} chars)",
        ]
        if self.improvement_note:
            lines.append(f"  Improvement: {self.improvement_note[:80]}...")
        return "\n".join(lines)
