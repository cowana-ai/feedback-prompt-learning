"""
Feedback data structures for prompt optimization.
Provides type-safe, structured feedback instead of raw dictionaries.
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Feedback:
    """
    Structured feedback for a single evaluation.

    Attributes:
        accuracy_feedback: Why the answer was correct/incorrect
        reasoning_feedback: Quality of reasoning (step-by-step, clarity, logic)
        prompt_feedback: How the prompt could be improved for this case
    """
    accuracy_feedback: str
    reasoning_feedback: str | None = None
    prompt_feedback: str | None = None

    def has_reasoning(self) -> bool:
        """Check if reasoning feedback is available"""
        return self.reasoning_feedback is not None and len(self.reasoning_feedback.strip()) > 0

    def has_prompt_feedback(self) -> bool:
        """Check if prompt feedback is available"""
        return self.prompt_feedback is not None and len(self.prompt_feedback.strip()) > 0

    def to_dict(self) -> dict:
        """Convert to dictionary format for backward compatibility"""
        return {
            "accuracy_feedback": self.accuracy_feedback,
            "reasoning_feedback": self.reasoning_feedback,
            "prompt_feedback": self.prompt_feedback,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Feedback':
        """Create Feedback from dictionary"""
        return cls(
            accuracy_feedback=data.get("accuracy_feedback", ""),
            reasoning_feedback=data.get("reasoning_feedback"),
            prompt_feedback=data.get("prompt_feedback"),
        )


@dataclass
class EvaluationResult:
    """
    Complete evaluation result for a single example.

    Attributes:
        input: The input question/prompt
        output: The model's response
        target: The expected answer
        score: Numerical score (0.0 to 1.0)
        feedback: Structured feedback object
    """
    input: str
    output: str
    target: str
    score: float
    feedback: Feedback | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary format for backward compatibility"""
        result = {
            "input": self.input,
            "output": self.output,
            "target": self.target,
            "score": self.score,
        }
        if self.feedback:
            result["feedback"] = self.feedback.to_dict()
        return result

    @classmethod
    def from_dict(cls, data: dict) -> 'EvaluationResult':
        """Create EvaluationResult from dictionary"""
        feedback = None
        if "feedback" in data:
            feedback = Feedback.from_dict(data["feedback"])

        return cls(
            input=data["input"],
            output=data["output"],
            target=data["target"],
            score=data["score"],
            feedback=feedback,
        )


@dataclass
class RewardOutput:
    """
    Output from a reward function.

    Attributes:
        score: Numerical reward (0.0 to 1.0)
        feedback: Structured feedback explaining the score
    """
    score: float
    feedback: Feedback

    def to_tuple(self) -> tuple[float, dict]:
        """Convert to (score, feedback_dict) tuple for backward compatibility"""
        return self.score, self.feedback.to_dict()

    @classmethod
    def from_tuple(cls, data: tuple) -> 'RewardOutput':
        """Create RewardOutput from (score, feedback_dict) tuple"""
        score, feedback_dict = data
        return cls(
            score=score,
            feedback=Feedback.from_dict(feedback_dict)
        )
