"""
World Model Interface for MCTS
Handles all data management and evaluation for MCTS prompt optimization
"""

import random
from abc import ABC, abstractmethod
from collections.abc import Callable

import numpy as np
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from feedback_prompt_learning.core import Example, ExampleSet
from feedback_prompt_learning.data import EvaluationResult, RewardOutput


class WorldModel(ABC):
    """
    Abstract base class for MCTS world models.
    Handles data management, evaluation, and reward calculation.
    """

    @abstractmethod
    async def evaluate_prompt(
        self,
        prompt: str,
        batch_type: str = 'train'
    ) -> tuple[float, list[EvaluationResult]]:
        """
        Evaluate a prompt on a batch of data.

        Args:
            prompt: The prompt to evaluate
            batch_type: 'train' or 'eval'

        Returns:
            Tuple of (average_score, list_of_evaluation_results)
        """
        pass

    @abstractmethod
    def get_batch(self, batch_type: str = 'train') -> ExampleSet:
        """
        Get next batch of data.

        Args:
            batch_type: 'train' or 'eval'

        Returns:
            ExampleSet batch
        """
        pass

    @abstractmethod
    def get_fixed_eval_batch(self, depth: int) -> ExampleSet:
        """
        Get a fixed evaluation batch for a given depth.
        All nodes at same depth should use the same batch for fair comparison.

        Args:
            depth: Node depth in the tree

        Returns:
            Fixed evaluation batch for this depth
        """
        pass

    @abstractmethod
    def sample_examples(
        self,
        all_evaluations: list[EvaluationResult],
        num_examples: int = 5,
    ) -> list[EvaluationResult]:
        """
        Sample examples from evaluation history for gradient analysis.
        Different algorithms may use different strategies (e.g., errors only, stratified, random).

        Args:
            all_evaluations: All evaluation results collected so far
            num_examples: Number of examples to sample

        Returns:
            List of sampled EvaluationResult objects
        """
        pass

    def calculate_reward(
        self,
        raw_output: str,
        target: str,
        prompt: str,
        input_text: str
    ) -> RewardOutput:
        """Calculate reward using reward function"""
        cleaned_output = self.clean_response_fn(raw_output)
        output = self.reward_fn(cleaned_output, target, prompt, raw_output, input_text)
        # Handle both old tuple format and new RewardOutput
        if isinstance(output, RewardOutput):
            return output
        elif isinstance(output, tuple):
            # Legacy format: (score, feedback_dict)
            return RewardOutput.from_tuple(output)
        else:
            raise TypeError(f"Reward function must return RewardOutput or (float, dict) tuple, got {type(output)}")

class PromptOptimizationWorldModel(WorldModel):
    """
    World model for prompt optimization tasks.
    Manages train/eval datasets, batching, and LLM-based evaluation.
    """

    def __init__(
        self,
        train_dataset: ExampleSet,
        eval_dataset: ExampleSet,
        llm: ChatOpenAI,
        reward_fn: Callable[[str, str, str], tuple[float, str]],
        clean_response_fn: Callable[[str], str],
        minibatch_size_train: int = 5,
        minibatch_size_eval: int = 5,
        post_instruction: bool = True,
    ):
        """
        Initialize the world model.

        Args:
            train_dataset: Training ExampleSet
            eval_dataset: Evaluation ExampleSet
            llm: Language model for evaluation
            reward_fn: Function to calculate reward (output, target, prompt) -> (score, feedback)
            clean_response_fn: Function to clean LLM responses
            minibatch_size_train: Batch size for training data
            minibatch_size_eval: Batch size for evaluation data
            post_instruction: If True, format as "input\nprompt", else "prompt\ninput"
        """
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.llm = llm
        self.reward_fn = reward_fn
        self.clean_response_fn = clean_response_fn
        self.minibatch_size_train = minibatch_size_train
        self.minibatch_size_eval = minibatch_size_eval
        self.post_instruction = post_instruction

        # Initialize batching
        self.train_dataset_copy = ExampleSet(examples=list(train_dataset.examples), name="train_copy")
        self.eval_dataset_copy = ExampleSet(examples=list(eval_dataset.examples), name="eval_copy")
        self.current_batch_idx = 0
        self.current_eval_batch_idx = 0

        # Create initial batches
        self._create_batches('train')
        self._create_batches('eval')

        # Cache for depth-based eval batches
        self.depth_eval_batches: dict[int, ExampleSet] = {}

    def _create_batches(self, dataset_type: str = 'train'):
        """Create batches from dataset"""
        if dataset_type == 'train':
            dataset_copy = self.train_dataset_copy
            random.shuffle(dataset_copy.examples)
            dataset_len = len(dataset_copy)
            batch_size = self.minibatch_size_train
            num_batches = max(1, dataset_len // batch_size)
            batches = []

            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min(start_idx + batch_size, dataset_len)
                batch_examples = dataset_copy.examples[start_idx:end_idx]
                batches.append(ExampleSet(examples=batch_examples, name=f"train_batch_{i}"))

            self.num_batches = num_batches
            self.batches = batches

        elif dataset_type == 'eval':
            if not self.eval_dataset_copy.examples:
                self.num_eval_batches = 0
                self.eval_batches = []
                return

            dataset_copy = self.eval_dataset_copy
            random.shuffle(dataset_copy.examples)
            dataset_len = len(dataset_copy)
            batch_size = self.minibatch_size_eval
            num_batches = max(1, dataset_len // batch_size)
            batches = []

            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min(start_idx + batch_size, dataset_len)
                batch_examples = dataset_copy.examples[start_idx:end_idx]
                batches.append(ExampleSet(examples=batch_examples, name=f"eval_batch_{i}"))

            self.num_eval_batches = num_batches
            self.eval_batches = batches

    def get_batch(self, batch_type: str = 'train') -> ExampleSet:
        """Get next batch of data"""
        if batch_type == 'train':
            if not self.batches:
                return ExampleSet(examples=[], name="empty_train_batch")

            batch = self.batches[self.current_batch_idx]
            self.current_batch_idx = (self.current_batch_idx + 1) % self.num_batches

            # Reshuffle when we complete an epoch
            if self.current_batch_idx == 0:
                random.shuffle(self.train_dataset_copy.examples)
                self._create_batches('train')

            return batch

        elif batch_type == 'eval':
            if not self.eval_batches:
                return ExampleSet(examples=[], name="empty_eval_batch")

            batch = self.eval_batches[self.current_eval_batch_idx]
            self.current_eval_batch_idx = (self.current_eval_batch_idx + 1) % self.num_eval_batches

            # Reshuffle when we complete an epoch
            if self.current_eval_batch_idx == 0:
                random.shuffle(self.eval_dataset_copy.examples)
                self._create_batches('eval')

            return batch

        else:
            raise ValueError(f"Unknown batch_type: {batch_type}")

    def get_fixed_eval_batch(self, depth: int) -> ExampleSet:
        """Get a fixed evaluation batch for a given depth."""
        # TODO: Implement depth-based batch selection if needed
        return ExampleSet(examples=list(self.eval_dataset.examples), name=f"fixed_eval_depth_{depth}")  # Always return full eval dataset

    def prepare_input(self, example: Example) -> str:
        """Prepare input text from a single example"""
        if len(example.inputs) == 1:
            # Single input field
            return str(list(example.inputs.values())[0]).strip()
        else:
            # Multiple input fields - format as key: value pairs
            parts = []
            for key, value in example.inputs.items():
                parts.append(f"{key}: {value}")
            return "\n".join(parts
                             ).strip()
    def prepare_inputs(self, examples: list[Example]) -> list[str]:
        """Prepare input texts from examples"""
        input_texts = []
        for example in examples:
            input_text = self.prepare_input(example)
            input_texts.append(input_text)
        return input_texts

    def prepare_output(self, example: Example) -> str:
        """Prepare target text from a single example"""
        if len(example.outputs) == 1:
            return str(list(example.outputs.values())[0]).strip()
        else:
            return "\n".join(f"{k}: {v}" for k, v in example.outputs.items()).strip()

    def prepare_outputs(self, examples: list[Example]) -> list[str]:
        """Prepare target texts from examples"""
        target_texts = []
        for example in examples:
            target_text = self.prepare_output(example)
            target_texts.append(target_text)
        return target_texts

    def build_prompts(self, examples: list[Example], prompt: str) -> list[str]:
        """Build full prompts by combining inputs with instruction prompt"""
        input_texts = self.prepare_inputs(examples)
        if self.post_instruction:
            return [f'{input_text}\n{prompt}' for input_text in input_texts]
        else:
            return [f'{prompt}\n{input_text}' for input_text in input_texts]

    async def evaluate_prompt(
        self,
        prompt: str,
        dataset: ExampleSet,
        eval: bool = False
    ) -> tuple[float, list[EvaluationResult]]:
        """
        Evaluate a prompt on a batch of data.

        Returns:
            Tuple of (average_score, list_of_evaluation_results)
        """
        scores = []
        evaluations = []

        # Process in batches to avoid overwhelming API
        batch_size = self.minibatch_size_eval
        examples = dataset.examples
        for i in range(0, len(examples), batch_size):
            batch_examples = examples[i:i + batch_size]
            full_prompts = self.build_prompts(batch_examples, prompt)
            messages_batch = [[HumanMessage(content=p)] for p in full_prompts]

            # Batch LLM inference
            responses = await self.llm.abatch(messages_batch)

            # Calculate scores and collect feedback
            for response, example in zip(responses, batch_examples):
                raw_output = response.content
                input_text = self.prepare_input(example)
                target = self.prepare_output(example)

                reward_output = self.calculate_reward(raw_output, target, prompt, input_text if not eval else "")
                scores.append(reward_output.score)

                evaluation = EvaluationResult(
                    input=input_text,
                    output=raw_output,
                    target=target,
                    score=reward_output.score,
                    feedback=reward_output.feedback
                )
                evaluations.append(evaluation)

        avg_score = np.mean(scores) if scores else 0.0

        return avg_score, evaluations

    async def evaluate_prompt_with_feedback(
        self,
        prompt: str,
        batch: ExampleSet
    ) -> list[EvaluationResult]:
        """Evaluate prompt and return detailed feedback (without average)"""
        _, evaluations = await self.evaluate_prompt(prompt, batch)
        return evaluations

    def get_dataset_info(self) -> dict[str, int]:
        """Get information about datasets"""
        return {
            'train_size': len(self.train_dataset),
            'eval_size': len(self.eval_dataset),
            'train_batches': self.num_batches,
            'eval_batches': self.num_eval_batches,
            'minibatch_size_train': self.minibatch_size_train,
            'minibatch_size_eval': self.minibatch_size_eval,
        }

    def sample_examples(
        self,
        all_evaluations: list[EvaluationResult],
        num_examples: int = 5,
    ) -> list[EvaluationResult]:
        """Sample most recent examples for gradient analysis"""
        if not all_evaluations:
            return []
        batch_evals = all_evaluations[-num_examples:]

        return batch_evals  # Return recent examples (could filter by score if needed)


class PromptAgentWorldModel(PromptOptimizationWorldModel):
    """
    World model for prompt agents that do not require feedback-based sampling.
    Uses random sampling from all evaluations.
    """
    def get_fixed_eval_batch(self, depth):
        return ExampleSet(examples=list(self.eval_dataset.examples), name=f"fixed_eval_depth_{depth}")  # Always return full dataset

    async def evaluate_prompt(
        self,
        prompt: str,
        dataset: ExampleSet,
        eval: bool = False
    ) -> tuple[float, list[EvaluationResult]]:
        """
        Evaluate a prompt on the entire dataset with batched inference.
        Processes dataset in chunks to avoid overwhelming the LLM API.

        Args:
            prompt: The prompt to evaluate
            dataset: Full dataset to evaluate on

        Returns:
            Tuple of (average_score, list_of_detailed_evaluations)
        """
        scores = []
        evaluations = []

        # Process in batches to avoid overwhelming API
        batch_size = self.minibatch_size_eval
        examples = dataset.examples
        for i in range(0, len(examples), batch_size):
            batch_examples = examples[i:i + batch_size]
            full_prompts = self.build_prompts(batch_examples, prompt)
            messages_batch = [[HumanMessage(content=p)] for p in full_prompts]

            # Batch LLM inference
            responses = await self.llm.abatch(messages_batch)

            # Calculate scores and collect feedback
            for response, example in zip(responses, batch_examples):
                raw_output = response.content
                input_text = self.prepare_input(example)
                target = self.prepare_output(example)

                reward_output = self.calculate_reward(raw_output, target, prompt, input_text if not eval else "")
                scores.append(reward_output.score)
                evaluation = EvaluationResult(
                    input=input_text,
                    output=raw_output,
                    target=target,
                    score=reward_output.score,
                    feedback=reward_output.feedback
                )
                evaluations.append(evaluation)

        avg_score = np.mean(scores) if scores else 0.0

        return avg_score, evaluations

    def sample_examples(
        self,
        all_evaluations: list[EvaluationResult],
        num_examples: int = 5,
    ) -> list[EvaluationResult]:
        if not all_evaluations:
            return []
        batch_evals = all_evaluations[-num_examples:]

        return batch_evals #[ev for ev in batch_evals if ev['score'] == 0]

    def build_prompts(self, examples: list[Example], prompt: str) -> list[str]:
        """Build full prompts by combining inputs with instruction prompt"""
        input_texts = self.prepare_inputs(examples)

        output_format = "At the end show the answer option bracketed between <answer> and </answer>."
        if self.post_instruction:
            return [f'{input_text}\n{prompt}\n{output_format}' for input_text in input_texts]
        else:
            return [f'{prompt}\n{input_text}\n{output_format}' for input_text in input_texts]
