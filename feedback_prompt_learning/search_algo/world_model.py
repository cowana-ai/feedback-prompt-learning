"""
World Model Interface for MCTS
Handles all data management and evaluation for MCTS prompt optimization
"""

import random
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Dict, List, Optional, Tuple

import numpy as np
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from feedback_prompt_learning import cfg


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
    ) -> tuple[float, list[dict]]:
        """
        Evaluate a prompt on a batch of data.

        Args:
            prompt: The prompt to evaluate
            batch_type: 'train' or 'eval'

        Returns:
            Tuple of (average_score, list_of_evaluations)
        """
        pass

    @abstractmethod
    def get_batch(self, batch_type: str = 'train') -> list[tuple[str, str]]:
        """
        Get next batch of data.

        Args:
            batch_type: 'train' or 'eval'

        Returns:
            List of (input, target) tuples
        """
        pass

    @abstractmethod
    def get_fixed_eval_batch(self, depth: int) -> list[tuple[str, str]]:
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
        all_evaluations: list[dict],
        num_examples: int = 5,
    ) -> list[dict]:
        """
        Sample examples from evaluation history for gradient analysis.
        Different algorithms may use different strategies (e.g., errors only, stratified, random).

        Args:
            all_evaluations: All evaluation results collected so far
            num_examples: Number of examples to sample

        Returns:
            List of sampled evaluation dictionaries
        """
        pass

    def calculate_reward(
        self,
        raw_output: str,
        target: str,
        prompt: str
    ) -> tuple[float, str]:
        """Calculate reward using reward function"""
        cleaned_output = self.clean_response_fn(raw_output)

        # Check if reward function accepts raw_output parameter
        import inspect
        sig = inspect.signature(self.reward_fn)
        if 'raw_output' in sig.parameters:
            output = self.reward_fn(
                cleaned_output, target, prompt, raw_output=raw_output
            )
        else:
            output = self.reward_fn(cleaned_output, target, prompt)

        return output

class PromptOptimizationWorldModel(WorldModel):
    """
    World model for prompt optimization tasks.
    Manages train/eval datasets, batching, and LLM-based evaluation.
    """

    def __init__(
        self,
        train_dataset: list[tuple[str, str]],
        eval_dataset: list[tuple[str, str]],
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
            train_dataset: Training data as (input, target) pairs
            eval_dataset: Evaluation data as (input, target) pairs
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
        self.train_dataset_copy = list(train_dataset)
        self.eval_dataset_copy = list(eval_dataset)
        self.current_batch_idx = 0
        self.current_eval_batch_idx = 0

        # Create initial batches
        self._create_batches('train')
        self._create_batches('eval')

        # Cache for depth-based eval batches
        self.depth_eval_batches: dict[int, list[tuple[str, str]]] = {}

    def _create_batches(self, dataset_type: str = 'train'):
        """Create batches from dataset"""
        if dataset_type == 'train':
            dataset_copy = self.train_dataset_copy
            random.shuffle(dataset_copy)
            dataset_len = len(dataset_copy)
            batch_size = self.minibatch_size_train
            num_batches = max(1, dataset_len // batch_size)
            batches = []

            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min(start_idx + batch_size, dataset_len)
                batches.append(dataset_copy[start_idx:end_idx])

            self.num_batches = num_batches
            self.batches = batches

        elif dataset_type == 'eval':
            if not self.eval_dataset_copy:
                self.num_eval_batches = 0
                self.eval_batches = []
                return

            dataset_copy = self.eval_dataset_copy
            random.shuffle(dataset_copy)
            dataset_len = len(dataset_copy)
            batch_size = self.minibatch_size_eval
            num_batches = max(1, dataset_len // batch_size)
            batches = []

            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min(start_idx + batch_size, dataset_len)
                batches.append(dataset_copy[start_idx:end_idx])

            self.num_eval_batches = num_batches
            self.eval_batches = batches

    def get_batch(self, batch_type: str = 'train') -> list[tuple[str, str]]:
        """Get next batch of data"""
        if batch_type == 'train':
            if not self.batches:
                return []

            batch = self.batches[self.current_batch_idx]
            self.current_batch_idx = (self.current_batch_idx + 1) % self.num_batches

            # Reshuffle when we complete an epoch
            if self.current_batch_idx == 0:
                random.shuffle(self.train_dataset_copy)
                self._create_batches('train')

            return batch

        elif batch_type == 'eval':
            if not self.eval_batches:
                return []

            batch = self.eval_batches[self.current_eval_batch_idx]
            self.current_eval_batch_idx = (self.current_eval_batch_idx + 1) % self.num_eval_batches

            # Reshuffle when we complete an epoch
            if self.current_eval_batch_idx == 0:
                random.shuffle(self.eval_dataset_copy)
                self._create_batches('eval')

            return batch

        else:
            raise ValueError(f"Unknown batch_type: {batch_type}")

    def get_fixed_eval_batch(self, depth: int) -> list[tuple[str, str]]:
        """
        Get a fixed evaluation batch for a given depth.
        All nodes at same depth use same batch for fair comparison.
        """
        # if depth not in self.depth_eval_batches:
        #     # First time we see this depth - get next eval batch and cache it
        #     eval_batch = self.get_batch('eval')
        #     self.depth_eval_batches[depth] = eval_batch

        return self.eval_dataset #self.depth_eval_batches[depth]

    def build_prompts(self, inputs: list[str], prompt: str) -> list[str]:
        """Build full prompts by combining inputs with instruction prompt"""
        if self.post_instruction:
            return [f'{input_text}\n{prompt}' for input_text in inputs]
        else:
            return [f'{prompt}\n{input_text}' for input_text in inputs]

    async def evaluate_prompt(
        self,
        prompt: str,
        dataset: list[tuple[str, str]]
    ) -> tuple[float, list[dict]]:
        """
        Evaluate a prompt on a batch of data.

        Returns:
            Tuple of (average_score, list_of_detailed_evaluations)
        """
        scores = []
        evaluations = []

        # Process in batches to avoid overwhelming API
        batch_size = self.minibatch_size_eval
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i + batch_size]
            inputs = [input_text for input_text, _ in batch]
            full_prompts = self.build_prompts(inputs, prompt)
            messages_batch = [[HumanMessage(content=p)] for p in full_prompts]

            # Batch LLM inference
            responses = await self.llm.abatch(messages_batch)

            # Calculate scores and collect feedback
            for response, (input_text, target) in zip(responses, batch):
                raw_output = response.content
                score, feedback = self.calculate_reward(raw_output, target, prompt)
                scores.append(score)
                evaluations.append({
                    'input': input_text,
                    'output': raw_output,
                    'target': target,
                    'score': score,
                    'feedback': feedback
                })

        avg_score = np.mean(scores) if scores else 0.0

        return avg_score, evaluations

    async def evaluate_prompt_with_feedback(
        self,
        prompt: str,
        batch: list[tuple[str, str]]
    ) -> list[dict]:
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
        all_evaluations: list[dict],
        num_examples: int = 5,
    ) -> list[dict]:
        if not all_evaluations:
            return []
        batch_evals = all_evaluations[-num_examples:]

        return batch_evals #[ev for ev in batch_evals if ev['score'] == 0]

    # def sample_examples(
    #     self,
    #     all_evaluations: List[Dict],
    #     num_examples: int = 5,
    # ) -> List[Dict]:
    #     """
    #     Stratified sampling strategy: balance recent examples with historical worst.
    #     This is specific to prompt optimization - other tasks might sample differently.

    #     Args:
    #         all_evaluations: All evaluation results
    #         num_examples: Total number of examples to sample

    #     Returns:
    #         List of sampled examples
    #     """
    #     if not all_evaluations:
    #         return []

    #     # Calculate split between recent and historical
    #     num_recent = num_examples
    #     num_historical = cfg.optimizer.sample_examples.num_historical

    #     # Separate recent batch from historical
    #     recent_size = min(self.minibatch_size_train, len(all_evaluations))
    #     recent_evals = all_evaluations[-recent_size:] if len(all_evaluations) >= recent_size else all_evaluations
    #     historical_evals = all_evaluations[:-recent_size] if len(all_evaluations) > recent_size else []

    #     # Get worst historical examples (sorted by score)
    #     if historical_evals:
    #         historical_sorted = sorted(historical_evals, key=lambda x: x['score'])
    #         k_historical = min(10, len(historical_sorted))
    #         historical_worst = historical_sorted[:k_historical]
    #     else:
    #         historical_worst = []

    #     # Track selected examples by identity to prevent duplicates
    #     selected_examples = []
    #     selected_ids = set()

    #     # Sample from recent
    #     if recent_evals:
    #         n_recent = min(num_recent, len(recent_evals))
    #         sampled_recent = random.sample(recent_evals, n_recent)
    #         for item in sampled_recent:
    #             item_id = id(item)
    #             if item_id not in selected_ids:
    #                 selected_examples.append(item)
    #                 selected_ids.add(item_id)

    #     # Sample from historical worst (skip duplicates)
    #     if historical_worst and num_historical > 0:
    #         available_historical = [item for item in historical_worst if id(item) not in selected_ids]

    #         if available_historical:
    #             n_historical = min(num_historical, len(available_historical))
    #             sampled_historical = random.sample(available_historical, n_historical)
    #             for item in sampled_historical:
    #                 selected_examples.append(item)
    #                 selected_ids.add(id(item))

    #             # Fill remaining from recent if needed
    #             shortfall = num_historical - len(sampled_historical)
    #             if shortfall > 0 and recent_evals:
    #                 remaining_recent = [item for item in recent_evals if id(item) not in selected_ids]
    #                 if remaining_recent:
    #                     n_fill = min(shortfall, len(remaining_recent))
    #                     fill_samples = random.sample(remaining_recent, n_fill)
    #                     for item in fill_samples:
    #                         selected_examples.append(item)
    #                         selected_ids.add(id(item))

    #     return selected_examples if selected_examples else all_evaluations[-num_recent:]


class PromptAgentWorldModel(PromptOptimizationWorldModel):
    """
    World model for prompt agents that do not require feedback-based sampling.
    Uses random sampling from all evaluations.
    """
    def get_fixed_eval_batch(self, depth):
        return self.eval_dataset  # Always return full dataset

    async def evaluate_prompt(
        self,
        prompt: str,
        dataset: list[tuple[str, str]]
    ) -> tuple[float, list[dict]]:
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
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i + batch_size]
            inputs = [input_text for input_text, _ in batch]
            full_prompts = self.build_prompts(inputs, prompt)
            messages_batch = [[HumanMessage(content=p)] for p in full_prompts]

            # Batch LLM inference
            responses = await self.llm.abatch(messages_batch)

            # Calculate scores and collect feedback
            for response, (input_text, target) in zip(responses, batch):
                raw_output = response.content
                score = self.calculate_reward(raw_output, target, prompt)
                scores.append(score)
                evaluations.append({
                    'input': input_text,
                    'output': raw_output,
                    'target': target,
                    'score': score
                })

        avg_score = np.mean(scores) if scores else 0.0

        return avg_score, evaluations

    def sample_examples(
        self,
        all_evaluations: list[dict],
        num_examples: int = 5,
    ) -> list[dict]:
        if not all_evaluations:
            return []
        batch_evals = all_evaluations[-num_examples:]

        return batch_evals #[ev for ev in batch_evals if ev['score'] == 0]

    def build_prompts(self, inputs: list[str], prompt: str) -> list[str]:
        """Build full prompts by combining inputs with instruction prompt"""
        output_format = "At the end show the answer option bracketed between <answer> and </answer>."
        if self.post_instruction:
            return [f'{input_text}\n{prompt}\n{output_format}' for input_text in inputs]
        else:
            return [f'{prompt}\n{input_text}\n{output_format}' for input_text in inputs]
