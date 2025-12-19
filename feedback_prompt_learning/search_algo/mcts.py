"""
MCTS Prompt Optimizer with Feedback-Driven Actions
"""
import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from hydra.utils import instantiate as hydra_instantiate
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from tqdm import tqdm

from feedback_prompt_learning import cfg
from feedback_prompt_learning.data import EvaluationResult
from feedback_prompt_learning.search_algo.world_model import WorldModel
from feedback_prompt_learning.utils.similarity import jaccard_ngram

logger = logging.getLogger(__name__)


# Format examples using templates from config
def format_examples(total: int, examples: list[EvaluationResult], include_feedback: bool) -> str:
    """Format evaluation results for display in prompts"""
    header = cfg.optimizer.example_format.header.format(
        total=total,
        shown=len(examples)
    )

    # Check if any example has feedback
    has_any_feedback = include_feedback and any(ex.feedback is not None for ex in examples)
    template = (cfg.optimizer.example_format.with_feedback if has_any_feedback
                else cfg.optimizer.example_format.without_feedback)

    examples_text = header
    last_prompt_feedback = None

    for i, result in enumerate(examples, 1):
        feedback = result.feedback if has_any_feedback else None
        examples_text += template.format(
            index=i,
            score=result.score,
            input=result.input,
            output=result.output,
            target=result.target,
            accuracy_feedback=feedback.accuracy_feedback if feedback else None,
            reasoning_feedback=feedback.reasoning_feedback if feedback else None,
        )
        # Track last non-None prompt feedback
        if feedback and feedback.has_prompt_feedback():
            last_prompt_feedback = feedback.prompt_feedback

    # Add overall prompt feedback if available
    if last_prompt_feedback:
        examples_text += f"\nOverall Prompt Feedback: {last_prompt_feedback}\n"

    return examples_text# ============================================================================
# DATA DEFINITIONS
# ============================================================================

@dataclass
class MCTSNode:
    """Node in MCTS tree for prompt optimization"""
    prompt: str
    parent: Optional['MCTSNode'] = None
    children: list['MCTSNode'] = field(default_factory=list)
    cum_rewards: list[float] = field(default_factory=list)
    reward: float = 0.0
    visited: int = 0
    _is_terminal: bool = False
    # Track all evaluations for better error analysis
    all_evaluations: list[EvaluationResult] = field(default_factory=list)  # Stores evaluation results with structured feedback

    @property
    def N(self) -> int:
        """Visit count"""
        return self.visited

    @property
    def Q(self) -> float:
        """Average cumulative reward (Q-value for UCT)"""
        return np.mean(self.cum_rewards) if self.cum_rewards else 0.0

    @property
    def depth(self) -> int:
        """Depth in tree"""
        d = 0
        current = self.parent
        while current is not None:
            d += 1
            current = current.parent
        return d

    def is_terminal(self, max_depth: int) -> bool:
        """Check if node is terminal"""
        return self.depth >= max_depth or self._is_terminal


# ============================================================================
# MCTS OPTIMIZER WITH FEEDBACK
# ============================================================================

class MCTSPromptOptimizerFeedback:
    """MCTS for Prompt Optimization using LLM-generated actions based on feedback"""

    def __init__(
        self,
        initial_prompt: str,
        world_model: WorldModel,
        llm_action: ChatOpenAI = hydra_instantiate(cfg.optimizer.llm.action),
        llm_critic: ChatOpenAI = hydra_instantiate(cfg.optimizer.llm.critic),
        num_iterations: int = cfg.optimizer.num_iterations,
        exploration_constant: float = cfg.optimizer.exploration_constant,
        max_depth: int = cfg.optimizer.max_depth,
        expand_width: int = cfg.optimizer.expand_width,  # Number of gradient analyses per expansion
        num_samples: int = cfg.optimizer.num_samples,   # Number of prompts generated per gradient
        log_freq: int = 2,
    ):

        self.initial_prompt = initial_prompt
        self.world_model = world_model
        self.llm_action = llm_action
        self.llm_critic = llm_critic
        self.num_iterations = num_iterations
        self.c = exploration_constant
        self.max_depth = max_depth
        self.expand_width = expand_width
        self.num_samples = num_samples
        self.log_freq = log_freq
        self.root = MCTSNode(prompt=initial_prompt)

        # Threshold tracking for early stopping
        self.mcts_threshold = 0.0
        self.min_threshold = 0.0
        self.min_depth = 2

        # Track all nodes and iteration paths
        self.all_nodes = [self.root]
        self.trace_in_each_iter = []

        # Initialize logger
        self.logger = logging.getLogger(__name__)

        # Log dataset info
        dataset_info = world_model.get_dataset_info()
        self.logger.info(f"Feedback-driven MCTS initialized")
        self.logger.info(f"Train Dataset: {dataset_info['train_size']} examples, {dataset_info['train_batches']} batches of ~{dataset_info['minibatch_size_train']}")
        if dataset_info['eval_size'] > 0:
            self.logger.info(f"Eval Dataset: {dataset_info['eval_size']} examples, {dataset_info['eval_batches']} batches of ~{dataset_info['minibatch_size_eval']}")

    async def run(self) -> MCTSNode:
        """Main MCTS algorithm"""
        self.logger.info(f"\nStarting MCTS with {self.num_iterations} iterations...")
        self.logger.info(f"Initial prompt: {self.initial_prompt}\n")

        # Initialize root - evaluate on eval batch for unbiased reward
        # Root is at depth 0, get eval batch for it
        eval_batch = self.world_model.get_fixed_eval_batch(0)
        self.root.reward, eval_evaluations = await self.world_model.evaluate_prompt(self.root.prompt, eval_batch, eval=True)

        # Also get train batch for gradient analysis
        train_batch = self.world_model.get_batch('train')
        _, train_evaluations = await self.world_model.evaluate_prompt(self.root.prompt, train_batch)
        self.root.all_evaluations.extend(train_evaluations)


        self.logger.info(f"Root initial reward: {self.root.reward:.4f}\n")

        # Initialize thresholds
        if self.min_threshold == 0:
            self.min_threshold = self.root.reward
            self.mcts_threshold = self.root.reward

        for iteration in tqdm(range(self.num_iterations), desc="MCTS Iterations", ncols=100, file=None, mininterval=0.1):
            # Selection
            path = self._select(self.root)

            # Expansion (fetches batches internally)
            if not path[-1].is_terminal(self.max_depth):
                await self._expand(path[-1])

                # Simulation (fetches batches internally)
                await self._simulate(path)

            # Backpropagation
            self._backpropagation(path)

            # Track path for this iteration
            self.trace_in_each_iter.append(path.copy())

            # Logging
            if (iteration + 1) % self.log_freq == 0:
                best = self._best_child_of(self.root)
                tqdm.write(f"Iter {iteration + 1} | Best Q: {best.Q:.4f} | Root Q: {self.root.Q:.4f}")

        self.logger.info("\n" + "="*80)
        self.logger.info("MCTS COMPLETE")
        self.logger.info("="*80)

        return self._best_child_of(self.root)

    def _uct(self, node: MCTSNode) -> float:
        """Calculate UCT value"""
        N_parent = node.parent.N if node.parent else 0
        return node.Q + self.c * np.sqrt(np.log(N_parent + 1) / max(1, node.N))

    def _select(self, node: MCTSNode) -> list[MCTSNode]:
        """Selection phase"""
        path = []
        while True:
            path.append(node)
            node.visited += 1

            if len(node.children) == 0 or node.is_terminal(self.max_depth):
                return path

            node = max(node.children, key=self._uct)

        return path

    def increase_threshold(self, threshold: float):
        """Update the global max threshold if a better reward is found"""
        if threshold > self.mcts_threshold:
            self.mcts_threshold = threshold

    def early_stop(self, node: MCTSNode) -> bool:
        """Check if node is good enough to stop simulation early"""
        return node.reward > self.mcts_threshold and node.depth > self.min_depth

    def _is_terminal_with_min_threshold(self, node: MCTSNode) -> bool:
        """Check if node reward is too low to continue exploring"""
        if node.parent is None:
            min_threshold = self.min_threshold
        else:
            min_threshold = (self.min_threshold + node.parent.reward) / 2
        return node.reward < min_threshold and node.depth > self.min_depth

    def _get_trajectory_prompts(self, node: MCTSNode) -> list[str]:
        """Collect the trajectory of prompts from root to given node"""
        trajectory_prompts = []
        temp_node = node
        while temp_node is not None:
            trajectory_prompts.append(temp_node.prompt)
            temp_node = temp_node.parent
        return trajectory_prompts[::-1]  # Reverse to get root -> node order


    async def _generate_gradient_and_prompts(
        self,
        node: MCTSNode,
        minibatch: list[tuple[str, str]],
        trajectory_prompts: list[str]
    ) -> list[str]:
        """Generate gradient analysis and new prompts for one expansion width"""
        # Evaluate current prompt and collect feedback
        batch_feedback = await self.world_model.evaluate_prompt_with_feedback(node.prompt, minibatch)

        # Store evaluations in parent node for tracking
        node.all_evaluations.extend(batch_feedback)
        # TODO: Consider refactoring this
        newline = '\n'
        self.logger.debug(f"""Feedback:
        {''.join([res.feedback.reasoning_feedback + newline for res in batch_feedback if res.feedback and res.feedback.reasoning_feedback])}
        """)
        # Ask LLM to analyze errors and generate gradient (improvement direction)
        gradient, sampled_example_string = await self._get_action_decisions(
            node.prompt, node.all_evaluations
        )
        self.logger.debug(f"Generated gradient: {gradient}")
        assert len(sampled_example_string) > 0, "Example string should not be empty"

        # Generate num_samples prompts from this gradient
        new_prompts = await self._generate_new_prompts(
            node.prompt,
            gradient,
            trajectory_prompts,
            self.num_samples,
            sampled_example_string
        )
        self.logger.debug(f"Generated new prompts: {new_prompts}")
        return new_prompts

    async def _expand(self, node: MCTSNode) -> list[MCTSNode]:
        """Expansion phase - generate new prompts using LLM with trajectory context"""
        if node.is_terminal(self.max_depth):
            return []

        expand_start = time.time()

        # Get trajectory from root to current node
        trajectory_prompts = self._get_trajectory_prompts(node)

        # OPTIMIZATION 1: Parallelize gradient generation across expand_width
        # Each width generates num_samples prompts
        gradient_tasks = []
        minibatches = []  # Store minibatches for later use
        for width_idx in range(self.expand_width):
            minibatch = self.world_model.get_batch('train')
            minibatches.append(minibatch)
            task = self._generate_gradient_and_prompts(node, minibatch, trajectory_prompts)
            gradient_tasks.append(task)

        # Wait for all gradient analyses to complete in parallel
        all_new_prompts_by_width = await asyncio.gather(*gradient_tasks)

        # Flatten the list of lists into a single list of all new prompts
        all_new_prompts = []
        for prompts in all_new_prompts_by_width:
            all_new_prompts.extend(prompts)

        # OPTIMIZATION 2: Parallelize all child evaluations on eval batch
        # Only evaluate on eval dataset for reward signal (no redundant train evals)
        child_depth = node.depth + 1
        eval_batch = self.world_model.get_fixed_eval_batch(child_depth)

        # Evaluate all new prompts in parallel
        eval_tasks = [self.world_model.evaluate_prompt(prompt, eval_batch, eval=True) for prompt in all_new_prompts]
        eval_results = await asyncio.gather(*eval_tasks)

        # Create child nodes with eval results
        all_children = []
        for new_prompt, (reward, eval_evaluations) in zip(all_new_prompts, eval_results):
            child = MCTSNode(
                prompt=new_prompt,
                parent=node,
                reward=reward  # Reward from eval batch (unbiased)
            )
            # Note: We don't store train evaluations here anymore to avoid redundant API calls
            # Train evaluations will be collected when this node is expanded
            all_children.append(child)
            self.all_nodes.append(child)

        node.children.extend(all_children)

        expand_time = time.time() - expand_start
        self.logger.debug(f"Expansion took {expand_time:.2f}s for {len(all_children)} children")

        return all_children

    async def _simulate(self, path: list[MCTSNode]):
        """Simulation phase with early stopping"""
        node = path[-1]

        while True:
            # Early stop if we found a very good node
            if self.early_stop(node):
                node._is_terminal = True
                self.increase_threshold(node.reward)
                break

            # Update global threshold
            self.increase_threshold(node.reward)

            # Check if terminal (depth limit or low reward)
            if node.is_terminal(self.max_depth) or self._is_terminal_with_min_threshold(node):
                break

            # Expand if no children
            if len(node.children) == 0:
                await self._expand(node)  # _expand fetches batch internally

            # Break if still no children after expansion
            if len(node.children) == 0:
                node._is_terminal = True
                break

            # Greedy selection - choose best child by reward
            node = max(node.children, key=lambda c: c.reward)
            node.visited += 1
            path.append(node)

    def _backpropagation(self, path: list[MCTSNode]):
        """Backpropagation phase"""
        rewards = []
        for node in reversed(path):
            rewards.append(node.reward)
            cum_reward = np.sum(rewards[::-1])
            node.cum_rewards.append(cum_reward)



    async def _get_action_decisions(
        self,
        prompt: str,
        all_evaluations: list[EvaluationResult],
    ) -> tuple[str, str]:
        """
        Ask LLM to analyze feedback and generate improvement gradient.
        """
        # TODO: consider adding gradient ascend analysis when average score is high
        return await self._get_descend_gradient(prompt, all_evaluations)

    async def _get_descend_gradient(
        self,
        prompt: str,
        selected_evaluations: list[EvaluationResult]
    ) -> tuple[str, str]:
        """Analyze errors to generate improvement gradient (original behavior)"""

        # Use WorldModel's sampling strategy (delegates to specific implementation)
        dataset_info = self.world_model.get_dataset_info()
        num_examples = dataset_info['minibatch_size_train']
        sampled_examples = self.world_model.sample_examples(
            all_evaluations=selected_evaluations,
            num_examples=num_examples,
        )

        # Calculate average score across all evaluations
        avg_score = np.mean([result.score for result in selected_evaluations]) if selected_evaluations else 0.0

        example_string_with_feedback = format_examples(total=len(selected_evaluations), examples=sampled_examples, include_feedback=True)
        example_string_without_feedback = format_examples(total=len(selected_evaluations), examples=sampled_examples, include_feedback=False)

        # Gradient analysis prompt (descend - focus on errors)
        gradient_prompt = cfg.optimizer.gradient_analysis_prompt.format(
            prompt=prompt,
            example_string_with_feedback=example_string_with_feedback,
            avg_score=avg_score
        ).strip()

        # Log which prompt template is being used (first 150 chars to verify config)
        logger.debug(f"[MCTS Runtime] Using gradient analysis prompt template: {cfg.optimizer.gradient_analysis_prompt[:150]}...")

        response = await self.llm_action.ainvoke([HumanMessage(content=gradient_prompt)])
        return response.content.strip(), example_string_without_feedback

    async def _generate_new_prompts(
        self,
        current_prompt: str,
        gradient: str,
        trajectory_prompts: list[str],
        num_prompts: int,
        example_string: str = ""
    ) -> list[str]:
        """Generate new prompt based on gradient analysis and trajectory"""

        # Format trajectory for context
        trajectory_text = ""
        for i, traj_prompt in enumerate(trajectory_prompts):
            trajectory_text += f"\n{i+1}. {traj_prompt}\n"

        optimize_prompt = cfg.optimizer.prompt_generation_prompt.format(
            current_prompt=current_prompt,
            example_string=example_string,
            gradient=gradient,
            trajectory_text=trajectory_text,
            num_prompts=num_prompts,
            plural="s" if num_prompts > 1 else "",
            plural_verb="s are" if num_prompts > 1 else " is"
        ).strip()

        # Log which prompt template is being used (first 150 chars to verify config)
        logger.debug(f"[MCTS Runtime] Using prompt generation template: {cfg.optimizer.prompt_generation_prompt[:150]}...")

        response = await self.llm_critic.ainvoke([HumanMessage(content=optimize_prompt)])
        improved_prompts_text = response.content.strip()

        # Extract all prompts between <START> and <END> tags
        import re
        matches = re.findall(r'<START>\s*(.+?)\s*<END>', improved_prompts_text, re.DOTALL)

        if matches:
            improved_prompts = [match.strip() for match in matches[:num_prompts]]
        else:
            # Fallback: if no tags found, return current prompt
            improved_prompts = [current_prompt]

        # Ensure we return exactly num_prompts (pad with current if needed)
        while len(improved_prompts) < num_prompts:
            improved_prompts.append(current_prompt)

        return improved_prompts[:num_prompts]

    def _best_child_of(self, node: MCTSNode) -> MCTSNode:
        """Return best child by Q value"""
        if not node.children:
            return node
        return max(node.children, key=lambda c: c.Q)

    def get_best_prompt(self) -> tuple[MCTSNode, str]:
        """Get the best prompt found"""
        best_node = self.root
        best_q = self.root.Q

        def search_tree(node: MCTSNode):
            nonlocal best_node, best_q
            if node.N > 0 and node.Q > best_q:
                best_q = node.Q
                best_node = node
            for child in node.children:
                search_tree(child)

        search_tree(self.root)

        return best_node, best_node.prompt

    def get_best_path_with_rewards(self) -> tuple[list[MCTSNode], list[float]]:
        """
        Get the best path from root to a leaf by following highest Q-value children

        Returns:
            Tuple of (path, rewards) where path is list of nodes and rewards is their Q-values
        """
        path = []
        rewards = []
        current = self.root

        while current is not None:
            path.append(current)
            rewards.append(current.Q)

            if len(current.children) == 0:
                break

            # Follow child with highest Q-value
            current = max(current.children, key=lambda c: c.Q)

        return path, rewards

    def get_best_prompt_in_best_path(self) -> tuple[MCTSNode, list[MCTSNode]]:
        """
        Find the node with highest Q-value along the best path (greedy path following highest Q children)

        Returns:
            Tuple of (best_node, path) where best_node has highest Q in the path
        """
        path, rewards = self.get_best_path_with_rewards()

        if not path:
            return self.root, [self.root]

        # Find node with maximum Q-value in the path
        best_idx = np.argmax(rewards)
        best_node = path[best_idx]

        return best_node, path

    def get_best_prompts_comprehensive(self) -> dict:
        """
        Get best prompts using multiple strategies for comparison
        Returns dictionary with different selection strategies
        """
        results = {}

        # Strategy 1: Best child of root (most immediate improvement)
        best_child = self._best_child_of(self.root)
        results['best_child_of_root'] = {
            'node': best_child,
            'prompt': best_child.prompt,
            'q_value': best_child.Q,
            'reward': best_child.reward,
            'visits': best_child.N,
            'depth': 1,
            'strategy': 'Highest Q-value among root\'s children (immediate improvement)'
        }

        # Strategy 2: Best path (greedy traversal)
        best_path_nodes, best_path_rewards = self.get_best_path_with_rewards()
        best_path_terminal = best_path_nodes[-1]
        results['best_path_terminal'] = {
            'node': best_path_terminal,
            'prompt': best_path_terminal.prompt,
            'q_value': best_path_terminal.Q,
            'reward': best_path_terminal.reward,
            'visits': best_path_terminal.N,
            'depth': len(best_path_nodes) - 1,
            'path': best_path_nodes,
            'path_rewards': best_path_rewards,
            'strategy': 'Terminal node of best path (greedy Q-value traversal)'
        }

        # Strategy 3: Best node in best path (highest Q in path)
        best_in_path, full_path = self.get_best_prompt_in_best_path()
        best_in_path_path = []
        current = best_in_path
        while current is not None:
            best_in_path_path.append(current)
            current = current.parent
        best_in_path_path.reverse()

        results['best_in_best_path'] = {
            'node': best_in_path,
            'prompt': best_in_path.prompt,
            'q_value': best_in_path.Q,
            'reward': best_in_path.reward,
            'visits': best_in_path.N,
            'depth': len(best_in_path_path) - 1,
            'path': best_in_path_path,
            'full_search_path': full_path,
            'strategy': 'Best node in best path by Q-value (recommended)'
        }

        return results

    def prepare_output(self) -> dict:
        """Prepare comprehensive output matching original MCTS format"""
        self.logger.info("\n" + "="*80)
        self.logger.info("PREPARING OUTPUT - ANALYZING ALL PATHS")
        self.logger.info("="*80)

        paths_nodes = []
        paths_qs = []
        paths_rewards = []
        paths_ucts = []

        # Analyze each iteration path
        for i, path in enumerate(self.trace_in_each_iter):
            path_qs = []
            path_rewards = []
            path_ucts = []

            for node in path:
                uct = self._uct(node)
                path_ucts.append(uct)
                path_qs.append(node.Q)
                path_rewards.append(node.reward)

            paths_nodes.append(path)
            paths_qs.append(path_qs)
            paths_rewards.append(path_rewards)
            paths_ucts.append(path_ucts)

            if (i + 1) % 5 == 0 or i < 3:  # Log first 3 and every 5th
                self.logger.info(f"\nPath {i}: depth={len(path)-1}")
                self.logger.info(f"  Mean UCT: {np.mean(path_ucts):.4f} | Mean Q: {np.mean(path_qs):.4f} | Mean Reward: {np.mean(path_rewards):.4f}")

        # =====================
        # Depth-Aligned Lexical Similarity Analysis (Jaccard bigram)
        # =====================

        # Only consider paths with more than 1 node
        all_path_prompts = [[node.prompt for node in path] for path in paths_nodes if len(path) > 1]
        if all_path_prompts:
            # Find minimum common depth
            min_depth = min(len(prompts) for prompts in all_path_prompts)
            # Truncate all paths to min_depth
            truncated_paths = [prompts[:min_depth] for prompts in all_path_prompts]
            # Intra-path: mean similarity between consecutive prompts in each path
            intra_sims = []
            for prompts in truncated_paths:
                sims = [jaccard_ngram(prompts[i], prompts[i+1], n=2) for i in range(min_depth-1)]
                intra_sims.extend(sims)
            # Inter-path: mean similarity between prompts at the same depth across all paths
            inter_sims = []
            for d in range(min_depth):
                depth_prompts = [prompts[d] for prompts in truncated_paths]
                for i in range(len(depth_prompts)):
                    for j in range(i+1, len(depth_prompts)):
                        inter_sims.append(jaccard_ngram(depth_prompts[i], depth_prompts[j], n=2))
            mean_intra = np.mean(intra_sims) if intra_sims else 1.0
            mean_inter = np.mean(inter_sims) if inter_sims else 1.0
            self.logger.info(f"\nDepth-aligned lexical n-gram (bigram) similarity:")
            self.logger.info(f"  Mean intra-path (consecutive, aligned) similarity: {mean_intra:.3f}")
            self.logger.info(f"  Mean inter-path (same depth, aligned) similarity: {mean_inter:.3f}")

        # Rank paths by mean Q and mean reward
        qs_rank = np.argsort([np.mean(row) for row in paths_qs])[::-1].tolist()
        rewards_rank = np.argsort([np.mean(row) for row in paths_rewards])[::-1].tolist()

        best_q_path = paths_nodes[qs_rank[0]] if qs_rank else [self.root]
        best_reward_path = paths_nodes[rewards_rank[0]] if rewards_rank else [self.root]

        # Get top-k nodes by reward
        top_k_reward_nodes = sorted(self.all_nodes, key=lambda node: node.reward, reverse=True)[:min(5, len(self.all_nodes))]

        # Select best node from best_reward_path (highest reward)
        selected_node = sorted(best_reward_path, key=lambda node: node.reward, reverse=True)[0]

        self.logger.info("\n" + "="*80)
        self.logger.info("PATH ANALYSIS SUMMARY")
        self.logger.info("="*80)
        self.logger.info(f"Total iterations: {len(self.trace_in_each_iter)}")
        self.logger.info(f"Total nodes created: {len(self.all_nodes)}")
        self.logger.info(f"\nBest Q path (iteration {qs_rank[0]}):")
        self.logger.info(f"  Depth: {len(best_q_path)-1}")
        self.logger.info(f"  Mean Q: {np.mean([n.Q for n in best_q_path]):.4f}")
        self.logger.info(f"  Mean Reward: {np.mean([n.reward for n in best_q_path]):.4f}")
        self.logger.info(f"\nBest Reward path (iteration {rewards_rank[0]}):")
        self.logger.info(f"  Depth: {len(best_reward_path)-1}")
        self.logger.info(f"  Mean Q: {np.mean([n.Q for n in best_reward_path]):.4f}")
        self.logger.info(f"  Mean Reward: {np.mean([n.reward for n in best_reward_path]):.4f}")
        self.logger.info(f"\nSelected node (highest reward in best_reward_path):")
        self.logger.info(f"  Reward: {selected_node.reward:.4f}")
        self.logger.info(f"  Q-value: {selected_node.Q:.4f}")
        self.logger.info(f"  Visits: {selected_node.N}")
        self.logger.info(f"  Depth: {selected_node.depth}")
        self.logger.info(f"\nTop-5 nodes by reward:")
        for i, node in enumerate(top_k_reward_nodes, 1):
            self.logger.info(f"  {i}. Reward: {node.reward:.4f}, Q: {node.Q:.4f}, Visits: {node.N}, Depth: {node.depth}")

        return {
            'all_paths': paths_nodes,
            'all_nodes': self.all_nodes,
            'best_q_path': best_q_path,
            'best_reward_path': best_reward_path,
            'top_k_reward_nodes': top_k_reward_nodes,
            'best_reward_path_last_node': [best_reward_path[-1]],
            'best_reward_path_selected_node': [selected_node],
        }
