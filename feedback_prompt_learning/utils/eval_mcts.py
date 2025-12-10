"""
Unified MCTS Evaluation Script for BigBench Tasks
Supports both PromptAgent and Feedback-MCTS algorithms with easy method switching
"""

import argparse
import asyncio
import logging
import os
from pathlib import Path
from typing import Any, List, Tuple

import nltk
import numpy as np
import tiktoken
import yaml
from hydra.utils import instantiate as hydra_instantiate
from langchain_core.messages import AIMessage, HumanMessage
from nltk.corpus import stopwords
from omegaconf import OmegaConf
from tqdm import tqdm

# Don't import cfg at module level to avoid Hydra initialization
# from feedback_prompt_learning import cfg
from feedback_prompt_learning.data import Feedback, RewardOutput
from feedback_prompt_learning.data.task.bigbench import CustomTask
from feedback_prompt_learning.search_algo.mcts import \
    MCTSPromptOptimizerFeedback
from feedback_prompt_learning.search_algo.world_model import (
    PromptAgentWorldModel, PromptOptimizationWorldModel)

# Configure logger
logger = logging.getLogger(__name__)

# Download stopwords if not already present
try:
    STOP_WORDS = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords', quiet=True)
    STOP_WORDS = set(stopwords.words('english'))


# ============================================================================
# DATASET CONFIGURATION LOADER
# ============================================================================

def load_dataset_config(dataset_name: str, config_path: str = None) -> dict:
    """Load dataset configuration from YAML file"""
    if config_path is None:
        # Default path relative to project root
        project_root = Path(__file__).parent.parent.parent
        config_path = project_root / "configs" / "datasets.yaml"

    with open(config_path) as f:
        all_configs = yaml.safe_load(f)

    if dataset_name not in all_configs['datasets']:
        available = list(all_configs['datasets'].keys())
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {available}")

    config = all_configs['datasets'][dataset_name]

    # Resolve data_path to absolute path
    project_root = Path(__file__).parent.parent.parent
    config['data_dir'] = str(project_root / config['data_path'])

    return config


# ============================================================================
# VERBOSITY EVALUATION (DO NOT MODIFY STRINGS)
# ============================================================================

def count_tokens_gpt4o_mini(text):
    """
    Calculates the number of tokens in a given text using the
    encoding for GPT-4o Mini.
    """
    # GPT-4o Mini uses the 'cl100k_base' encoding
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    return len(tokens)


def evaluate_prompt_verbosity(prompt, min_tokens=200, max_tokens=420,
                             min_content_tokens=100, max_content_tokens=380, language='english'):
    """
    Evaluates if a prompt is overly verbose or overly brief using token count and NLTK stopwords.
    Defaults are optimized for Big Bench Hard tasks which require detailed, complex prompts.

    Args:
        prompt (str): The prompt text to evaluate
        min_tokens (int): Minimum token count threshold (default: 15 for the task)
        max_tokens (int): Maximum token count threshold (default: 350 for the task)
        min_content_tokens (int): Minimum content tokens after removing stop words (default: 10 for the task)
        max_content_tokens (int): Maximum content tokens after removing stop words (default: 250 for the task)
        language (str): Language for stopwords (default: 'english')

    Returns:
        dict: Evaluation results with metrics and status

    Note:
        Big Bench Hard tasks typically require:
        - Clear problem statement with examples
        - Specific instructions for reasoning steps
        - Output format specifications
        - Context about the task domain
    """
    # Get stopwords for the specified language
    try:
        stop_words_set = set(stopwords.words(language))
    except:
        stop_words_set = STOP_WORDS  # Fallback to English

    # Calculate basic metrics
    words = prompt.split()
    token_count = count_tokens_gpt4o_mini(prompt)
    sentence_count = prompt.count('.') + prompt.count('!') + prompt.count('?')

    # Remove stop words and calculate content text
    content_words = [word for word in words
                     if word.lower().strip('.,!?;:()[]{}"\'"') not in stop_words_set and word.strip()]
    content_text = ' '.join(content_words)
    content_token_count = count_tokens_gpt4o_mini(content_text)
    stop_token_count = token_count - content_token_count
    content_ratio = content_token_count / token_count if token_count > 0 else 0

    # Determine status based on both total tokens and content tokens
    is_too_brief = (token_count < min_tokens or content_token_count < min_content_tokens)
    is_too_verbose = (token_count > max_tokens or content_token_count > max_content_tokens)

    if is_too_brief:
        status = "TOO_BRIEF"
        if content_token_count < min_content_tokens:
            recommendation = f"Prompt lacks meaningful content for the task tasks. Add problem context and clear instructions to improve clarity and guidance."
        else:
            recommendation = f"Prompt is too short for the task complexity. Consider breaking down the problem and adding more detailed instructions or structure (including reasoning steps, rules or examples)."
    elif is_too_verbose:
        status = "TOO_VERBOSE"
        if content_token_count > max_content_tokens:
            recommendation = f"Prompt has excessive content. Focus on essential instructions and distill knowledge into a shorter prompt. Current: {content_token_count} content tokens, target: under {max_content_tokens}."
        else:
            recommendation = f"Prompt is too long. Remove redundant info and distill knowledge into a shorter prompt. Current: {token_count} tokens, target: under {max_tokens} tokens."
    else:
        status = "OPTIMAL"
        if content_ratio < 0.35:
            recommendation = f"Prompt length is appropriate for the task, but has many filler words ({content_ratio*100:.1f}% actual content). Make instructions more direct and technical."
        elif content_ratio >= 0.60:
            recommendation = f"Excellent content density for the task tasks ({content_ratio*100:.1f}% actual content). Good balance of detail and clarity!"
        else:
            recommendation = f"Prompt length and content balance are appropriate ({content_ratio*100:.1f}% actual content) for the task. You may try to further increase content density by reducing filler words and focusing on details and steps."

    return {
        "status": status,
        "is_too_brief": is_too_brief,
        "is_too_verbose": is_too_verbose,
        "metrics": {
            "token_count": token_count,
            "content_token_count": content_token_count,
            "stop_token_count": stop_token_count,
            "content_ratio": round(content_ratio, 3),
            "sentence_count": sentence_count,
        },
        "thresholds": {
            "min_tokens": min_tokens,
            "max_tokens": max_tokens,
            "min_content_tokens": min_content_tokens,
            "max_content_tokens": max_content_tokens,
        },
        "recommendation": recommendation,
    }


# ============================================================================
# DATA PREPARATION
# ============================================================================

def create_dataset(task: CustomTask) -> tuple[list[tuple[str, str]], list[tuple[str, str]], list[tuple[str, str]]]:
    """Convert CustomTask dataset to (input, target) format"""
    def format_examples(dataset):
        formatted = []
        for example in dataset:
            formatted.append((example['question'], example['answer']))
        return formatted

    train_data = format_examples(task.dataset["train"])
    eval_data = format_examples(task.dataset["eval"])
    test_data = format_examples(task.dataset["test"]) if "test" in task.dataset else []

    return train_data, eval_data, test_data


def create_reward_function(train_data: list[tuple[str, str]], use_feedback: bool = True) -> Any:
    """
    Create reward function for the task

    Args:
        train_data: Training examples
        use_feedback: If True, return detailed feedback; if False, simple binary reward
    """
    unique_targets = {target.strip().upper() for _, target in train_data}

    if not use_feedback:
        # PromptAgent-style: simple binary reward
        def reward_fn(output: str, target: str, prompt: str, raw_output: str = "") -> RewardOutput:
            output_clean = output.strip().upper()
            target_clean = target.strip().upper()

            score = 1.0 if output_clean == target_clean else 0.0
            feedback = Feedback(accuracy_feedback="Correct" if score == 1.0 else "Incorrect")

            return RewardOutput(score=score, feedback=feedback)
    else:
        # Feedback-MCTS: detailed feedback
        def reward_fn(output: str, target: str, prompt: str, raw_output: str = "") -> RewardOutput:
            """
            Enhanced reward function with TF-IDF keyword analysis.
            Analyzes output after clean_response() processing.
            Checks if model's raw response mentions discriminative keywords identified via TF-IDF.
            Provides feedback on correctness and reasoning quality.
            """
            output_clean = output.strip().upper()
            target_clean = target.strip().upper()

            # 1. Check for exact match (correct answer)
            if output_clean == target_clean:
                accuracy = 1.0
                accuracy_feedback = "Correct answer ✓"
            else:
                accuracy = 0.0

                # Diagnose why the answer is wrong
                issues = []

                # Check if clean_response returned a format error
                # Check if any valid target value was extracted from output
                has_valid_answer = any(possible_target in output_clean for possible_target in unique_targets)

                if not has_valid_answer:
                    issues.append(f"Format error: clean_response() could not extract answer, possible answers are {', '.join(unique_targets)}. Model's output must either: (1) use <answer>X</answer> tags with letter inside, OR (2) contain a clear option letter (A-E) or (A)-(E) format, OR (3) match one of the option text strings. Prompt should specify: 'Provide your reasoning, then output your final answer as <answer>X</answer> .'")
                else:
                    # clean_response successfully extracted something, but it's wrong
                    wrong_answer = output_clean
                    issues.append(f"Wrong answer extracted: '{wrong_answer}' instead of '{target_clean}'. Identify what went wrong in current reasoning and propose critical domain knowledge to improve reasoning process on this kind of problem.")

                accuracy_feedback = f"Incorrect. Related issues: {' | '.join(issues)}"

            # 2. Analyze model's reasoning - check if raw output mentions discriminative keywords
            reasoning_feedback = []
            # Check if prompt mentions <answer> tags (preferred format for clean_response)
            has_answer_tags = '<answer>' in prompt.lower()
            if not has_answer_tags:
                reasoning_feedback.append("Should specify <answer>X</answer> tag format for reliable extraction")

            if raw_output and accuracy == 0.0:
                raw_lower = raw_output.lower()

                # === A. STRUCTURED THINKING (Step-by-step breakdown) ===
                step_indicators = [
                    'first', 'second', 'third', 'finally',
                    'step',  '1', '2', '3',
                    'let me', 'let\'s',
                ]
                has_structured_thinking = sum(1 for indicator in step_indicators if indicator in raw_lower)
                if has_structured_thinking >= 2:
                    reasoning_feedback.append(f"Uses structured thinking ({has_structured_thinking} step indicators) ✓")
                else:
                    reasoning_feedback.append(
                        "Response lacks structured approach. LLM struggles to solve this kind of problem. Try to break down reasoning step-by-step."
                    )

            if not reasoning_feedback and accuracy == 0.0:
                reasoning_feedback.append(
                    "Reasoning structure looks okay but conclusion was wrong. Consider adding domain-specific guidance on solving this issue (e.g., 'A pentagon has 5 vertices')"
                )

            prompt_feedback = []
            prompt_verbosity_feedback = evaluate_prompt_verbosity(prompt)
            prompt_feedback.append(f"Prompt verbosity status: {prompt_verbosity_feedback['status']}. Recommendation: {prompt_verbosity_feedback['recommendation']}")

            # 3. Combined reward: 100% accuracy (focus on correctness)
            total_score = accuracy

            # Return structured feedback
            feedback = Feedback(
                accuracy_feedback=accuracy_feedback,
                reasoning_feedback=', '.join(reasoning_feedback) if reasoning_feedback else None,
                prompt_feedback=', '.join(prompt_feedback) if prompt_feedback else None,
            )
            return RewardOutput(score=total_score, feedback=feedback)

    return reward_fn


# ============================================================================
# EVALUATION HELPERS
# ============================================================================

def calculate_accuracy(preds: list[str], labels: list[str]) -> float:
    """Calculate accuracy"""
    all_lower = lambda texts: [t.lower() for t in texts]
    preds_lower = all_lower(preds)
    labels_lower = all_lower(labels)
    correct = np.array(preds_lower) == np.array(labels_lower)
    return float(np.mean(correct))


async def evaluate_on_test_set(
    prompt: str,
    test_data: list[tuple[str, str]],
    llm: Any,
    clean_response_fn: Any,
    post_instruction: bool,
    desc: str = "Test eval"
) -> tuple[float, list[str], list[str]]:
    """Evaluate a prompt on test set"""
    test_messages = []
    test_labels = []

    for input_text, target in test_data:
        if post_instruction:
            full_prompt = f"{input_text}\n{prompt}"
        else:
            full_prompt = f"{prompt}\n{input_text}"
        test_messages.append([HumanMessage(content=full_prompt)])
        test_labels.append(target)

    # Batch evaluation
    BATCH_SIZE = 50
    responses = []

    for batch_start in tqdm(range(0, len(test_messages), BATCH_SIZE), desc=desc, ncols=100):
        batch_end = min(batch_start + BATCH_SIZE, len(test_messages))
        batch_msgs = test_messages[batch_start:batch_end]

        try:
            batch_responses = await llm.abatch(batch_msgs)
            responses.extend(batch_responses)
        except Exception as e:
            logger.error(f"Error in batch {batch_start}-{batch_end}: {e}")
            for i in range(batch_start, batch_end):
                try:
                    response = await llm.ainvoke(test_messages[i])
                    responses.append(response)
                except:
                    responses.append(AIMessage(content="N/A"))

    preds = [clean_response_fn(resp.content) for resp in responses]
    accuracy = calculate_accuracy(preds, test_labels)

    return accuracy, preds, test_labels


# ============================================================================
# MAIN EVALUATION FUNCTION
# ============================================================================

async def evaluate_dataset(
    dataset_name: str,
    method: str = "feedback",
    config_path: str = None
):
    """
    Main evaluation function

    Args:
        dataset_name: Name of dataset from configs/datasets.yaml
        method: 'promptagent' or 'feedback' (default)
        config_path: Optional custom path to datasets.yaml
    """
    # Load dataset configuration
    config = load_dataset_config(dataset_name, config_path)
    use_feedback = (method == "feedback")

    # Override the optimizer config based on method
    # This is a simpler approach that doesn't reinitialize Hydra
    optimizer_config = "mcts_feedback" if method == "feedback" else "mcts_promptagent"

    logger.info("="*80)
    logger.info(f"MCTS OPTIMIZATION - {method.upper()} METHOD")
    logger.info("="*80)
    logger.info(f"Dataset: {config['name']}")
    logger.info(f"Description: {config['description']}")
    logger.info(f"Data file: {config['data_dir']}")
    logger.info(f"Optimizer config: optimizer/{optimizer_config}.yaml\n")

    # Load dataset
    task = CustomTask(
        train_size=config['train_size'],
        eval_size=config['eval_size'],
        test_size=config['test_size'],
        seed=config['seed'],
        task_name=config['name'],
        task_description=config['description'],
        data_dir=config['data_dir'],
        post_instruction=config['post_instruction']
    )

    train_data, eval_data, test_data = create_dataset(task)

    logger.info(f"Dataset split:")
    logger.info(f"  Train: {len(train_data)} examples")
    logger.info(f"  Eval: {len(eval_data)} examples")
    logger.info(f"  Test: {len(test_data)} examples\n")

    # Create reward function
    reward_fn = create_reward_function(train_data, use_feedback=use_feedback)

    # Load the appropriate optimizer config manually
    from omegaconf import OmegaConf
    project_root = Path(__file__).parent.parent.parent
    optimizer_config_path = project_root / "configs" / "optimizer" / f"{optimizer_config}.yaml"
    optimizer_cfg = OmegaConf.load(optimizer_config_path)

    # Initialize LLM and world model using the loaded config
    llm = hydra_instantiate(optimizer_cfg.llm.student)

    if method == "promptagent":
        world_model = PromptAgentWorldModel(
            train_dataset=train_data,
            eval_dataset=eval_data,
            llm=llm,
            reward_fn=reward_fn,
            clean_response_fn=task.clean_response,
            minibatch_size_train=optimizer_cfg.minibatch_size_train,
            minibatch_size_eval=optimizer_cfg.minibatch_size_eval,
            post_instruction=config['post_instruction'],
        )
    else:  # feedback
        world_model = PromptOptimizationWorldModel(
            train_dataset=train_data,
            eval_dataset=eval_data,
            llm=llm,
            reward_fn=reward_fn,
            clean_response_fn=task.clean_response,
            minibatch_size_train=optimizer_cfg.minibatch_size_train,
            minibatch_size_eval=optimizer_cfg.minibatch_size_eval,
            post_instruction=config['post_instruction'],
        )

    # Update global cfg to use the loaded optimizer config
    # This ensures MCTS uses the correct prompts for gradient analysis
    import feedback_prompt_learning
    feedback_prompt_learning.cfg.optimizer = optimizer_cfg

    # Print the prompts being used for transparency
    logger.info("\n" + "="*80)
    logger.info("OPTIMIZER CONFIGURATION")
    logger.info("="*80)
    logger.info(f"Method: {method.upper()}")
    logger.info(f"Config file: optimizer/{optimizer_config}.yaml")
    logger.info(f"Iterations: {optimizer_cfg.num_iterations}")
    logger.info(f"Max depth: {optimizer_cfg.max_depth}")
    logger.info(f"Expand width: {optimizer_cfg.expand_width}")
    logger.info(f"Minibatch train: {optimizer_cfg.minibatch_size_train}")
    logger.info(f"Minibatch eval: {optimizer_cfg.minibatch_size_eval}")

    logger.info("\n" + "-"*80)
    logger.info("GRADIENT ANALYSIS PROMPT:")
    logger.info("-"*80)
    logger.info(optimizer_cfg.gradient_analysis_prompt)

    logger.info("\n" + "-"*80)
    logger.info("PROMPT GENERATION PROMPT:")
    logger.info("-"*80)
    logger.info(optimizer_cfg.prompt_generation_prompt)
    logger.info("="*80 + "\n")

    # Create optimizer
    optimizer = MCTSPromptOptimizerFeedback(
        initial_prompt=config['init_prompt'],
        world_model=world_model,
    )

    # Evaluate initial prompt on test set
    logger.info("="*80)
    logger.info("EVALUATING INITIAL PROMPT ON TEST SET")
    logger.info("="*80)
    logger.info(f"Initial prompt: {config['init_prompt']}\n")

    init_accuracy, _, _ = await evaluate_on_test_set(
        config['init_prompt'],
        test_data,
        llm,
        task.clean_response,
        config['post_instruction'],
        desc="Initial eval"
    )

    logger.info(f"\nINITIAL TEST ACCURACY: {init_accuracy:.4f}\n")

    # Run optimization
    logger.info("="*80)
    logger.info("RUNNING MCTS OPTIMIZATION")
    logger.info("="*80)
    await optimizer.run()

    # Get results
    output = optimizer.prepare_output()
    best_q_path = output['best_q_path']

    logger.info("\n" + "="*80)
    logger.info("SELECTING BEST PROMPT VIA EVAL SET")
    logger.info("="*80)

    # Evaluate all nodes on eval set
    eval_results = []
    for node_idx, node in enumerate(best_q_path):
        eval_acc, _, _ = await evaluate_on_test_set(
            node.prompt,
            eval_data,
            llm,
            task.clean_response,
            config['post_instruction'],
            desc=f"Node {node_idx}"
        )

        eval_results.append({
            'node_idx': node_idx,
            'node': node,
            'eval_accuracy': eval_acc,
            'train_q': node.Q,
            'train_reward': node.reward,
        })

        logger.info(f"Node {node_idx} | Eval: {eval_acc:.4f} | Train Q: {node.Q:.4f}")

    # Select best by eval accuracy
    best_result = max(eval_results, key=lambda x: x['eval_accuracy'])
    best_node = best_result['node']

    logger.info("\n" + "="*80)
    logger.info("BEST PROMPT SELECTED")
    logger.info("="*80)
    logger.info(f"Node Index: {best_result['node_idx']}")
    logger.info(f"Eval Accuracy: {best_result['eval_accuracy']:.4f}")
    logger.info(f"Training Q: {best_result['train_q']:.4f}")
    logger.info(f"\nPrompt:\n{best_node.prompt}\n")

    # Final test evaluation
    logger.info("="*80)
    logger.info("FINAL TEST SET EVALUATION")
    logger.info("="*80)

    final_accuracy, _, _ = await evaluate_on_test_set(
        best_node.prompt,
        test_data,
        llm,
        task.clean_response,
        config['post_instruction'],
        desc="Final test"
    )

    logger.info("\n" + "="*80)
    logger.info("FINAL RESULTS")
    logger.info("="*80)
    logger.info(f"Method: {method.upper()}")
    logger.info(f"Dataset: {config['name']}")
    logger.info(f"Initial Test Accuracy: {init_accuracy:.4f}")
    logger.info(f"Final Test Accuracy: {final_accuracy:.4f}")
    logger.info(f"Improvement: {final_accuracy - init_accuracy:+.4f}")
    logger.info(f"\nBest Prompt:\n{best_node.prompt}")
    logger.info("="*80)

    return optimizer, best_node, config, {
        'init_accuracy': init_accuracy,
        'final_accuracy': final_accuracy,
        'improvement': final_accuracy - init_accuracy,
    }


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MCTS Prompt Optimization")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name from configs/datasets.yaml"
    )
    parser.add_argument(
        "--method",
        type=str,
        default="feedback",
        choices=["promptagent", "feedback"],
        help="Optimization method (default: feedback)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to datasets.yaml config file (optional)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    # Run evaluation
    asyncio.run(evaluate_dataset(
        dataset_name=args.dataset,
        method=args.method,
        config_path=args.config
    ))
