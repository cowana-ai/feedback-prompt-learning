"""
Unified MCTS Evaluation Script for BigBench Tasks
Supports both PromptAgent and Feedback-MCTS algorithms with easy method switching
"""

import argparse
import asyncio
import logging
import re
from collections import defaultdict
from typing import Any

import nltk
import numpy as np
import spacy
import tiktoken
from hydra.utils import instantiate as hydra_instantiate
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import OpenAIEmbeddings
from nltk.corpus import stopwords
from tqdm import tqdm

from feedback_prompt_learning import config
from feedback_prompt_learning.core import (Example, ExampleSet, FieldType,
                                           Signature, SignatureField)
from feedback_prompt_learning.data import Feedback, RewardOutput
from feedback_prompt_learning.data.task.bigbench import CustomTask
from feedback_prompt_learning.search_algo.mcts import \
    MCTSPromptOptimizerFeedback
from feedback_prompt_learning.search_algo.world_model import (
    PromptAgentWorldModel, PromptOptimizationWorldModel)

# Configure logger
logger = logging.getLogger(__name__)

# Constants
STOP_WORDS = None
if STOP_WORDS is None:
    nltk.download('stopwords', quiet=True)
    STOP_WORDS = set(stopwords.words('english'))

NLP = spacy.load("en_core_web_sm")
EMBEDDINGS_MODEL = OpenAIEmbeddings(model="text-embedding-3-small")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_dataset_config(dataset_name: str) -> dict[str, Any]:
    dataset_cfg = config._cfg.datasets[dataset_name]
    return dataset_cfg

def count_tokens_gpt4o_mini(text: str) -> int:
    """Calculate the number of tokens in a given text using GPT-4o Mini encoding."""
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    return len(tokens)


def evaluate_prompt_verbosity(
    prompt: str,
    min_tokens: int = 200,
    max_tokens: int = 420,
    min_content_tokens: int = 100,
    max_content_tokens: int = 380,
    language: str = 'english'
) -> dict[str, Any]:
    """
    Evaluate if a prompt is overly verbose or overly brief using token count and NLTK stopwords.
    Defaults are optimized for Big Bench Hard tasks which require detailed, complex prompts.
    """
    try:
        stop_words_set = set(stopwords.words(language))
    except LookupError:
        stop_words_set = STOP_WORDS

    words = prompt.split()
    token_count = count_tokens_gpt4o_mini(prompt)
    sentence_count = prompt.count('.') + prompt.count('!') + prompt.count('?')

    content_words = [
        word for word in words
        if word.lower().strip('.,!?;:()[]{}"\'"') not in stop_words_set and word.strip()
    ]
    content_text = ' '.join(content_words)
    content_token_count = count_tokens_gpt4o_mini(content_text)
    stop_token_count = token_count - content_token_count
    content_ratio = content_token_count / token_count if token_count > 0 else 0

    is_too_brief = (token_count < min_tokens or content_token_count < min_content_tokens)
    is_too_verbose = (token_count > max_tokens or content_token_count > max_content_tokens)

    if is_too_brief:
        status = "TOO_BRIEF"
        if content_token_count < min_content_tokens:
            recommendation = (
                "Prompt lacks meaningful content for the task tasks. "
                "Add problem context and clear instructions to improve clarity and guidance."
            )
        else:
            recommendation = (
                "Prompt is too short for the task complexity. Consider breaking down the problem "
                "and adding more detailed instructions or structure (including reasoning steps, rules or examples)."
            )
    elif is_too_verbose:
        status = "TOO_VERBOSE"
        if content_token_count > max_content_tokens:
            recommendation = (
                f"Prompt has excessive content. Focus on essential instructions and distill knowledge into a shorter prompt. "
                f"Current: {content_token_count} content tokens, target: under {max_content_tokens}."
            )
        else:
            recommendation = (
                f"Prompt is too long. Remove redundant info and distill knowledge into a shorter prompt. "
                f"Current: {token_count} tokens, target: under {max_tokens} tokens."
            )
    else:
        status = "OPTIMAL"
        if content_ratio < 0.35:
            recommendation = (
                f"Prompt length is appropriate for the task, but has many filler words ({content_ratio*100:.1f}% actual content). "
                "Make instructions more direct and technical."
            )
        elif content_ratio >= 0.60:
            recommendation = (
                f"Excellent content density for the task tasks ({content_ratio*100:.1f}% actual content). "
                "Good balance of detail and clarity!"
            )
        else:
            recommendation = (
                f"Prompt length and content balance are appropriate ({content_ratio*100:.1f}% actual content) for the task. "
                "You may try to further increase content density by reducing filler words and focusing on details and steps."
            )

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


def get_option_label(question_str: str, predicted_letter: str) -> str:
    """
    Retrieve the option label given the question string and the predicted option letter.
    """
    options_pattern = r"Options:\n((?:\([A-Z]\) .+\n?)+)"
    match = re.search(options_pattern, question_str)
    if not match:
        return None

    options_str = match.group(1)
    options = {}
    option_pattern = r"\(([A-Z])\) (.+)"
    for option_match in re.finditer(option_pattern, options_str):
        letter, label = option_match.groups()
        options[letter] = label

    return options.get(predicted_letter.upper(), None)


def split_reasoning(reasoning: str) -> list[str]:
    """Split reasoning into units using spaCy."""
    doc = NLP(reasoning)
    units = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 15]
    return units


def embed_units(units: list[str]) -> list[np.ndarray]:
    """Embed units using OpenAI embeddings."""
    embeddings = EMBEDDINGS_MODEL.embed_documents(units)
    return [np.array(emb) for emb in embeddings]


# ============================================================================
# DATA PREPARATION
# ============================================================================

def create_dataset(task: CustomTask) -> tuple[ExampleSet, ExampleSet, ExampleSet]:
    """Convert CustomTask dataset to ExampleSet format."""
    def format_examples(dataset, name: str):
        examples = []
        for example in dataset:
            examples.append(Example(
                inputs={"question": example['question']},
                outputs={"answer": example['answer']}
            ))
        return ExampleSet(examples=examples, name=name)

    train_data = format_examples(task.dataset["train"], "train")
    eval_data = format_examples(task.dataset["eval"], "eval")
    test_data = format_examples(task.dataset["test"], "test") if "test" in task.dataset else ExampleSet(examples=[], name="test")

    return train_data, eval_data, test_data


# ============================================================================
# REWARD FUNCTION CREATION
# ============================================================================

def create_reward_function(
    train_data: ExampleSet,
    use_feedback: bool = True,
    reward_type: str = "general"
) -> Any:
    """
    Create reward function for the task.
    """
    print(train_data)
    unique_targets = {example.outputs["answer"].upper() for example in train_data.examples}
    reasoning_memo = defaultdict(list)
    reasoning_basis = defaultdict(list)

    cos_sim = lambda a, b: np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

    def update_reasoning_basis(label: str, units: list[str], unit_embeddings: list[np.ndarray], sim_threshold: float = 0.8):
        basis = reasoning_basis[label]
        for text, emb in zip(units, unit_embeddings):
            if not basis:
                basis.append({"center": emb, "examples": [text]})
                continue

            sims = [cos_sim(emb, c["center"]) for c in basis]
            best_idx = max(range(len(sims)), key=lambda i: sims[i])

            if sims[best_idx] >= sim_threshold:
                basis[best_idx]["examples"].append(text)
            else:
                basis.append({"center": emb, "examples": [text]})

    def detect_missing_reasoning(
        failed_embeddings: list[np.ndarray],
        basis: list[dict],
        threshold: float = 0.75,
    ) -> list[dict]:
        missing_clusters = []
        for cluster in basis:
            center = cluster["center"]
            if not failed_embeddings:
                missing_clusters.append(cluster)
                continue

            max_similarity = max(cos_sim(center, emb) for emb in failed_embeddings)
            if max_similarity < threshold:
                missing_clusters.append(cluster)
        return missing_clusters

    def detect_hallucinations(
        failed_units: list[str],
        failed_embeddings: list[np.ndarray],
        basis: list[dict],
        threshold: float = 0.5,
    ) -> list[str]:
        hallucinated_units = []
        for text, emb in zip(failed_units, failed_embeddings):
            if not basis:
                hallucinated_units.append(text)
                continue

            max_similarity = max(cos_sim(emb, cluster["center"]) for cluster in basis)
            if max_similarity < threshold:
                hallucinated_units.append(text)
        return hallucinated_units

    def detect_wrong_label_attraction(
        failed_units: list[str],
        failed_embeddings: list[np.ndarray],
        target_label: str,
        reasoning_basis: dict,
    ) -> dict:
        best_label = None
        best_similarity = -1.0
        best_example = None

        for label, basis in reasoning_basis.items():
            if not basis or label == target_label:
                continue

            for cluster in basis:
                center = cluster["center"]
                for emb in failed_embeddings:
                    sim = cos_sim(emb, center)
                    if sim > best_similarity:
                        best_similarity = sim
                        best_label = label
                        best_example = cluster["examples"][0]

        if best_label is None:
            return None

        return {
            "wrong_label": best_label,
            "similarity": best_similarity,
            "example": best_example,
        }

    if not use_feedback:
        def reward_fn(output: str, target: str, prompt: str, raw_output: str = "", question: str = "") -> RewardOutput:
            output_clean = output.strip().upper()
            target_clean = target.strip().upper()

            score = 1.0 if output_clean == target_clean else 0.0
            feedback = Feedback(accuracy_feedback="Correct" if score == 1.0 else "Incorrect")

            return RewardOutput(score=score, feedback=feedback)
    elif reward_type == "general":
        def reward_fn(output: str, target: str, prompt: str, raw_output: str = "", question: str = "") -> RewardOutput:
            """
            Enhanced reward function with TF-IDF keyword analysis.
            Analyzes output after clean_response() processing.
            Checks if model's raw response mentions discriminative keywords identified via TF-IDF.
            Provides feedback on correctness and reasoning quality.
            """
            output_clean = output.strip().upper()
            target_clean = target.strip().upper()

            if output_clean == target_clean:
                accuracy = 1.0
                accuracy_feedback = "Correct answer ✓"
            else:
                accuracy = 0.0
                issues = []

                has_valid_answer = any(possible_target in output_clean for possible_target in unique_targets)

                if not has_valid_answer:
                    issues.append(f"Format error: clean_response() could not extract answer, possible answers are {', '.join(unique_targets)}. Model's output must either: (1) use <answer>X</answer> tags with letter inside, OR (2) contain a clear option letter (A-E) or (A)-(E) format, OR (3) match one of the option text strings. Prompt should specify: 'Provide your reasoning, then output your final answer as <answer>X</answer> .'")
                else:
                    wrong_answer = output_clean
                    issues.append(f"Wrong answer extracted: '{wrong_answer}' instead of '{target_clean}'. Identify what went wrong in current reasoning and propose critical domain knowledge to improve reasoning process on this kind of problem.")

                accuracy_feedback = f"Incorrect. Related issues: {' | '.join(issues)}"

            reasoning_feedback = []
            has_answer_tags = '<answer>' in prompt.lower()
            if not has_answer_tags:
                reasoning_feedback.append("Should specify <answer>X</answer> tag format for reliable extraction")

            if raw_output and accuracy == 0.0:
                raw_lower = raw_output.lower()

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

            total_score = accuracy

            feedback = Feedback(
                accuracy_feedback=accuracy_feedback,
                reasoning_feedback=', '.join(reasoning_feedback) if reasoning_feedback else None,
                prompt_feedback=', '.join(prompt_feedback) if prompt_feedback else None,
            )
            return RewardOutput(score=total_score, feedback=feedback)
    elif reward_type == "memory_enhanced":
        def reward_fn(output: str, target: str, prompt: str, raw_output: str = "", question: str = "") -> RewardOutput:
            """
            Geometric shapes specialized reward function.

            Strategy: Pure accuracy focus with rich diagnostic feedback ONLY when wrong.
            - Correct answer → score = 1.0 (no penalties)
            - Wrong answer → score = 0.0 (detailed feedback for improvement)

            This matches PromptAgent's binary reward structure while providing
            actionable spatial reasoning feedback to guide prompt evolution.
            """
            output_clean = output.strip().upper()
            target_clean = target.strip().upper()
            score = int(output_clean == target_clean)
            accuracy_feedback_parts = []
            reasoning_feedback = []
            prompt_feedback = []
            if score == 0.0:
                has_valid_answer = any(possible_target in output_clean for possible_target in unique_targets)

                if not has_valid_answer:
                    accuracy_feedback_parts.append(f"Format error: could not extract answer from {unique_targets}. Use <answer>X</answer> tags.")
                else:
                    wrong_answer = output_clean
                    accuracy_feedback_parts.append(f"Wrong answer extracted: '{wrong_answer}' instead of '{target_clean}'. Identify what went wrong in current reasoning and propose critical domain knowledge to improve reasoning process on this kind of problem.")

            if score == 1.0:
                accuracy_feedback_parts.append("Correct answer ✓. Digest this reasoning style for future problems.")

            if len(question) and len(raw_output):
                raw_lower = raw_output.lower()
                reasoning_section = raw_lower
                if "<answer>" in raw_lower and "</answer>" in raw_lower:
                    reasoning_section = " ".join(re.split(r'<answer>.*?</answer>', raw_lower, flags=re.DOTALL)).strip()
                target_label = get_option_label(question, target_clean)

                if score == 1.0:
                    reasoning_memo[target_label].append(raw_output)

                    units = split_reasoning(reasoning_section)
                    unit_embeddings = embed_units(units)

                    update_reasoning_basis(target_label, units, unit_embeddings)
                    if len(units) < 3:
                        reasoning_feedback.append("Good answer, but try to provide more detailed reasoning with at least 3-4 distinct steps analyzing steps to reinforce understanding.")

            if score == 0.0 and len(question) and len(raw_output):
                predicted_label = get_option_label(question, output_clean)
                failed_units = split_reasoning(reasoning_section)
                failed_embeddings = embed_units(failed_units)
                basis = reasoning_basis[target_label]

                missing = detect_missing_reasoning(failed_embeddings, basis)
                hallucinated = detect_hallucinations(failed_units, failed_embeddings, basis)
                attraction = detect_wrong_label_attraction(
                    failed_units,
                    failed_embeddings,
                    target_label,
                    reasoning_basis
                )
                for c in missing[:2]:
                    reasoning_feedback.append(
                        f"Missing reasoning similar to: '{c['examples'][0]}' to reach correct answer '{target_clean}'."
                    )

                for h in hallucinated[:2]:
                    reasoning_feedback.append(
                        f"Unsupported reasoning step detected: '{h}'. Avoid such hallucinations in future reasoning."
                    )

                if attraction:
                    reasoning_feedback.append(
                        f"Reasoning resembles solutions for option '{attraction['wrong_label']}'. "
                        f"Example: '{attraction['example']}'."
                    )

            has_answer_tags = '<answer>' in prompt.lower()

            if not has_answer_tags:
                prompt_feedback.append("Specify <answer>X</answer> format for reliable extraction.")

            feedback = Feedback(
                accuracy_feedback=' | '.join(accuracy_feedback_parts),
                reasoning_feedback='; '.join(reasoning_feedback) if reasoning_feedback else None,
                prompt_feedback='; '.join(prompt_feedback) if prompt_feedback else None,
            )

            return RewardOutput(score=score, feedback=feedback)
    else:
        raise ValueError(f"Unknown reward_type: {reward_type}. Supported types: 'general', 'memory_enhanced'")

    return reward_fn


# ============================================================================
# EVALUATION HELPERS
# ============================================================================

def calculate_accuracy(preds: list[str], labels: list[str]) -> float:
    """Calculate accuracy."""
    all_lower = lambda texts: [t.lower() for t in texts]
    preds_lower = all_lower(preds)
    labels_lower = all_lower(labels)
    correct = np.array(preds_lower) == np.array(labels_lower)
    return float(np.mean(correct))


async def evaluate_on_test_set(
    prompt: str,
    world_model: Any,
    test_data: ExampleSet,
    llm: Any,
    clean_response_fn: Any,
    post_instruction: bool,
    desc: str = "Test eval"
) -> tuple[float, list[str], list[str]]:
    """Evaluate a prompt on test set."""
    test_messages = []
    test_labels = []

    for example in test_data.examples:
        # Format input text
        input_text = world_model.prepare_input(example)
        target = world_model.prepare_output(example)

        if post_instruction:
            full_prompt = f"{input_text}\n{prompt}"
        else:
            full_prompt = f"{prompt}\n{input_text}"
        test_messages.append([HumanMessage(content=full_prompt)])
        test_labels.append(target)

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
    reward_type: str = "general"
):
    """
    Main evaluation function.

    Args:
        dataset_name: Name of dataset from configs/datasets.yaml
        method: 'promptagent' or 'feedback' (default)
        config_path: Optional custom path to datasets.yaml
    """
    dataset_config = load_dataset_config(dataset_name)
    use_feedback = (method == "feedback")
    logger.info("="*80)
    logger.info(f"MCTS OPTIMIZATION - {method.upper()} METHOD")
    logger.info("="*80)
    logger.info(f"Dataset: {dataset_config['name']}")
    logger.info(f"Description: {dataset_config['description']}")
    logger.info(f"Data file: {dataset_config['data_path']}")

    task = CustomTask(
        train_size=dataset_config['train_size'],
        eval_size=dataset_config['eval_size'],
        test_size=dataset_config['test_size'],
        seed=dataset_config['seed'],
        task_name=dataset_config['name'],
        task_description=dataset_config['description'],
        data_dir=dataset_config['data_path'],
        post_instruction=dataset_config['post_instruction']
    )

    train_data, eval_data, test_data = create_dataset(task)

    logger.info(f"Dataset split:")
    logger.info(f"  Train: {len(train_data)} examples")
    logger.info(f"  Eval: {len(eval_data)} examples")
    logger.info(f"  Test: {len(test_data)} examples\n")

    reward_fn = create_reward_function(train_data, use_feedback=use_feedback, reward_type=reward_type)


    config.reload([f"optimizer/search_algo=mcts_{method}"])

    llm = hydra_instantiate(config._cfg.optimizer.llm.student)

    if method == "promptagent":
        world_model = PromptAgentWorldModel(
            train_dataset=train_data,
            eval_dataset=eval_data,
            llm=llm,
            reward_fn=reward_fn,
            clean_response_fn=task.clean_response,
            minibatch_size_train=config._cfg.optimizer.minibatch_size_train,
            minibatch_size_eval=config._cfg.optimizer.minibatch_size_eval,
            post_instruction=dataset_config['post_instruction'],
        )
    else:
        world_model = PromptOptimizationWorldModel(
            train_dataset=train_data,
            eval_dataset=eval_data,
            llm=llm,
            reward_fn=reward_fn,
            clean_response_fn=task.clean_response,
            minibatch_size_train=config._cfg.optimizer.minibatch_size_train,
            minibatch_size_eval=config._cfg.optimizer.minibatch_size_eval,
            post_instruction=dataset_config['post_instruction'],
        )

    logger.info("\n" + "="*80)
    logger.info("OPTIMIZER CONFIGURATION")
    logger.info("="*80)
    logger.info(f"Method: {method.upper()}")
    logger.info(f"Iterations: {config._cfg.optimizer.num_iterations}")
    logger.info(f"Max depth: {config._cfg.optimizer.max_depth}")
    logger.info(f"Expand width: {config._cfg.optimizer.expand_width}")
    logger.info(f"Minibatch train: {config._cfg.optimizer.minibatch_size_train}")
    logger.info(f"Minibatch eval: {config._cfg.optimizer.minibatch_size_eval}")

    logger.info("\n" + "-"*80)
    logger.info("GRADIENT ANALYSIS PROMPT:")
    logger.info("-"*80)
    logger.info(config._cfg.optimizer.search_algo.gradient_analysis_prompt)

    logger.info("\n" + "-"*80)
    logger.info("PROMPT GENERATION PROMPT:")
    logger.info("-"*80)
    logger.info(config._cfg.optimizer.search_algo.prompt_generation_prompt)
    logger.info("="*80 + "\n")

    # Create signature for the task
    signature = Signature(
        name=dataset_config['name'],
        inputs=[SignatureField("input", "The input question or text", FieldType.INPUT)],
        outputs=[SignatureField("target", "The expected answer or output", FieldType.OUTPUT)],
        instructions=f"Task: {dataset_config['description']}"
    )

    optimizer = MCTSPromptOptimizerFeedback(
        initial_prompt=dataset_config['init_prompt'],
        world_model=world_model,
        signature=signature,
    )

    logger.info("="*80)
    logger.info("EVALUATING INITIAL PROMPT ON TEST SET")
    logger.info("="*80)
    logger.info(f"Initial prompt: {dataset_config['init_prompt']}\n")

    init_accuracy, _, _ = await evaluate_on_test_set(
        dataset_config['init_prompt'],
        world_model,
        test_data,
        llm,
        task.clean_response,
        dataset_config['post_instruction'],
        desc="Initial eval"
    )

    logger.info(f"\nINITIAL TEST ACCURACY: {init_accuracy:.4f}\n")

    logger.info("="*80)
    logger.info("RUNNING MCTS OPTIMIZATION")
    logger.info("="*80)
    await optimizer.run()

    output = optimizer.prepare_output()
    best_q_path = output['best_q_path']
    if len(best_q_path) > 1:
        best_q_path = best_q_path[1:]

    logger.info("\n" + "="*80)
    logger.info("SELECTING BEST PROMPT VIA EVAL SET")
    logger.info("="*80)

    eval_results = []
    for node_idx, node in enumerate(best_q_path):
        eval_acc, _, _ = await evaluate_on_test_set(
            node.prompt_version.prompt_text,
            world_model,
            eval_data,
            llm,
            task.clean_response,
            dataset_config['post_instruction'],
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

    best_result = max(eval_results, key=lambda x: x['eval_accuracy'])
    best_node = best_result['node']

    logger.info("\n" + "="*80)
    logger.info("BEST PROMPT SELECTED")
    logger.info("="*80)
    logger.info(f"Node Index: {best_result['node_idx']}")
    logger.info(f"Eval Accuracy: {best_result['eval_accuracy']:.4f}")
    logger.info(f"Training Q: {best_result['train_q']:.4f}")
    logger.info(f"\nPrompt:\n{best_node.prompt_version.prompt_text}\n")

    logger.info("="*80)
    logger.info("FINAL TEST SET EVALUATION")
    logger.info("="*80)

    final_accuracy, _, _ = await evaluate_on_test_set(
        best_node.prompt_version.prompt_text,
        world_model,
        test_data,
        llm,
        task.clean_response,
        dataset_config['post_instruction'],
        desc="Final test"
    )

    logger.info("\n" + "="*80)
    logger.info("FINAL RESULTS")
    logger.info("="*80)
    logger.info(f"Method: {method.upper()}")
    logger.info(f"Dataset: {dataset_config['name']}")
    logger.info(f"Initial Test Accuracy: {init_accuracy:.4f}")
    logger.info(f"Final Test Accuracy: {final_accuracy:.4f}")
    logger.info(f"Improvement: {final_accuracy - init_accuracy:+.4f}")
    logger.info(f"\nBest Prompt:\n{best_node.prompt_version.prompt_text}")
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
        "--reward_type",
        type=str,
        default="general",
        help="Reward type to use (optional)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    asyncio.run(evaluate_dataset(
        dataset_name=args.dataset,
        method=args.method,
        reward_type=args.reward_type,
    ))
