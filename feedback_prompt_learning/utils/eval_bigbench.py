"""
Unified MCTS Evaluation Script for BigBench Tasks
Supports easy configuration switching between different JSON datasets
"""

import asyncio
import logging
from typing import List, Tuple, Dict, Any
import numpy as np
from tqdm import tqdm

from feedback_prompt_learning.data.task.bigbench import CustomTask
from feedback_prompt_learning.search_algo.mcts import MCTSPromptOptimizerFeedback

import nltk
from nltk.corpus import stopwords

# Configure logger
logger = logging.getLogger(__name__)

# Download stopwords if not already present (only needs to run once)
try:
    STOP_WORDS = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords', quiet=True)
    STOP_WORDS = set(stopwords.words('english'))
# def evaluate_prompt_verbosity(prompt, min_words=15, max_words=500, min_chars=75, max_chars=3000, 
#                               min_content_words=10, max_content_words=350, language='english'):

def evaluate_prompt_verbosity(prompt, min_words=15, max_words=250, min_chars=75, max_chars=1200,
                             min_content_words=10, max_content_words=150, language='english'):
    """
    Evaluates if a prompt is overly verbose or overly brief using NLTK stopwords.
    Defaults are optimized for Big Bench Hard tasks which require detailed, complex prompts.
    
    Args:
        prompt (str): The prompt text to evaluate
        min_words (int): Minimum word count threshold (default: 15 for BBH)
        max_words (int): Maximum word count threshold (default: 700 for BBH)
        min_chars (int): Minimum character count threshold (default: 75 for BBH)
        max_chars (int): Maximum character count threshold (default: 3000 for BBH)
        min_content_words (int): Minimum content words after removing stop words (default: 10 for BBH)
        max_content_words (int): Maximum content words after removing stop words (default: 350 for BBH)
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
    word_count = len(words)
    char_count = len(prompt.strip())
    sentence_count = prompt.count('.') + prompt.count('!') + prompt.count('?')
    avg_word_length = char_count / word_count if word_count > 0 else 0
    
    # Remove stop words and calculate content metrics
    content_words = [word.lower().strip('.,!?;:()[]{}"\'"') for word in words 
                     if word.lower().strip('.,!?;:()[]{}"\'"') not in stop_words_set and word.strip()]
    content_word_count = len(content_words)
    stop_word_count = word_count - content_word_count
    content_ratio = content_word_count / word_count if word_count > 0 else 0
    
    # Determine status based on both total words and content words
    is_too_brief = (word_count < min_words or char_count < min_chars or 
                    content_word_count < min_content_words)
    is_too_verbose = (word_count > max_words or char_count > max_chars or 
                      content_word_count > max_content_words)
    
    if is_too_brief:
        status = "TOO_BRIEF"
        if content_word_count < min_content_words:
            recommendation = f"Prompt lacks meaningful content for BBH tasks. Add problem context and clear instructions. Current: {content_word_count} content words, target: at least {min_content_words}."
        else:
            recommendation = f"Prompt is too short for BBH complexity. Include: problem statement, reasoning steps, and output format. Current: {word_count} words, target: at least {min_words} words."
    elif is_too_verbose:
        status = "TOO_VERBOSE"
        if content_word_count > max_content_words:
            recommendation = f"Prompt has excessive content. Focus on essential instructions and distill knowledge into a shorter prompt. Current: {content_word_count} content words, target: under {max_content_words}."
        else:
            recommendation = f"Prompt is too long. Remove redundant info and distill knowledge into a shorter prompt. Current: {word_count} words, target: under {max_words} words."
    else:
        status = "OPTIMAL"
        if content_ratio < 0.35:
            recommendation = f"Prompt length is appropriate for BBH, but has many filler words ({content_ratio*100:.1f}% content). Make instructions more direct and technical."
        elif content_ratio >= 0.60:
            recommendation = f"Excellent content density for BBH tasks ({content_ratio*100:.1f}% content). Good balance of detail and clarity!"
        else:
            recommendation = f"Prompt length and content balance are appropriate ({content_ratio*100:.1f}% content) for Big Bench Hard tasks. You may try to further increase content density by reducing filler words and focusing on details and steps."
    
    return {
        "status": status,
        "is_too_brief": is_too_brief,
        "is_too_verbose": is_too_verbose,
        "metrics": {
            "word_count": word_count,
            "content_word_count": content_word_count,
            "stop_word_count": stop_word_count,
            "content_ratio": round(content_ratio, 3),
            "char_count": char_count,
            "sentence_count": sentence_count,
            "avg_word_length": round(avg_word_length, 2)
        },
        "thresholds": {
            "min_words": min_words,
            "max_words": max_words,
            "min_content_words": min_content_words,
            "max_content_words": max_content_words,
            "min_chars": min_chars,
            "max_chars": max_chars
        },
        "recommendation": recommendation,
        "content_words_sample": content_words[:10] if content_words else []
    }

# ============================================================================
# DATASET CONFIGURATIONS
# ============================================================================

DATASET_CONFIGS = {
    "penguins": {
        "name": "penguins_in_a_table",
        "description": "Answer questions about a table of penguins and their attributes",
        "data_dir": "/Users/chingisoinar/feedback-prompt-learning/dataset/penguins_in_a_table.json",
        "init_prompt": "Answer questions about a table of penguins and their attributes.",
        "train_size": 70,
        "eval_size": 70,
        "test_size": 79,
        "seed": 42,
        "post_instruction": False,  # Question before prompt
    },
    "geometric_shapes": {
        "name": "geometric_shapes",
        "description": "Name geometric shapes from their SVG paths",
        "data_dir": "/Users/chingisoinar/feedback-prompt-learning/dataset/geometric_shapes.json",
        "init_prompt": "Identify the geometric shape from the SVG path element.",
        "train_size": 150,
        "eval_size": 150,
        "test_size": 200,
        "seed": 42,
        "post_instruction": False,  # Question before prompt
    },
    "epistemic": {
        "name": "epistemic",
        "description": "Determine whether one sentence entails the next",
        "data_dir": "/workspace/tripllla_context_guard/pipelines/eval/epistemic.json",
        "init_prompt": "Determine whether one sentence entails the next.",
        "train_size": 300,
        "eval_size": 200,
        "test_size": 500,
        "seed": 42,
        "post_instruction": False,  # Question before prompt
    },
    "object_counting": {
        "name": "object_counting",
        "description": "Questions that involve enumerating objects of different types and asking the model to count them",
        "data_dir": "/Users/chingisoinar/feedback-prompt-learning/dataset/object_counting.json",
        "init_prompt": "Count the overall number of all items.",
        "train_size": 150,
        "eval_size": 150,
        "test_size": 500,
        "seed": 42,
        "post_instruction": False,  # Question before prompt
    },
    "temporal_sequences": {
        "name": "temporal_sequences",
        "description": "Answer questions about which times certain events could have occurred",
        "data_dir": "/workspace/tripllla_context_guard/pipelines/eval/temporal_sequences.json",
        "init_prompt": "Answer questions about which times certain events could have occurred.",
        "train_size": 150,
        "eval_size": 150,
        "test_size": 500,
        "seed": 42,
        "post_instruction": False,  # Question before prompt
    },
    "casual_judgement": {
        "name": "causal_judgment",
        "description": "Answer questions about causal attribution",
        "data_dir": "/Users/chingisoinar/feedback-prompt-learning/dataset/casual_judgement.json",
        "init_prompt": "Answer questions about causal attribution.",
        "train_size": 90,
        "eval_size": 90,
        "test_size": 100,
        "seed": 42,
        "post_instruction": False,  # Question before prompt
    },
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_dataset(task: CustomTask) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]], List[Tuple[str, str]]]:
    """
    Convert CustomTask dataset to (input, target) format for MCTS
    
    Returns:
        train_data, eval_data, test_data
    """
    def format_examples(dataset):
        formatted = []
        for example in dataset:
            question = example['question']
            answer = example['answer']
            formatted.append((question, answer))
        return formatted
    
    train_data = format_examples(task.dataset["train"])
    eval_data = format_examples(task.dataset["eval"])
    test_data = format_examples(task.dataset["test"]) if "test" in task.dataset else []
    
    return train_data, eval_data, test_data


def create_reward_function(train_data: List[Tuple[str, str]]) -> Any:
    """
    Create reward function for the task with keyword analysis from training data
    
    Args:
        task: CustomTask instance
        task_config: Dataset configuration dictionary
        train_data: Training examples for keyword extraction
        
    Returns:
        Reward function that returns (score, feedback) tuple
    """
    # Extract unique possible target values from training data
    unique_targets = set(target.strip().upper() for _, target in train_data)
    
    # ========================================================================
    # REWARD FUNCTION WITH KEYWORD ANALYSIS
    # ========================================================================
    import re
    def reward_fn(output: str, target: str, prompt: str, raw_output: str = "") -> Tuple[float, str]:
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
        
        # Combine feedback
        if reasoning_feedback:
            feedback = f"{accuracy_feedback} | Reasoning feedback: {', '.join(reasoning_feedback)} | Prompt feedback: {', '.join(prompt_feedback)}"
        else:
            feedback = f"{accuracy_feedback} | Prompt feedback: {', '.join(prompt_feedback)}"
        
        return total_score, feedback
    
    return reward_fn


async def evaluate_dataset(dataset_name: str = "penguins"):
    """
    Main evaluation function for any BigBench task
    
    Args:
        dataset_name: Name of dataset config to use (from DATASET_CONFIGS)
    """
    # Get configuration
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASET_CONFIGS.keys())}")
    
    config = DATASET_CONFIGS[dataset_name]
    
    # Load dataset using CustomTask
    logger.info("="*80)
    logger.info(f"LOADING DATASET: {config['name'].upper()}")
    logger.info("="*80)
    logger.info(f"Description: {config['description']}")
    logger.info(f"Data file: {config['data_dir']}\n")

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
    
    # Convert to MCTS format
    train_data, eval_data, test_data = create_dataset(task)
    
    logger.info(f"Dataset split:")
    logger.info(f"  Train: {len(train_data)} examples")
    logger.info(f"  Eval: {len(eval_data)} examples")
    logger.info(f"  Test: {len(test_data)} examples")
    logger.info(f"  Total: {len(train_data) + len(eval_data) + len(test_data)} examples\n")
    
    # Print sample training examples
    logger.info("="*80)
    logger.info("SAMPLE TRAINING EXAMPLES")
    logger.info("="*80)
    num_samples = min(3, len(train_data))
    for i in range(num_samples):
        question, answer = train_data[i]
        logger.info(f"\nExample {i+1}:")
        logger.info(f"Question:\n{question[:500]}..." if len(question) > 500 else f"Question:\n{question}")
        logger.info(f"Answer: {answer}")
        logger.info("-"*80)
    logger.info("")
    
    # Create reward function with keyword analysis from training data
    reward_fn = create_reward_function(train_data)
    
    # Initialize MCTS optimizer
    logger.info("="*80)
    logger.info("INITIALIZING MCTS PROMPT OPTIMIZER (FEEDBACK VERSION)")
    logger.info("="*80 + "\n")
    
    optimizer = MCTSPromptOptimizerFeedback(
        initial_prompt=config['init_prompt'],
        train_dataset=train_data,  # Use train data for optimization
        eval_dataset=eval_data,  # Use eval data for validation during optimization
        reward_fn=reward_fn,
        clean_response_fn=task.clean_response,  # Use task's clean_response function
        post_instruction=config['post_instruction'],
    )
    llm = optimizer.llm  # For direct API calls later
    
    # Evaluate initial prompt on test data before optimization
    logger.info("="*80)
    logger.info("EVALUATING INITIAL PROMPT ON TEST DATA")
    logger.info("="*80)
    logger.info(f"Initial prompt: {config['init_prompt']}\n")
    
    from langchain_core.messages import HumanMessage
    
    # Helper functions
    all_lower = lambda texts: [t.lower() for t in texts]
    def cal_correct(preds, labels):
        return list(np.array((np.array(all_lower(preds)) == np.array(all_lower(labels)))).astype(int))
    def cal_metric(preds, labels):
        correct = cal_correct(preds=all_lower(preds), labels=all_lower(labels))
        return np.mean(correct)
    
    # Build test messages for initial prompt
    init_test_messages = []
    test_labels = []
    for input_text, target in test_data:
        if config['post_instruction']:
            full_prompt = f"{input_text}\n{config['init_prompt']}\nWrap your final answer as <answer>X</answer> where X is one of the presented options."
        else:
            full_prompt = f"{config['init_prompt']}\n{input_text}\nWrap your final answer as <answer>X</answer> where X is one of the presented options."
        init_test_messages.append([HumanMessage(content=full_prompt)])
        test_labels.append(target)
    
    # Evaluate in batches
    logger.info(f"Evaluating initial prompt on {len(test_data)} test examples...")
    BATCH_SIZE = 50
    init_test_responses = []
    
    for batch_start in tqdm(range(0, len(init_test_messages), BATCH_SIZE), desc="Initial eval", ncols=100):
        batch_end = min(batch_start + BATCH_SIZE, len(init_test_messages))
        batch_messages = init_test_messages[batch_start:batch_end]
        
        try:
            batch_responses = await llm.abatch(batch_messages)
            init_test_responses.extend(batch_responses)
        except Exception as e:
            logger.error(f"Error in batch {batch_start}-{batch_end}: {e}")
            for i in range(batch_start, batch_end):
                try:
                    response = await llm.ainvoke(init_test_messages[i])
                    init_test_responses.append(response)
                except Exception as retry_error:
                    logger.error(f"Failed on message {i}: {retry_error}")
                    from langchain_core.messages import AIMessage
                    init_test_responses.append(AIMessage(content="N/A"))
    
    # Calculate initial accuracy
    init_test_preds = [task.clean_response(resp.content) for resp in init_test_responses]
    init_test_accuracy = cal_metric(init_test_preds, test_labels)
    
    logger.info(f"\n{'='*80}")
    logger.info(f"INITIAL PROMPT TEST ACCURACY: {init_test_accuracy:.4f}")
    logger.info(f"{'='*80}\n")
    
    # Run optimization
    best_node = await optimizer.run()
    
    # Get comprehensive results using prepare_output
    logger.info("\n" + "="*80)
    logger.info("OPTIMIZATION COMPLETE - PREPARING OUTPUT")
    logger.info("="*80)
    logger.info(f"\nInitial prompt: {config['init_prompt']}\n")
    
    output = optimizer.prepare_output()
    
    # Get the best Q path
    best_q_path = output['best_q_path']
    
    logger.info("\n" + "="*80)
    logger.info("EVALUATING ALL NODES IN BEST Q PATH ON EVAL SET")
    logger.info("="*80)
    logger.info(f"Best Q path has {len(best_q_path)} nodes (depth 0 to {len(best_q_path)-1})")
    logger.info(f"Mean Q: {np.mean([n.Q for n in best_q_path]):.4f}")
    logger.info(f"Mean Reward: {np.mean([n.reward for n in best_q_path]):.4f}\n")
    
    # Evaluate on eval set to select best prompt
    logger.info("="*80)
    logger.info("SELECTING BEST PROMPT: EVALUATING EACH NODE ON FULL EVAL SET")
    logger.info("="*80)
    
    all_lower = lambda texts: [t.lower() for t in texts]
    # Helper functions
    def cal_correct(preds, labels):
        return list(np.array((np.array(all_lower(preds)) == np.array(all_lower(labels)))).astype(int))
    
    def cal_metric(preds, labels):
        correct = cal_correct(preds=all_lower(preds), labels=all_lower(labels))
        return np.mean(correct)
    
    from langchain_core.messages import HumanMessage
    
    # Build all prompts for all nodes on eval set
    all_eval_messages = []
    node_eval_labels = []
    
    for node_idx, node in enumerate(best_q_path):
        eval_labels = []
        for input_text, target in eval_data:
            # Format based on post_instruction
            if config['post_instruction']:
                full_prompt = f"{input_text}\n{node.prompt}"
            else:
                full_prompt = f"{node.prompt}\n{input_text}"
            all_eval_messages.append([HumanMessage(content=full_prompt)])
            eval_labels.append(target)
        node_eval_labels.append(eval_labels)
    
    # Single batched API call for all nodes on eval set
    logger.info(f"Evaluating {len(best_q_path)} nodes on {len(eval_data)} eval examples ({len(all_eval_messages)} total API calls)...")
    
    # Process in batches to avoid rate limits and timeouts
    BATCH_SIZE = 50  # Process 50 requests at a time
    all_eval_responses = []
    
    for batch_start in tqdm(range(0, len(all_eval_messages), BATCH_SIZE), desc="API batches", ncols=100):
        batch_end = min(batch_start + BATCH_SIZE, len(all_eval_messages))
        batch_messages = all_eval_messages[batch_start:batch_end]
        
        try:
            batch_responses = await llm.abatch(batch_messages)
            all_eval_responses.extend(batch_responses)
        except Exception as e:
            logger.error(f"\nError in batch {batch_start}-{batch_end}: {e}")
            logger.info("Retrying with smaller batch size...")
            # Retry with smaller batches
            for i in range(batch_start, batch_end):
                try:
                    response = await llm.ainvoke(all_eval_messages[i])
                    all_eval_responses.append(response)
                except Exception as retry_error:
                    logger.error(f"Failed on message {i}: {retry_error}")
                    # Create dummy response to maintain indexing
                    from langchain_core.messages import AIMessage
                    all_eval_responses.append(AIMessage(content="N/A"))
    
    logger.info(f"Completed {len(all_eval_responses)} API calls")
    
    # Process eval results for each node
    eval_path_results = []
    eval_response_idx = 0
    
    for node_idx, node in enumerate(best_q_path):
        eval_preds = []
        eval_labels = node_eval_labels[node_idx]
        
        # Extract responses for this node
        for _ in range(len(eval_data)):
            output = task.clean_response(all_eval_responses[eval_response_idx].content)
            eval_preds.append(output)
            eval_response_idx += 1
        
        eval_accuracy = cal_metric(eval_preds, eval_labels)
        correct_list = cal_correct(eval_preds, eval_labels)
        
        eval_path_results.append({
            'node_idx': node_idx,
            'node': node,
            'eval_accuracy': eval_accuracy,
            'eval_correct': sum(correct_list),
            'eval_total': len(correct_list),
            'train_reward': node.reward,
            'train_q': node.Q,
            'visits': node.N
        })
        
        logger.info(f"Node {node_idx} | Eval Acc: {eval_accuracy:.4f} ({sum(correct_list)}/{len(correct_list)}) | Train Q: {node.Q:.4f}")
    
    # Select best node by eval accuracy
    best_eval_result = max(eval_path_results, key=lambda x: x['eval_accuracy'])
    best_eval_node = best_eval_result['node']
    
    logger.info("\n" + "="*80)
    logger.info("BEST NODE SELECTED BY EVAL ACCURACY")
    logger.info("="*80)
    logger.info(f"Node Index: {best_eval_result['node_idx']}")
    logger.info(f"Depth: {best_eval_node.depth}")
    logger.info(f"Eval Accuracy: {best_eval_result['eval_accuracy']:.4f} ({best_eval_result['eval_correct']}/{best_eval_result['eval_total']})")
    logger.info(f"Training Q-value: {best_eval_result['train_q']:.4f}")
    logger.info(f"Visits: {best_eval_result['visits']}")
    logger.info(f"\nSelected Prompt:\n{best_eval_node.prompt}\n")
    
    # Now evaluate the selected best prompt on test set
    # Now evaluate the selected best prompt on test set
    if len(test_data) > 0:
        logger.info("="*80)
        logger.info("FINAL EVALUATION: TESTING SELECTED BEST PROMPT ON HELD-OUT TEST SET")
        logger.info("="*80)
        
        # Build prompts for test set using only the best selected node
        test_messages = []
        test_labels = []
        
        for input_text, target in test_data:
            # Format based on post_instruction
            if config['post_instruction']:
                full_prompt = f"{input_text}\n{best_eval_node.prompt}"
            else:
                full_prompt = f"{best_eval_node.prompt}\n{input_text}"
            test_messages.append([HumanMessage(content=full_prompt)])
            test_labels.append(target)
        
        # Evaluate on test set
        logger.info(f"Evaluating best prompt on {len(test_data)} test examples...")
        
        # Process in batches to avoid rate limits
        BATCH_SIZE = 50
        test_responses = []
        
        for batch_start in tqdm(range(0, len(test_messages), BATCH_SIZE), desc="Test API batches", ncols=100):
            batch_end = min(batch_start + BATCH_SIZE, len(test_messages))
            batch_test_messages = test_messages[batch_start:batch_end]
            
            try:
                batch_responses = await llm.abatch(batch_test_messages)
                test_responses.extend(batch_responses)
            except Exception as e:
                logger.error(f"\nError in test batch {batch_start}-{batch_end}: {e}")
                # Retry individually
                for i in range(batch_start, batch_end):
                    try:
                        response = await llm.ainvoke(test_messages[i])
                        test_responses.append(response)
                    except Exception as retry_error:
                        logger.error(f"Failed on test message {i}: {retry_error}")
                        from langchain_core.messages import AIMessage
                        test_responses.append(AIMessage(content="N/A"))
        
        logger.info(f"Completed {len(test_responses)} test API calls")
        
        # Process test results
        test_preds = []
        for response in test_responses:
            output = task.clean_response(response.content)
            test_preds.append(output)
        
        test_accuracy = cal_metric(test_preds, test_labels)
        test_correct_list = cal_correct(test_preds, test_labels)
        
        logger.info("\n" + "="*80)
        logger.info("FINAL TEST RESULTS")
        logger.info("="*80)
        logger.info(f"Dataset: {config['name']}")
        logger.info(f"Selected Node Index: {best_eval_result['node_idx']}")
        logger.info(f"Depth: {best_eval_node.depth}")
        logger.info(f"\nEval Set Performance:")
        logger.info(f"  Accuracy: {best_eval_result['eval_accuracy']:.4f} ({best_eval_result['eval_correct']}/{best_eval_result['eval_total']})")
        logger.info(f"\nTest Set Performance:")
        logger.info(f"  Accuracy: {test_accuracy:.4f} ({sum(test_correct_list)}/{len(test_correct_list)})")
        logger.info(f"\nTraining Metrics:")
        logger.info(f"  Q-value: {best_eval_result['train_q']:.4f}")
        logger.info(f"  Reward: {best_eval_result['train_reward']:.4f}")
        logger.info(f"  Visits: {best_eval_result['visits']}")
        logger.info(f"\nFinal Prompt:\n{best_eval_node.prompt}\n")
        
        recommended_node = best_eval_node
    else:
        # No test set available, use best node from eval
        recommended_node = best_eval_node
    
    return optimizer, recommended_node, config


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="MCTS Prompt Optimization for BigBench Tasks")
    parser.add_argument(
        "--dataset",
        type=str,
        default="penguins",
        choices=list(DATASET_CONFIGS.keys()),
        help=f"Dataset to evaluate. Choices: {list(DATASET_CONFIGS.keys())}"
    )
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Suppress httpx logs from OpenAI API requests
    logging.getLogger("httpx").setLevel(logging.WARNING)
    
    logger.info("="*80)
    logger.info("MCTS PROMPT OPTIMIZATION - BIGBENCH TASKS")
    logger.info("="*80)
    logger.info(f"Selected Dataset: {args.dataset}")
    logger.info("="*80 + "\n")
    
    optimizer, best, config = asyncio.run(evaluate_dataset(args.dataset))
    
    logger.info("\n" + "="*80)
    logger.info("EVALUATION COMPLETE")
    logger.info("="*80)
    logger.info(f"Dataset: {config['name']}")
    logger.info(f"Best prompt found and evaluated on test set")
    logger.info("="*80)
