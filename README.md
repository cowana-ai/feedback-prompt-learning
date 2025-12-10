# Feedback-Prompt-Learning

> **Own your prompts. Define what you want. Let feedback drive the system.**

**Feedback-Prompt-Learning** is an open-source research framework for automatic prompt optimization using Monte Carlo Tree Search (MCTS) with feedback-driven gradient generation. This project critically examines PromptAgent's approach and proposes enhancements for more controllable, interpretable prompt optimization.

**🔬 Open Research Statement:** This is experimental research exploring the frontier of automatic prompt optimization. We provide transparent implementations of both PromptAgent's original approach and our feedback-enhanced variant to enable reproducible comparisons and community innovation.

---

## Philosophy: Eval-Driven Development

**Stop prompt engineering by trial-and-error.** Start with rigorous evaluation:

1. **Define what you want** - Write a reward function that captures your actual objectives
2. **Own your data** - Curate train/eval/test splits that reflect real-world usage  
3. **Let feedback guide optimization** - The system learns from detailed feedback, not just scores
4. **Validate systematically** - Proper data separation ensures prompts generalize

This isn't about finding "the perfect prompt" through intuition. It's about **building a system** that measurably improves on your specific task.

---

## The Problem with PromptAgent

PromptAgent pioneered MCTS for prompt optimization, but has critical limitations:

### 1. **Blind Gradient Generation**

PromptAgent's gradient (prompt improvement direction) comes from analyzing *only error cases*:
- ✗ No visibility into **why** prompts succeeded
- ✗ Cannot reinforce effective patterns
- ✗ Loses information about prompt strengths
- ✗ Optimization becomes random walk in later iterations

**Result:** The algorithm cannot steer optimization toward desired characteristics. It only knows what's broken, not what to build.

### 2. **No Directional Control**

Without rich feedback, you cannot guide optimization toward specific objectives:
- ✗ Cannot enforce reasoning quality (step-by-step, chain-of-thought)
- ✗ Cannot optimize for style, tone, or format
- ✗ Cannot balance multiple objectives (accuracy + brevity + reasoning)
- ✗ Cannot incorporate domain constraints

**Result:** You get prompts that maximize accuracy but may be verbose, poorly structured, or miss important reasoning steps.

### 3. **Reward Signal Poverty**

Binary rewards (0.0 or 1.0) provide minimal learning signal:
- ✗ Two wrong answers get identical feedback (both 0.0)
- ✗ Cannot distinguish "close but wrong" from "completely off"
- ✗ Cannot detect partial credit or intermediate progress
- ✗ Gradient generation relies solely on error pattern matching

**Result:** Slow convergence, brittle optimization, difficulty escaping local optima.

---

## Our Approach: Feedback-Driven MCTS

We extend PromptAgent with **rich textual feedback** throughout the optimization loop:

### Key Innovation 1: Reward Functions with Feedback

Instead of returning just a score, reward functions return structured feedback:

**Structured Feedback Format:**
- `accuracy_feedback`: Why the answer was correct/incorrect
- `reasoning_feedback`: Quality of reasoning process (step-by-step, clarity, logic)
- `prompt_feedback`: How the prompt could be improved for this case

This feedback is injected into the gradient generation phase, giving the LLM critic concrete directions for improvement.

### Key Innovation 2: Balanced Example Sampling

Unlike PromptAgent (errors only), we sample from both successes and failures:

**Adaptive Sampling Strategy:**
- **High-performing prompts (reward > 0.7):** Sample more successes to reinforce what works
- **Low-performing prompts (reward < 0.5):** Sample more errors to identify critical issues  
- **Mid-range prompts:** Balanced mix to guide refinement

**Why this matters:**
- Learn from positive examples: "This reasoning pattern worked well"
- Avoid breaking strengths: "Don't lose the step-by-step structure that succeeded"
- Targeted improvements: "Keep X, but improve Y"

### Key Innovation 3: Feedback-Aware Gradient Generation

The gradient analysis prompt receives:
- Current prompt
- Performance trajectory (historical prompts + scores)
- **Sampled examples with detailed feedback** (not just errors)
- Success/failure distribution

The LLM critic generates gradients by:
1. **Analyzing patterns across successes AND failures**
2. **Identifying what to preserve** (from high-scoring examples)
3. **Diagnosing specific issues** (from detailed feedback)
4. **Proposing targeted improvements** (not random variations)


---

## Algorithmic Comparison

### PromptAgent (Baseline)

**MCTS Loop:**
1. Select promising node via UCT (Upper Confidence Bound for Trees)
2. Expand: Generate new prompts by analyzing **error cases only**
3. Simulate: Evaluate new prompts on validation set
4. Backpropagate: Update Q-values based on reward

**Gradient Generation:**
- Input: Current prompt + error examples
- Output: Generic improvement suggestion
- Limitation: No positive reinforcement, no directional control

### Feedback-MCTS (This Work)

**Enhanced MCTS Loop:**
1. Select promising node via UCT (same)
2. Expand: Generate new prompts using **balanced example sampling + detailed feedback**
3. Simulate: Evaluate on validation set (same)
4. Backpropagate: Update Q-values (same)

**Enhanced Gradient Generation:**
- Input: Current prompt + trajectory + balanced examples with rich feedback
- Output: Targeted improvement preserving strengths
- Benefit: Directional optimization toward defined criteria

**Key Difference:** The expansion phase has access to WHY prompts succeed/fail, enabling smarter gradient generation.

---

## Core Principles

### 1. **Evaluation First**

Define rigorous evaluation *before* optimizing:

- **Reward function** - What does "good" mean for your task? Return scores AND feedback
- **Data splits** - Train/eval/test that mirror production
- **Metrics** - Beyond accuracy (reasoning quality, format compliance, etc.)

Your data reflects your real-world distribution. Own it.

### 2. **Feedback > Scores**

Rich feedback enables smarter optimization:
- Diagnostic: Understand failure modes
- Prescriptive: Guide improvement direction
- Preservative: Maintain successful patterns

---

## Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                   MCTS Tree Search (Selection)                     │
│  • UCT formula balances exploration vs exploitation                │
│  • Track Q-values (average cumulative reward per node)             │
└────────────────────────────────────────────────────────────────────┘
                                   ▼
┌────────────────────────────────────────────────────────────────────┐
│                  Expansion: Generate Child Prompts                 │
│                                                                    │
│  PromptAgent Mode:           │   Feedback-MCTS Mode:              │
│  • Sample errors only        │   • Adaptive balanced sampling     │
│  • Analyze error patterns    │   • Rich feedback (accuracy +      │
│  • Generic gradient          │     reasoning + prompt advice)     │
│                              │   • Targeted gradient preserving   │
│                              │     strengths                      │
└────────────────────────────────────────────────────────────────────┘
                                   ▼
┌────────────────────────────────────────────────────────────────────┐
│              Simulation: Evaluate New Prompts on Eval Set          │
│  • Get reward from your custom reward function                    │
│  • Collect feedback for next iteration                            │
└────────────────────────────────────────────────────────────────────┘
                                   ▼
┌────────────────────────────────────────────────────────────────────┐
│            Backpropagation: Update Q-values Up Tree                │
│  • Cumulative reward flows from leaf to root                      │
│  • Update visit counts and Q-values                               │
└────────────────────────────────────────────────────────────────────┘
```

**Configuration:** Switch between PromptAgent and Feedback-MCTS via config:
- `configs/optimizer/mcts_promptagent.yaml`: Error-only baseline
- `configs/optimizer/mcts_feedback.yaml`: Feedback-enhanced variant

---

## Research Questions & Open Problems

This framework enables empirical investigation of:

### 1. **Does Feedback Help?**
- Convergence speed: Feedback-MCTS vs PromptAgent
- Final performance: Best prompts found by each method
- Sample efficiency: Evaluations needed to reach target performance

### 2. **What Kind of Feedback Matters?**
- Accuracy feedback only vs reasoning feedback vs combined
- Prompt-level feedback vs example-level feedback
- Human-written feedback vs LLM-generated feedback

### 3. **Sampling Strategy Impact**
- Error-only vs balanced vs success-only sampling
- Static vs adaptive sampling based on node performance
- Optimal sample size and diversity

### 4. **Generalization vs Overfitting**
- Do feedback-optimized prompts generalize better?
- Train/eval/test performance gaps
- Robustness to distribution shift

**We provide both implementations to enable controlled experiments.** Results will vary by task, domain, and reward function design.

---

## Experimental Status & Limitations
**Current Limitations:**
- ⚠ No theoretical guarantees on convergence or optimality
- ⚠ Performance depends heavily on reward function design
- ⚠ LLM-generated gradients may be inconsistent or noisy
- ⚠ Computational cost: Each expansion requires multiple LLM calls
- ⚠ Limited to tasks where evaluation can be automated
- ⚠ Currently only OpenAI api is supported
---

## FAQ

**Q: Is feedback-driven MCTS always better than PromptAgent?**  
A: Unknown. This is open research. We provide tools to test it on your tasks. Early experiments show promise, but results are task-dependent.

**Q: What if my task doesn't have clear correct answers?**  
A: Define custom reward functions based on your criteria (coherence, relevance, style, etc.). The feedback mechanism is even more valuable for subjective tasks where you can articulate quality criteria.

**Q: Can I use this in production?**  
A: Use the optimized prompts, not the optimization process itself. Run MCTS offline, validate the best prompt on your test set, then deploy it as a static prompt.

**Q: Why MCTS instead of other search algorithms?**  
A: MCTS balances exploration (trying diverse prompts) and exploitation (refining good prompts). It's sample-efficient and handles large search spaces. But other algorithms (beam search, evolutionary methods, RL) could work too.

---

## Real-World Constraints: Style, Length, and Format

In production, you often need prompts that meet specific requirements beyond accuracy:


### Key Insight: Constraints Drive Business Value

In production:
- **Length limits** → Reduce API costs, improve latency, better UX
- **Style/tone** → Brand consistency, professionalism, compliance
- **Format** → Downstream integration, parsing reliability, automation
- **Content restrictions** → Legal compliance, safety, policy adherence

**The optimizer will find prompts that maximize quality WITHIN your constraints.**

---

## Contributing & Research Collaboration

This is open research. We welcome:

- **Empirical comparisons:** Run both algorithms on your tasks and share results
- **Reward function designs:** Share effective reward functions for different domains
- **Algorithmic improvements:** Better sampling strategies, gradient generation prompts
- **Theoretical analysis:** Convergence properties, sample complexity bounds
- **New search algorithms:** Beyond MCTS (beam search, evolutionary, RL-based)

**Reproducibility Commitment:** All configurations, prompts, and experimental setups are version-controlled. We aim for full transparency in how algorithms behave.

---

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{feedback_prompt_learning_2025,
  title={Feedback-Prompt-Learning: MCTS for Prompt Optimization with Rich Feedback},
  author={Oinar, Chingis},
  year={2025},
  url={https://github.com/cowana-ai/feedback-prompt-learning},
  note={Open-source research framework extending PromptAgent with feedback-driven optimization}
}
```

And please cite the original PromptAgent paper:

```bibtex
@article{wang2023promptagent,
  title={PromptAgent: Strategic Planning with Language Models Enables Expert-level Prompt Optimization},
  author={Wang, Xinyuan and Li, Chenxi and Wang, Zhen and Bai, Fan and Luo, Haotian and Zhang, Jiayou and Jojic, Nebojsa and Xing, Eric P and Hu, Zhiting},
  journal={arXiv preprint arXiv:2310.16427},
  year={2023}
}
```

---

## License

MIT License - See LICENSE file for details.

---

## Acknowledgments

This work builds on [PromptAgent](https://arxiv.org/abs/2310.16427) by Wang et al. We implement their approach faithfully and extend it with feedback mechanisms. Any improvements or limitations in our variant are our responsibility.

---

**Remember: Own your prompts. Define what you want. Let evaluation and feedback drive the system.**
