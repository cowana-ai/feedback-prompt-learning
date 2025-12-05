# Feedback-Prompt-Learning

> **Own your prompts. Define what you want. Let feedback drive the system.**

**Feedback-Prompt-Learning** is an evaluation-driven framework for automatic prompt optimization. Instead of guessing what makes a good prompt, you define clear evaluation criteria and let the system discover effective prompts through iterative feedback.

Inspired by PromptAgent, this framework extends prompt optimization with **rich textual feedback** in the reward function, enabling smarter gradient generation and faster convergence.

---

## Philosophy: Eval-Driven Development

**Stop prompt engineering by trial-and-error.** Start with rigorous evaluation:

1. **Define what you want** - Write a reward function that captures your actual objectives
2. **Own your data** - Curate train/eval/test splits that reflect real-world usage  
3. **Let feedback guide optimization** - The system learns from detailed feedback, not just scores
4. **Validate systematically** - Proper data separation ensures prompts generalize

This isn't about finding "the perfect prompt" through intuition. It's about **building a system** that measurably improves on your specific task.

## Key Innovation: Feedback-Driven Rewards

Traditional prompt optimization uses binary rewards (correct/incorrect). We inject **rich textual feedback** into the reward function:

```python
def reward_fn(output: str, target: str, prompt: str, raw_output: str = "") -> Tuple[float, str]:
    """
    Returns: (score, feedback)
    - score: quantitative reward (0.0 - 1.0)
    - feedback: qualitative analysis of what went wrong/right
    """
    if output == target:
        return 1.0, "Correct answer ✓"
    else:
        # Analyze WHY it failed
        issues = []
        if not has_structured_reasoning(raw_output):
            issues.append("Lacks step-by-step breakdown")
        if not uses_domain_knowledge(raw_output):
            issues.append("Missing key domain concepts")
        
        feedback = f"Wrong answer. Issues: {', '.join(issues)}"
        return 0.0, feedback
```

The feedback drives **gradient generation** - an LLM critic analyzes failures and suggests concrete prompt improvements.


## Core Principles

### 1. **Evaluation First**

Define rigorous evaluation *before* optimizing:

- **Reward function** - What does "good" mean for your task?
- **Data splits** - Train/eval/test that mirror production
- **Metrics** - Beyond accuracy (reasoning quality, format compliance, etc.)


Your data reflects your real-world distribution. Own it.

### 2. **Feedback > Scores**

Rich feedback enables smarter optimization.
---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    MCTS Prompt Optimizer                     │
│  (Algorithm-agnostic: MCTS, RL, Gradient Descent, etc.)     │
└─────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                 Reward Function (Your Logic)                 │
│  • Calculate score: correct/incorrect, quality metrics      │
│  • Generate feedback: what went wrong, what to improve      │
└─────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                   Feedback → Gradient                        │
│  • LLM critic analyzes failures + feedback                  │
│  • Generates concrete prompt improvements                   │
│  • Uses plan-and-solve prompting for systematic analysis    │
└─────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                   New Prompt Generation                      │
│  • Apply gradient (improve) or explore (variations)         │
│  • Adaptive: ascend mode (good prompts) vs descend (bad)    │
└─────────────────────────────────────────────────────────────┘
```

---

## FAQ

**Q: How is this different from prompt engineering?**  
A: Manual prompt engineering is trial-and-error. This is systematic optimization driven by quantitative evaluation and qualitative feedback.

**Q: What if my task doesn't have clear correct answers?**  
A: Define custom reward functions based on your criteria (coherence, relevance, style, etc.). Return feedback on how to improve.

**Q: How do I enforce constraints like specific style, tone, or length limits?**  
A: Build constraints directly into your reward function. The system will optimize prompts that satisfy both your quality objectives AND your constraints.

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


---

## License

MIT License - See LICENSE file for details.

---

**Remember: Own your prompts. Define what you want. Let evaluation drive the system.**
