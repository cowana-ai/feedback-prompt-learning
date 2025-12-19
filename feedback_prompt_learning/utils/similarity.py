"""
Similarity utilities for text analysis in prompt optimization.
"""

def ngram_set(text: str, n: int = 2) -> set:
    """Generate n-gram set from text."""
    tokens = text.lower().split()
    return {tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)}

def jaccard_ngram(a: str, b: str, n: int = 2) -> float:
    """Compute Jaccard similarity for n-grams between two texts."""
    set_a = ngram_set(a, n)
    set_b = ngram_set(b, n)
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union) if union else 1.0
