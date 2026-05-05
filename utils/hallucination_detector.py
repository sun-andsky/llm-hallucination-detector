"""
Hallucination detection utilities using text similarity and keyword overlap.
"""

import re
import math
from collections import Counter


def preprocess(text: str) -> str:
    """Lowercase and remove punctuation."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text


def tokenize(text: str) -> list:
    """Simple whitespace tokenizer."""
    return preprocess(text).split()


def get_ngrams(tokens: list, n: int) -> Counter:
    """Return Counter of n-grams from token list."""
    return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))


def bleu_score(reference: str, hypothesis: str, max_n: int = 2) -> float:
    """
    Simplified BLEU score (1-gram + 2-gram precision with brevity penalty).
    """
    ref_tokens = tokenize(reference)
    hyp_tokens = tokenize(hypothesis)

    if len(hyp_tokens) == 0:
        return 0.0

    precisions = []
    for n in range(1, max_n + 1):
        ref_ngrams = get_ngrams(ref_tokens, n)
        hyp_ngrams = get_ngrams(hyp_tokens, n)

        if not hyp_ngrams:
            precisions.append(0.0)
            continue

        clipped = sum(min(count, ref_ngrams[gram]) for gram, count in hyp_ngrams.items())
        total = sum(hyp_ngrams.values())
        precisions.append(clipped / total if total > 0 else 0.0)

    if any(p == 0 for p in precisions):
        return 0.0

    log_avg = sum(math.log(p) for p in precisions) / len(precisions)
    bp = min(1.0, math.exp(1 - len(ref_tokens) / max(len(hyp_tokens), 1)))
    return round(bp * math.exp(log_avg), 4)


def keyword_overlap_score(reference: str, hypothesis: str) -> float:
    """
    Measures the proportion of key reference tokens found in the hypothesis.
    Filters out stopwords for better quality signal.
    """
    STOPWORDS = {
        "the", "a", "an", "is", "it", "of", "to", "in", "and", "or", "that",
        "this", "are", "was", "were", "be", "been", "being", "have", "has",
        "had", "do", "does", "did", "will", "would", "could", "should", "may",
        "might", "can", "for", "on", "at", "by", "with", "from", "as", "into",
        "through", "during", "before", "after", "above", "below", "between",
        "each", "all", "both", "few", "more", "most", "other", "some", "such",
        "than", "then", "its", "their", "our", "your", "his", "her", "we",
        "they", "he", "she", "i", "you", "not", "no", "so", "if", "when"
    }

    ref_tokens = set(tokenize(reference)) - STOPWORDS
    hyp_tokens = set(tokenize(hypothesis)) - STOPWORDS

    if not ref_tokens:
        return 0.0

    overlap = ref_tokens & hyp_tokens
    return round(len(overlap) / len(ref_tokens), 4)


def jaccard_similarity(reference: str, hypothesis: str) -> float:
    """Token-level Jaccard similarity between two texts."""
    ref_set = set(tokenize(reference))
    hyp_set = set(tokenize(hypothesis))
    union = ref_set | hyp_set
    if not union:
        return 0.0
    return round(len(ref_set & hyp_set) / len(union), 4)


def length_ratio(reference: str, hypothesis: str) -> float:
    """Ratio of hypothesis length to reference length."""
    ref_len = len(tokenize(reference))
    hyp_len = len(tokenize(hypothesis))
    if ref_len == 0:
        return 0.0
    return round(hyp_len / ref_len, 4)


def compute_hallucination_score(reference: str, hypothesis: str) -> dict:
    """
    Compute composite hallucination detection score.
    Returns scores and final hallucination classification.

    Composite score = weighted combination of BLEU, keyword overlap, Jaccard.
    Lower composite → more likely hallucinated.
    """
    if not hypothesis or len(hypothesis.strip()) < 5:
        return {
            "bleu": 0.0,
            "keyword_overlap": 0.0,
            "jaccard": 0.0,
            "length_ratio": 0.0,
            "composite_score": 0.0,
            "is_hallucinated": True,
            "confidence": "High",
            "verdict": "HALLUCINATED"
        }

    bleu = bleu_score(reference, hypothesis)
    kw = keyword_overlap_score(reference, hypothesis)
    jac = jaccard_similarity(reference, hypothesis)
    lr = length_ratio(reference, hypothesis)

    # Weighted composite: keyword overlap matters most for factual QA
    composite = round(0.35 * kw + 0.35 * bleu + 0.30 * jac, 4)

    # Thresholds determined empirically
    if composite >= 0.30:
        is_hallucinated = False
        confidence = "High" if composite >= 0.45 else "Medium"
        verdict = "CORRECT"
    elif composite >= 0.18:
        is_hallucinated = False
        confidence = "Low"
        verdict = "LIKELY CORRECT"
    elif composite >= 0.10:
        is_hallucinated = True
        confidence = "Low"
        verdict = "LIKELY HALLUCINATED"
    else:
        is_hallucinated = True
        confidence = "High" if composite < 0.05 else "Medium"
        verdict = "HALLUCINATED"

    return {
        "bleu": bleu,
        "keyword_overlap": kw,
        "jaccard": jac,
        "length_ratio": lr,
        "composite_score": composite,
        "is_hallucinated": is_hallucinated,
        "confidence": confidence,
        "verdict": verdict
    }


def evaluate_batch(results: list) -> dict:
    """
    Compute aggregate evaluation metrics for a batch of results.

    Args:
        results: List of dicts with keys 'model', 'prompt_style', 'scores'

    Returns:
        Summary dict with per-model and per-prompt accuracy + hallucination rate
    """
    summary = {}

    for r in results:
        key = f"{r['model']}|{r['prompt_style']}"
        if key not in summary:
            summary[key] = {
                "model": r["model"],
                "prompt_style": r["prompt_style"],
                "total": 0,
                "correct": 0,
                "hallucinated": 0,
                "composite_scores": [],
                "response_times": []
            }
        s = summary[key]
        s["total"] += 1
        scores = r.get("scores", {})
        if not scores.get("is_hallucinated", True):
            s["correct"] += 1
        else:
            s["hallucinated"] += 1
        s["composite_scores"].append(scores.get("composite_score", 0))
        s["response_times"].append(r.get("response_time", 0))

    # Compute derived metrics
    for key, s in summary.items():
        n = s["total"]
        s["accuracy"] = round(s["correct"] / n, 4) if n else 0
        s["hallucination_rate"] = round(s["hallucinated"] / n, 4) if n else 0
        s["avg_composite"] = round(sum(s["composite_scores"]) / n, 4) if n else 0
        s["avg_response_time"] = round(sum(s["response_times"]) / n, 3) if n else 0

    return summary
