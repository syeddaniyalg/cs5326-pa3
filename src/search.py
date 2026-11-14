import time
from typing import Any, Callable

from .generation import compute_majority_vote, run_self_consistency, sample_candidates
from .prm_judge import score_path

GenerateFn = Callable[[str, int, dict], list[dict]]
JudgeFn = Callable[[str], dict]

STRATEGIES = ("min", "product", "step_penalty")
PROBABILITY_STRATEGY = "product_logprob"
TIE_TOLERANCE = 1e-9


def _clamp_probability(score: float) -> float:
    return min(max((score + 1.0) / 2.0, 0.01), 0.99)


def aggregate_path_score(
    step_scores: list[float],
    strategy: str = "min",
    penalty_lambda: float = 2.0,
) -> float:
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown aggregation strategy: {strategy}")
    if not step_scores:
        return -1.0
    if strategy == "min":
        return float(min(step_scores))
    if strategy == "product":
        product = 1.0
        for score in step_scores:
            product *= _clamp_probability(score)
        return product
    invalid = sum(1 for score in step_scores if score == -1.0)
    return sum(step_scores) / len(step_scores) - penalty_lambda * invalid


def aggregate_probability_score(step_probs: list[float]) -> float:
    if not step_probs:
        return -1.0
    product = 1.0
    for probability in step_probs:
        product *= probability
    return product


def candidate_score(candidate: dict[str, Any], strategy: str, penalty_lambda: float) -> float:
    if strategy == PROBABILITY_STRATEGY:
        return aggregate_probability_score(candidate.get("step_probs") or [])
    return aggregate_path_score(candidate.get("step_scores") or [], strategy, penalty_lambda)


def select_best_path(
    candidate_paths: list[dict[str, Any]],
    aggregation_strategy: str = "min",
    penalty_lambda: float = 2.0,
) -> dict[str, Any]:
    if not candidate_paths:
        raise ValueError("candidate_paths must not be empty")
    scores = [candidate_score(c, aggregation_strategy, penalty_lambda) for c in candidate_paths]
    eligible = [i for i, c in enumerate(candidate_paths) if c.get("answer") is not None]
    if not eligible:
        eligible = list(range(len(candidate_paths)))
    best = max(scores[i] for i in eligible)
    tied = [i for i in eligible if scores[i] >= best - TIE_TOLERANCE]
    chosen = tied[0]
    if len(tied) > 1:
        vote = compute_majority_vote([candidate_paths[i].get("answer") for i in tied])
        if vote is not None:
            chosen = next(i for i in tied if candidate_paths[i].get("answer") == vote)
    selected = dict(candidate_paths[chosen])
    selected["path_score"] = float(scores[chosen])
    selected["num_tied"] = len(tied)
    return selected


def run_prm_best_of_n(
    model: Any = None,
    judge_model: Any = None,
    gen_tokenizer: Any = None,
    judge_tokenizer: Any = None,
    question: str = "",
    n: int = 5,
    aggregation_strategy: str = "min",
    penalty_lambda: float = 2.0,
    candidates: list[dict[str, Any]] | None = None,
    generate_fn: GenerateFn | None = None,
    judge_fn: JudgeFn | None = None,
) -> dict[str, Any]:
    start = time.perf_counter()
    generation_calls = 0
    generated_tokens = 0
    if candidates is None:
        sampled = run_self_consistency(model, gen_tokenizer, question, n=n, generate_fn=generate_fn)
        pool = sampled["candidates"]
        generation_calls = sampled["generation_calls"]
        generated_tokens = sampled["generated_tokens"]
    else:
        pool = candidates
    scored = []
    judge_calls = 0
    judge_tokens = 0
    judge_prompt_tokens = 0
    for candidate in pool:
        result = score_path(judge_model, judge_tokenizer, question, candidate, judge_fn)
        augmented = dict(candidate)
        augmented["step_scores"] = result["step_scores"]
        augmented["rationales"] = result["rationales"]
        if "step_probs" in result:
            augmented["step_probs"] = result["step_probs"]
        scored.append(augmented)
        judge_calls += result["judge_calls"]
        judge_tokens += result["judge_tokens"]
        judge_prompt_tokens += result["judge_prompt_tokens"]
    selected = select_best_path(scored, aggregation_strategy, penalty_lambda)
    return {
        "selected": selected,
        "answer": selected["answer"],
        "candidates": scored,
        "generation_calls": generation_calls,
        "judge_calls": judge_calls,
        "generated_tokens": generated_tokens,
        "judge_tokens": judge_tokens,
        "judge_prompt_tokens": judge_prompt_tokens,
        "time_seconds": time.perf_counter() - start,
    }


def run_compute_matched_self_consistency(
    model: Any = None,
    tokenizer: Any = None,
    question: str = "",
    target_calls: int = 15,
    temperature: float = 0.7,
    existing_candidates: list[dict[str, Any]] | None = None,
    generate_fn: GenerateFn | None = None,
) -> dict[str, Any]:
    start = time.perf_counter()
    existing = list(existing_candidates or [])
    deficit = max(0, target_calls - len(existing))
    new = sample_candidates(model, tokenizer, question, deficit, temperature, generate_fn)
    candidates = existing + new
    return {
        "candidates": candidates,
        "answer": compute_majority_vote([c.get("answer") for c in candidates]),
        "generation_calls": len(new),
        "total_samples": len(candidates),
        "generated_tokens": sum(c["num_tokens"] for c in new),
        "time_seconds": time.perf_counter() - start,
    }
