import math
import random
import time
from typing import Any

import numpy as np

from .generation import compute_majority_vote
from .search import PROBABILITY_STRATEGY, select_best_path

POOL_SIZE = 5
TARGET_CALLS = 15
DEFAULT_LAMBDA = 2.0
BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_SEED = 5326
VARIANTS = [("min", None), ("product", None), ("step_penalty", 0.0), ("step_penalty", 0.5), ("step_penalty", 2.0)]
PRM_KEYS = ["min", "product", "step_penalty_l2.0"]
METHOD_KEYS = ["greedy", "sc5", "sc15"] + PRM_KEYS
METHOD_NAMES = {
    "greedy": "Greedy (N=1)",
    "sc5": "Self-Consistency (N=5)",
    "sc15": "Compute-Matched SC (N=15)",
    "min": "PRM Best-of-5 (min)",
    "product": "PRM Best-of-5 (product)",
    "step_penalty_l2.0": "PRM Best-of-5 (step_penalty, lambda=2.0)",
}


def variant_key(strategy: str, penalty_lambda: float | None) -> str:
    return strategy if penalty_lambda is None else f"{strategy}_l{penalty_lambda}"


def is_correct(answer: int | None, ground_truth: int) -> bool:
    return answer is not None and answer == ground_truth


def build_problem_record(gen: dict[str, Any], judge: dict[str, Any]) -> dict[str, Any]:
    truth = gen["ground_truth_answer"]
    probabilities = judge.get("step_probs")
    candidates = []
    for index, source in enumerate(gen["candidates"]):
        item = {
            "index": index,
            "completion": source["completion"],
            "steps": source["steps"],
            "answer": source["answer"],
            "correct": is_correct(source["answer"], truth),
            "num_tokens": source["num_tokens"],
        }
        if index < POOL_SIZE:
            item["step_scores"] = judge["step_scores"][index]
            item["rationales"] = judge["rationales"][index]
            if probabilities:
                item["step_probs"] = probabilities[index]
        candidates.append(item)

    curve_answers = [compute_majority_vote([c["answer"] for c in candidates[:n]]) for n in range(1, len(candidates) + 1)]
    answers = {
        "greedy": gen["greedy"]["answer"],
        "sc5": curve_answers[POOL_SIZE - 1],
        "sc15": curve_answers[TARGET_CALLS - 1],
    }

    pool = candidates[:POOL_SIZE]
    variants = list(VARIANTS)
    if probabilities:
        variants.append((PROBABILITY_STRATEGY, None))
    prm: dict[str, Any] = {}
    selection_seconds: dict[str, float] = {}
    for strategy, penalty_lambda in variants:
        key = variant_key(strategy, penalty_lambda)
        start = time.perf_counter()
        selected = select_best_path(pool, strategy, DEFAULT_LAMBDA if penalty_lambda is None else penalty_lambda)
        selection_seconds[key] = time.perf_counter() - start
        prm[key] = {
            "answer": selected["answer"],
            "correct": is_correct(selected["answer"], truth),
            "selected_index": selected["index"],
            "path_score": selected["path_score"],
            "num_tied": selected["num_tied"],
        }
        answers[key] = selected["answer"]

    correct = {key: is_correct(answer, truth) for key, answer in answers.items()}
    return {
        "id": gen["id"],
        "index": gen["index"],
        "question": gen["question"],
        "ground_truth_answer": truth,
        "greedy": {
            "prompt": gen["greedy"]["prompt"],
            "completion": gen["greedy"]["completion"],
            "steps": gen["greedy"]["steps"],
            "answer": gen["greedy"]["answer"],
            "correct": correct["greedy"],
            "generated_tokens": gen["greedy"]["generated_tokens"],
        },
        "candidates": candidates,
        "sc_curve_answers": curve_answers,
        "answers": answers,
        "correct": correct,
        "prm": prm,
        "tokens": {
            "greedy": gen["greedy"]["generated_tokens"],
            "pool": sum(c["num_tokens"] for c in pool),
            "all": sum(c["num_tokens"] for c in candidates),
            "curve": [sum(c["num_tokens"] for c in candidates[:n]) for n in range(1, len(candidates) + 1)],
            "judge": judge["judge_tokens"],
            "judge_prompt": judge.get("judge_prompt_tokens", 0),
        },
        "calls": {"judge": judge["judge_calls"]},
        "seconds": {
            "greedy": gen["greedy"]["time_seconds"],
            "sc5": gen["sc5_time_seconds"],
            "sc15_extra": gen["sc15_extra_time_seconds"],
            "judge": judge["time_seconds"],
            "selection": selection_seconds,
        },
    }


def percent(values: list[bool] | np.ndarray) -> float:
    return float(np.mean(np.asarray(values, dtype=float)) * 100.0) if len(values) else 0.0


def bootstrap_indices(size: int, resamples: int = BOOTSTRAP_RESAMPLES, seed: int = BOOTSTRAP_SEED) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, size, size=(resamples, size))


def bootstrap_ci(values: np.ndarray, indices: np.ndarray) -> dict[str, float]:
    samples = values[indices].mean(axis=1) * 100.0
    low, high = np.percentile(samples, [2.5, 97.5])
    return {"accuracy": float(values.mean() * 100.0), "ci_low": float(low), "ci_high": float(high)}


def exact_sign_test(wins: int, losses: int) -> float:
    total = wins + losses
    if total == 0:
        return 1.0
    tail = sum(math.comb(total, k) for k in range(0, min(wins, losses) + 1)) / (2**total)
    return float(min(1.0, 2.0 * tail))


def paired_comparison(first: np.ndarray, second: np.ndarray, indices: np.ndarray) -> dict[str, Any]:
    wins = int(np.sum(first & ~second))
    losses = int(np.sum(~first & second))
    both_correct = int(np.sum(first & second))
    both_wrong = int(np.sum(~first & ~second))
    difference = (first.astype(float)[indices] - second.astype(float)[indices]).mean(axis=1) * 100.0
    low, high = np.percentile(difference, [2.5, 97.5])
    return {
        "first_wins": wins,
        "second_wins": losses,
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "agreed": both_correct + both_wrong,
        "accuracy_difference": float((first.mean() - second.mean()) * 100.0),
        "difference_ci_low": float(low),
        "difference_ci_high": float(high),
        "sign_test_p_value": exact_sign_test(wins, losses),
        "ci_excludes_zero": bool(low > 0.0 or high < 0.0),
    }


def results_summary(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    count = len(records)
    judge_calls = sum(r["calls"]["judge"] for r in records)
    judge_tokens = sum(r["tokens"]["judge"] for r in records)
    sc5_seconds = sum(r["seconds"]["sc5"] for r in records)
    judge_seconds = sum(r["seconds"]["judge"] for r in records)
    pool_tokens = sum(r["tokens"]["pool"] for r in records)

    def row(key: str, gen_calls: int, judge: int, gen_tokens: int, judge_tok: int, seconds: float) -> dict[str, Any]:
        return {
            "Method": METHOD_NAMES[key],
            "Accuracy (%)": round(percent([r["correct"][key] for r in records]), 2),
            "Generation Calls": gen_calls,
            "Judge Calls": judge,
            "Total LLM Calls": gen_calls + judge,
            "Avg Gen Tokens / Problem": round(gen_tokens / count, 2),
            "Avg Judge Tokens / Problem": round(judge_tok / count, 2),
            "Total Time (s)": round(seconds, 2),
        }

    rows = [
        row("greedy", count, 0, sum(r["tokens"]["greedy"] for r in records), 0, sum(r["seconds"]["greedy"] for r in records)),
        row("sc5", POOL_SIZE * count, 0, pool_tokens, 0, sc5_seconds),
        row(
            "sc15",
            TARGET_CALLS * count,
            0,
            sum(r["tokens"]["all"] for r in records),
            0,
            sc5_seconds + sum(r["seconds"]["sc15_extra"] for r in records),
        ),
    ]
    for key in PRM_KEYS:
        selection = sum(r["seconds"]["selection"][key] for r in records)
        rows.append(row(key, POOL_SIZE * count, judge_calls, pool_tokens, judge_tokens, sc5_seconds + judge_seconds + selection))
    return rows


def aggregation_ablation(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for strategy, penalty_lambda in VARIANTS:
        key = variant_key(strategy, penalty_lambda)
        rows.append(
            {
                "Strategy": strategy,
                "Lambda": penalty_lambda,
                "Accuracy (%)": round(percent([r["prm"][key]["correct"] for r in records]), 2),
                "Tie Rate (%)": round(percent([r["prm"][key]["num_tied"] > 1 for r in records]), 2),
            }
        )
    return rows


def sc_scaling_curve(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    count = len(records)
    rows = []
    for n in range(1, TARGET_CALLS + 1):
        correct = [is_correct(r["sc_curve_answers"][n - 1], r["ground_truth_answer"]) for r in records]
        tokens = sum(r["tokens"]["curve"][n - 1] for r in records)
        rows.append(
            {"N": n, "Accuracy (%)": round(percent(correct), 2), "Avg Gen Tokens / Problem": round(tokens / count, 2)}
        )
    return rows


def probability_bonus(records: list[dict[str, Any]]) -> dict[str, Any]:
    key = PROBABILITY_STRATEGY
    available = bool(records) and all(key in r["prm"] for r in records)
    if not available:
        return {"available": False}
    return {
        "available": True,
        "strategy": key,
        "approach": "first-token logits at the verdict position: p = sigmoid(logsumexp(z_VALID) - logsumexp(z_INVALID))",
        "accuracy": round(percent([r["prm"][key]["correct"] for r in records]), 2),
        "tie_rate": round(percent([r["prm"][key]["num_tied"] > 1 for r in records]), 2),
        "min_accuracy": round(percent([r["prm"]["min"]["correct"] for r in records]), 2),
        "product_accuracy": round(percent([r["prm"]["product"]["correct"] for r in records]), 2),
    }


def budget_analysis(records: list[dict[str, Any]], curve: list[dict[str, Any]]) -> dict[str, Any]:
    greedy_tokens = sum(r["tokens"]["greedy"] for r in records)
    budget = 10 * greedy_tokens
    prm_tokens = sum(r["tokens"]["pool"] + r["tokens"]["judge"] for r in records)
    curve_tokens = [
        {"N": row["N"], "total_tokens": sum(r["tokens"]["curve"][row["N"] - 1] for r in records), "accuracy": row["Accuracy (%)"]}
        for row in curve
    ]
    within = [item for item in curve_tokens if item["total_tokens"] <= budget]
    return {
        "greedy_generated_tokens": greedy_tokens,
        "budget_tokens": budget,
        "prm_total_tokens": prm_tokens,
        "prm_budget_fraction": prm_tokens / budget if budget else None,
        "prm_within_budget": prm_tokens <= budget,
        "sc_curve_tokens": curve_tokens,
        "largest_sc_n_within_budget": within[-1]["N"] if within else None,
        "sc_accuracy_at_largest_n_within_budget": within[-1]["accuracy"] if within else None,
        "sc_max_n_total_tokens": curve_tokens[-1]["total_tokens"],
        "sc_max_n_budget_fraction": curve_tokens[-1]["total_tokens"] / budget if budget else None,
    }


def cost_asymmetry(records: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(records)
    judge_calls = sum(r["calls"]["judge"] for r in records)
    judge_tokens = sum(r["tokens"]["judge"] for r in records)
    judge_prompt = sum(r["tokens"]["judge_prompt"] for r in records)
    pool_tokens = sum(r["tokens"]["pool"] for r in records)
    gen_calls = POOL_SIZE * count
    sc5_total = pool_tokens
    prm_total = pool_tokens + judge_tokens
    return {
        "generation_calls": gen_calls,
        "judge_calls": judge_calls,
        "mean_steps_per_path": judge_calls / gen_calls if gen_calls else 0.0,
        "judge_to_generation_call_ratio": judge_calls / gen_calls if gen_calls else None,
        "avg_output_tokens_per_generation_call": pool_tokens / gen_calls if gen_calls else None,
        "avg_output_tokens_per_judge_call": judge_tokens / judge_calls if judge_calls else None,
        "avg_prompt_tokens_per_judge_call": judge_prompt / judge_calls if judge_calls else None,
        "generated_tokens_total": pool_tokens,
        "judge_tokens_total": judge_tokens,
        "judge_prompt_tokens_total": judge_prompt,
        "prm_to_sc5_output_token_ratio": prm_total / sc5_total if sc5_total else None,
        "prm_to_sc5_call_ratio": (gen_calls + judge_calls) / gen_calls if gen_calls else None,
    }


def judge_statistics(records: list[dict[str, Any]]) -> dict[str, Any]:
    steps = invalid = 0
    candidates = all_valid = 0
    correct_all_valid = correct_with_invalid = with_invalid = 0
    valid_probs: list[float] = []
    invalid_probs: list[float] = []
    for record in records:
        for candidate in record["candidates"][:POOL_SIZE]:
            scores = candidate["step_scores"]
            probs = candidate.get("step_probs") or []
            steps += len(scores)
            invalid += sum(1 for s in scores if s < 0)
            for score, prob in zip(scores, probs):
                (valid_probs if score > 0 else invalid_probs).append(prob)
            candidates += 1
            clean = bool(scores) and all(s > 0 for s in scores)
            if clean:
                all_valid += 1
                correct_all_valid += int(candidate["correct"])
            else:
                with_invalid += 1
                correct_with_invalid += int(candidate["correct"])
    return {
        "judged_steps": steps,
        "invalid_steps": invalid,
        "invalid_step_rate": invalid / steps if steps else 0.0,
        "pool_candidates": candidates,
        "candidates_all_steps_valid": all_valid,
        "candidates_with_invalid_step": with_invalid,
        "accuracy_all_steps_valid": correct_all_valid / all_valid if all_valid else None,
        "accuracy_with_invalid_step": correct_with_invalid / with_invalid if with_invalid else None,
        "mean_p_valid_when_valid": float(np.mean(valid_probs)) if valid_probs else None,
        "mean_p_valid_when_invalid": float(np.mean(invalid_probs)) if invalid_probs else None,
    }


def candidate_statistics(records: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(records)
    pool = [c for r in records for c in r["candidates"][:POOL_SIZE]]
    everything = [c for r in records for c in r["candidates"]]
    greedy_parsed = sum(1 for r in records if r["greedy"]["answer"] is not None)
    greedy_has_steps = sum(1 for r in records if r["greedy"]["steps"])
    greedy_marker = sum(1 for r in records if "final answer" in r["greedy"]["completion"].lower())
    return {
        "greedy_answer_parse_rate": greedy_parsed / count,
        "greedy_final_answer_marker_rate": greedy_marker / count,
        "greedy_has_steps_rate": greedy_has_steps / count,
        "pool_answer_parse_rate": sum(1 for c in pool if c["answer"] is not None) / len(pool),
        "pool_candidate_accuracy": sum(1 for c in pool if c["correct"]) / len(pool),
        "all_candidate_accuracy": sum(1 for c in everything if c["correct"]) / len(everything),
        "pass_at_5": sum(1 for r in records if any(c["correct"] for c in r["candidates"][:POOL_SIZE])) / count,
        "pass_at_15": sum(1 for r in records if any(c["correct"] for c in r["candidates"])) / count,
        "problems_all_pool_answers_agree": sum(1 for r in records if len({c["answer"] for c in r["candidates"][:POOL_SIZE]}) == 1),
        "mean_tied_candidates_min": float(np.mean([r["prm"]["min"]["num_tied"] for r in records])),
    }


def analysis(records: list[dict[str, Any]], curve: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(records)
    indices = bootstrap_indices(count)
    vectors = {key: np.array([r["correct"][key] for r in records], dtype=bool) for key in records[0]["correct"]}
    intervals = {METHOD_NAMES[key]: bootstrap_ci(vectors[key].astype(float), indices) for key in METHOD_KEYS}
    curve_intervals = []
    for n in range(1, TARGET_CALLS + 1):
        correct = np.array([is_correct(r["sc_curve_answers"][n - 1], r["ground_truth_answer"]) for r in records], dtype=float)
        curve_intervals.append({"N": n, **bootstrap_ci(correct, indices)})
    paired = {}
    for key in PRM_KEYS:
        for baseline in ("greedy", "sc5", "sc15"):
            paired[f"{METHOD_NAMES[key]} vs {METHOD_NAMES[baseline]}"] = paired_comparison(vectors[key], vectors[baseline], indices)
    paired["PRM Best-of-5 (min) vs PRM Best-of-5 (step_penalty, lambda=2.0)"] = paired_comparison(
        vectors["min"], vectors["step_penalty_l2.0"], indices
    )
    return {
        "problems": count,
        "bootstrap": {"resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED, "confidence": 0.95},
        "accuracy_confidence_intervals": intervals,
        "sc_curve_confidence_intervals": curve_intervals,
        "paired_comparisons": paired,
        "budget": budget_analysis(records, curve),
        "cost_asymmetry": cost_asymmetry(records),
        "judge": judge_statistics(records),
        "candidates": candidate_statistics(records),
        "bonus_product_logprob": probability_bonus(records),
    }


def find_failure_cases(records: list[dict[str, Any]], limit: int = 5) -> dict[str, list[dict[str, Any]]]:
    def describe(record: dict[str, Any], candidate: dict[str, Any] | None = None) -> dict[str, Any]:
        info = {"id": record["id"], "question": record["question"], "ground_truth_answer": record["ground_truth_answer"]}
        if candidate is not None:
            info.update(
                {
                    "candidate_index": candidate["index"],
                    "answer": candidate["answer"],
                    "steps": candidate["steps"],
                    "step_scores": candidate["step_scores"],
                    "rationales": candidate["rationales"],
                }
            )
        return info

    false_positive, false_negative, shared, helped, hurt, unrecoverable = [], [], [], [], [], []
    for record in records:
        selected = record["prm"]["min"]["selected_index"]
        pool = record["candidates"][:POOL_SIZE]
        for candidate in pool:
            scores = candidate["step_scores"]
            if not candidate["correct"] and scores and all(s > 0 for s in scores):
                entry = describe(record, candidate)
                entry["selected_by_prm_min"] = candidate["index"] == selected
                false_positive.append(entry)
            if candidate["correct"] and any(s < 0 for s in scores):
                entry = describe(record, candidate)
                entry["invalid_step_indices"] = [i for i, s in enumerate(scores) if s < 0]
                false_negative.append(entry)
        answers = [c["answer"] for c in pool]
        top = max(set(answers), key=answers.count)
        if top is not None and top != record["ground_truth_answer"] and answers.count(top) >= 3:
            first = next(c for c in pool if c["answer"] == top)
            entry = describe(record, first)
            entry["shared_wrong_answer"] = top
            entry["agreeing_candidates"] = answers.count(top)
            entry["pool_answers"] = answers
            shared.append(entry)
        if record["correct"]["min"] and not record["correct"]["sc5"]:
            helped.append({**describe(record), "pool_answers": answers, "prm_answer": record["answers"]["min"], "sc5_answer": record["answers"]["sc5"]})
        if record["correct"]["sc5"] and not record["correct"]["min"]:
            hurt.append({**describe(record), "pool_answers": answers, "prm_answer": record["answers"]["min"], "sc5_answer": record["answers"]["sc5"]})
        if not any(c["correct"] for c in record["candidates"]):
            unrecoverable.append({**describe(record), "all_answers": [c["answer"] for c in record["candidates"]]})
    false_positive.sort(key=lambda e: (not e["selected_by_prm_min"], len(e["steps"])))
    shared.sort(key=lambda e: -e["agreeing_candidates"])
    return {
        "judge_false_positive_candidates": false_positive[:limit],
        "judge_false_negative_candidates": false_negative[:limit],
        "shared_flawed_premise_candidates": shared[:limit],
        "prm_helped_over_sc5": helped[:limit],
        "prm_hurt_vs_sc5": hurt[:limit],
        "no_correct_candidate_among_15": unrecoverable[:limit],
        "counts": {
            "judge_false_positive_candidates": len(false_positive),
            "judge_false_negative_candidates": len(false_negative),
            "shared_flawed_premise_candidates": len(shared),
            "prm_helped_over_sc5": len(helped),
            "prm_hurt_vs_sc5": len(hurt),
            "no_correct_candidate_among_15": len(unrecoverable),
        },
    }


def sample_audit_steps(
    records: list[dict[str, Any]], invalid_target: int = 20, valid_target: int = 20, seed: int = BOOTSTRAP_SEED
) -> list[dict[str, Any]]:
    invalid_steps, valid_steps = [], []
    for record in records:
        for candidate in record["candidates"][:POOL_SIZE]:
            for position, score in enumerate(candidate["step_scores"]):
                row = {
                    "problem_id": record["id"],
                    "candidate_index": candidate["index"],
                    "step_index": position + 1,
                    "question": record["question"],
                    "preceding_steps": " | ".join(f"Step {i}: {s}" for i, s in enumerate(candidate["steps"][:position], start=1)),
                    "step": candidate["steps"][position],
                    "judge_verdict": "VALID" if score > 0 else "INVALID",
                    "judge_rationale": candidate["rationales"][position],
                    "candidate_answer": candidate["answer"],
                    "ground_truth_answer": record["ground_truth_answer"],
                    "human_label": "",
                }
                (valid_steps if score > 0 else invalid_steps).append(row)
    rng = random.Random(seed)
    chosen_invalid = rng.sample(invalid_steps, min(invalid_target, len(invalid_steps)))
    chosen_valid = rng.sample(valid_steps, min(valid_target, len(valid_steps)))
    chosen = chosen_invalid + chosen_valid
    chosen.sort(key=lambda row: (row["problem_id"], row["candidate_index"], row["step_index"]))
    for audit_id, row in enumerate(chosen, start=1):
        row["audit_id"] = audit_id
    return chosen


def audit_confusion(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tp = fp = tn = fn = 0
    for row in rows:
        human = str(row.get("human_label", "")).strip().upper()
        judge = str(row.get("judge_verdict", "")).strip().upper()
        if human not in ("VALID", "INVALID") or judge not in ("VALID", "INVALID"):
            continue
        if judge == "INVALID" and human == "INVALID":
            tp += 1
        elif judge == "INVALID" and human == "VALID":
            fp += 1
        elif judge == "VALID" and human == "VALID":
            tn += 1
        else:
            fn += 1
    labelled = tp + fp + tn + fn
    return {
        "labelled_steps": labelled,
        "judge_invalid_steps_labelled": tp + fp,
        "meets_minimum_30_steps": labelled >= 30,
        "meets_minimum_10_judge_invalid": tp + fp >= 10,
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "precision_invalid": tp / (tp + fp) if tp + fp else None,
        "recall_invalid": tp / (tp + fn) if tp + fn else None,
    }
