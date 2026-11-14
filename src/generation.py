import time
from typing import Any, Callable

from .extraction import extract_answer, parse_steps
from .utils import GREEDY_GEN_KWARGS, SAMPLING_GEN_KWARGS, make_hf_generate_fn

GenerateFn = Callable[[str, int, dict], list[dict]]

PROMPT_TEMPLATE = (
    "Solve the following math problem step by step.\n\n"
    "Output rules:\n"
    "- Write each reasoning step on its own line, labelled 'Step 1:', 'Step 2:', and so on.\n"
    "- Each step is one short deduction or one arithmetic operation, with the numbers written out.\n"
    "- Use plain text only. Do not use markdown, headings, bold text, LaTeX or \\boxed{}.\n"
    "- After the last step, write one final line in the form 'Final Answer: <integer>'.\n"
    "- The final answer is a single integer: digits only, with no units, no commas and no extra words.\n\n"
    "Format:\n"
    "Step 1: <first deduction or arithmetic operation>\n"
    "Step 2: <next deduction that builds on Step 1>\n"
    "...\n"
    "Step K: <final calculation>\n"
    "Final Answer: <integer>\n\n"
    "Problem: {question}"
)


def format_step_prompt(question: str) -> str:
    return PROMPT_TEMPLATE.replace("{question}", question)


def compute_majority_vote(answers: list[int | None]) -> int | None:
    counts: dict[int, int] = {}
    for answer in answers:
        if answer is not None:
            counts[answer] = counts.get(answer, 0) + 1
    if not counts:
        return None
    best = max(counts.values())
    for answer, count in counts.items():
        if count == best:
            return answer
    return None


def build_candidate(output: dict) -> dict[str, Any]:
    completion = output["text"]
    return {
        "completion": completion,
        "steps": parse_steps(completion),
        "answer": extract_answer(completion),
        "num_tokens": int(output["num_tokens"]),
    }


def sample_candidates(
    model: Any,
    tokenizer: Any,
    question: str,
    n: int,
    temperature: float = 0.7,
    generate_fn: GenerateFn | None = None,
) -> list[dict[str, Any]]:
    if n <= 0:
        return []
    generate = generate_fn or make_hf_generate_fn(model, tokenizer)
    kwargs = dict(SAMPLING_GEN_KWARGS)
    kwargs["temperature"] = temperature
    outputs = generate(format_step_prompt(question), n, kwargs)
    return [build_candidate(output) for output in outputs]


def run_greedy(
    model: Any = None,
    tokenizer: Any = None,
    question: str = "",
    generate_fn: GenerateFn | None = None,
) -> dict[str, Any]:
    start = time.perf_counter()
    generate = generate_fn or make_hf_generate_fn(model, tokenizer)
    prompt = format_step_prompt(question)
    output = generate(prompt, 1, dict(GREEDY_GEN_KWARGS))[0]
    candidate = build_candidate(output)
    return {
        "prompt": prompt,
        "completion": candidate["completion"],
        "steps": candidate["steps"],
        "answer": candidate["answer"],
        "generation_calls": 1,
        "generated_tokens": candidate["num_tokens"],
        "time_seconds": time.perf_counter() - start,
    }


def run_self_consistency(
    model: Any = None,
    tokenizer: Any = None,
    question: str = "",
    n: int = 5,
    temperature: float = 0.7,
    generate_fn: GenerateFn | None = None,
) -> dict[str, Any]:
    start = time.perf_counter()
    candidates = sample_candidates(model, tokenizer, question, n, temperature, generate_fn)
    return {
        "candidates": candidates,
        "answer": compute_majority_vote([c["answer"] for c in candidates]),
        "generation_calls": len(candidates),
        "generated_tokens": sum(c["num_tokens"] for c in candidates),
        "time_seconds": time.perf_counter() - start,
    }
