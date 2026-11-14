import re
from typing import Any, Callable

from .utils import make_hf_judge_fn

JudgeFn = Callable[[str], dict]

JUDGE_TEMPLATE = (
    "You are a strict verifier of grade-school math solutions. "
    "You check one reasoning step at a time.\n\n"
    "Problem:\n{question}\n\n"
    "Preceding steps:\n{preceding}\n\n"
    "Current Step: {current}\n\n"
    "Judge only the local validity of the current step: assume all preceding steps are correct, "
    "even if they look wrong, and check the current step against the problem and those steps as written.\n"
    "- The step is VALID if its arithmetic is correct and it follows logically from the problem "
    "statement and the preceding steps.\n"
    "- The step is INVALID if it contains an arithmetic mistake, misreads the problem, uses a quantity "
    "that the problem or the preceding steps do not support, or does not follow from them.\n"
    "- An error in an earlier step does not make this step invalid by itself.\n"
    "- Do not judge style, and do not judge whether the final answer is reached yet.\n\n"
    "Respond in exactly this format, with a one-sentence rationale:\n"
    "Verdict: VALID\n"
    "Rationale: <one sentence>\n"
    "or\n"
    "Verdict: INVALID\n"
    "Rationale: <one sentence>"
)

VERDICT_PATTERN = re.compile(r"verdict\s*:\s*\**\s*(VALID|INVALID)\b", re.IGNORECASE)
INVALID_PATTERN = re.compile(r"\bINVALID\b|\bNOT\s+VALID\b", re.IGNORECASE)
VALID_PATTERN = re.compile(r"\bVALID\b", re.IGNORECASE)
RATIONALE_PATTERN = re.compile(r"rationale\s*:[ \t]*\**[ \t]*", re.IGNORECASE)
SENTENCE_END_PATTERN = re.compile(r"[.!?](?=\s|$)")


def build_prm_judge_prompt(question: str, preceding_steps: list[str], current_step: str) -> str:
    if preceding_steps:
        preceding = "\n".join(f"Step {index}: {step}" for index, step in enumerate(preceding_steps, start=1))
    else:
        preceding = "(no preceding steps)"
    return (
        JUDGE_TEMPLATE.replace("{question}", question)
        .replace("{preceding}", preceding)
        .replace("{current}", current_step)
    )


def _parse_rationale(text: str) -> str:
    match = RATIONALE_PATTERN.search(text)
    if match is None:
        return ""
    line = text[match.end():].split("\n")[0]
    end = SENTENCE_END_PATTERN.search(line)
    if end is not None:
        line = line[: end.end()]
    return line.strip()


def parse_judge_decision(judge_output: str) -> tuple[float, str]:
    text = judge_output if isinstance(judge_output, str) else ""
    verdict = VERDICT_PATTERN.search(text)
    if verdict is not None:
        score = 1.0 if verdict.group(1).upper() == "VALID" else -1.0
    elif INVALID_PATTERN.search(text):
        score = -1.0
    elif VALID_PATTERN.search(text):
        score = 1.0
    else:
        score = -1.0
    return score, _parse_rationale(text)


def score_path(
    judge_model: Any = None,
    judge_tokenizer: Any = None,
    question: str = "",
    candidate: dict[str, Any] | None = None,
    judge_fn: JudgeFn | None = None,
) -> dict[str, Any]:
    steps = list((candidate or {}).get("steps") or [])
    judge = judge_fn or make_hf_judge_fn(judge_model, judge_tokenizer)
    prompts = [build_prm_judge_prompt(question, steps[:index], step) for index, step in enumerate(steps)]
    batch = getattr(judge, "batch", None)
    if not prompts:
        outputs = []
    elif batch is not None:
        outputs = batch(prompts)
    else:
        outputs = [judge(prompt) for prompt in prompts]
    step_scores, rationales = [], []
    for output in outputs:
        score, rationale = parse_judge_decision(output["text"])
        step_scores.append(score)
        rationales.append(rationale)
    result = {
        "step_scores": step_scores,
        "rationales": rationales,
        "judge_calls": len(steps),
        "judge_tokens": sum(int(output["num_tokens"]) for output in outputs),
        "judge_prompt_tokens": sum(int(output.get("prompt_tokens") or 0) for output in outputs),
    }
    if any(output.get("p_valid") is not None for output in outputs):
        result["step_probs"] = [
            float(output["p_valid"]) if output.get("p_valid") is not None else (0.99 if score > 0 else 0.01)
            for output, score in zip(outputs, step_scores)
        ]
    return result
