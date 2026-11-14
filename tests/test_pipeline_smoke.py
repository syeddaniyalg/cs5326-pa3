import json
import os
import random
import re
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import evaluation as ev
from src.pipeline import PipelineConfig, load_problems, pending_work, run_pipeline
from src.reporting import finalize_audit
from src.utils import read_json, read_jsonl

PROBLEM_COUNT = 8


def write_dataset(directory: str) -> str:
    path = os.path.join(directory, "data.jsonl")
    with open(path, "w", encoding="utf-8") as handle:
        for index in range(PROBLEM_COUNT):
            row = {"id": f"p{index}", "question": f"Question number {index}: what is {index} + 10?", "ground_truth_answer": index + 10}
            handle.write(json.dumps(row) + "\n")
    return path


def make_generate_fn(truths: dict[str, int]):
    def generate_fn(prompt: str, num_samples: int, gen_kwargs: dict) -> list[dict]:
        question = prompt.split("Problem: ", 1)[1]
        truth = truths[question]
        outputs = []
        for _ in range(num_samples):
            wrong = gen_kwargs.get("do_sample") and random.random() < 0.5
            answer = truth + random.choice([1, 2, 3]) if wrong else truth
            marker = " WRONG" if wrong else ""
            steps = random.choice([2, 3, 4])
            lines = [f"Step {i}: computing part {i}{marker if i == steps else ''}." for i in range(1, steps + 1)]
            text = "\n".join(lines) + f"\nFinal Answer: {answer}"
            outputs.append({"text": text, "num_tokens": 10 + len(text) // 8})
        return outputs

    return generate_fn


def judge_fn(prompt: str) -> dict:
    current = prompt.split("Current Step:", 1)[1].split("\n", 1)[0]
    invalid = "WRONG" in current
    text = "Verdict: INVALID\nRationale: Bad step." if invalid else "Verdict: VALID\nRationale: Fine."
    return {"text": text, "num_tokens": 9, "p_valid": 0.05 if invalid else 0.9, "prompt_tokens": 120}


def make_config(directory: str, parallel: bool) -> PipelineConfig:
    return PipelineConfig(
        data_path=write_dataset(directory),
        output_dir=os.path.join(directory, "outputs"),
        generator_device="cpu",
        judge_device="cpu",
        parallel=parallel,
    )


def run_once(directory: str, parallel: bool):
    config = make_config(directory, parallel)
    problems = load_problems(config.data_path)
    truths = {p["question"]: p["ground_truth_answer"] for p in problems}
    artifacts = run_pipeline(problems, config, make_generate_fn(truths), judge_fn, parallel)
    return config, problems, truths, artifacts


def check_artifacts(config: PipelineConfig, problems: list) -> None:
    out = config.output_dir
    summary = read_json(os.path.join(out, "results_summary.json"))
    assert [row["Method"] for row in summary] == [ev.METHOD_NAMES[k] for k in ev.METHOD_KEYS]
    keys = {"Method", "Accuracy (%)", "Generation Calls", "Judge Calls", "Total LLM Calls", "Avg Gen Tokens / Problem", "Avg Judge Tokens / Problem", "Total Time (s)"}
    assert all(set(row) == keys for row in summary)
    count = len(problems)
    assert [row["Generation Calls"] for row in summary] == [count, 5 * count, 15 * count, 5 * count, 5 * count, 5 * count]
    prm_rows = summary[3:]
    assert len({(r["Judge Calls"], r["Avg Gen Tokens / Problem"], r["Avg Judge Tokens / Problem"]) for r in prm_rows}) == 1
    assert prm_rows[0]["Judge Calls"] > 0
    assert prm_rows[0]["Total LLM Calls"] == prm_rows[0]["Generation Calls"] + prm_rows[0]["Judge Calls"]
    assert all(r["Total Time (s)"] >= summary[1]["Total Time (s)"] for r in prm_rows)

    curve = read_json(os.path.join(out, "sc_scaling_curve.json"))
    assert [row["N"] for row in curve] == list(range(1, 16))
    assert all(set(row) == {"N", "Accuracy (%)", "Avg Gen Tokens / Problem"} for row in curve)
    assert curve[4]["Accuracy (%)"] == summary[1]["Accuracy (%)"]
    assert curve[14]["Accuracy (%)"] == summary[2]["Accuracy (%)"]
    assert curve[4]["Avg Gen Tokens / Problem"] == summary[1]["Avg Gen Tokens / Problem"]
    assert curve[14]["Avg Gen Tokens / Problem"] == summary[2]["Avg Gen Tokens / Problem"]

    ablation = read_json(os.path.join(out, "aggregation_ablation.json"))
    assert [(r["Strategy"], r["Lambda"]) for r in ablation] == [("min", None), ("product", None), ("step_penalty", 0.0), ("step_penalty", 0.5), ("step_penalty", 2.0)]
    assert all(set(r) == {"Strategy", "Lambda", "Accuracy (%)", "Tie Rate (%)"} for r in ablation)
    assert ablation[0]["Accuracy (%)"] == summary[3]["Accuracy (%)"]
    assert ablation[1]["Accuracy (%)"] == summary[4]["Accuracy (%)"]
    assert ablation[4]["Accuracy (%)"] == summary[5]["Accuracy (%)"]

    records = read_jsonl(os.path.join(out, "per_problem.jsonl"))
    assert len(records) == count
    first = records[0]
    assert len(first["candidates"]) == 15
    assert all("step_scores" in c and "rationales" in c for c in first["candidates"][:5])
    assert all("step_scores" not in c for c in first["candidates"][5:])
    assert set(first["answers"]) >= {"greedy", "sc5", "sc15", "min", "product", "step_penalty_l2.0"}
    assert first["greedy"]["completion"]

    for name in ("analysis.json", "failure_cases.json", "bonus_product_logprob.json", "run_meta.json"):
        assert os.path.exists(os.path.join(out, name)), name
    assert os.path.exists(os.path.join(out, "tables", "report_tables.md"))
    for name in ("accuracy_vs_calls.png", "accuracy_vs_tokens.png", "aggregation_ablation.png", "path_statistics.png"):
        assert os.path.getsize(os.path.join(out, "figures", name)) > 1000, name
    audit = os.path.join(out, "judge_audit", "audit_sample.csv")
    assert os.path.exists(audit)
    analysis = read_json(os.path.join(out, "analysis.json"))
    assert analysis["bonus_product_logprob"]["available"] is True
    assert analysis["bootstrap"]["resamples"] == 1000 and analysis["bootstrap"]["seed"] == 5326
    assert analysis["budget"]["budget_tokens"] == 10 * analysis["budget"]["greedy_generated_tokens"]


def stable(records: list) -> list:
    return [{k: v for k, v in r.items() if k != "seconds"} for r in records]


def test_pipeline_parallel_and_resume():
    directory = tempfile.mkdtemp()
    try:
        config, problems, truths, _ = run_once(directory, True)
        check_artifacts(config, problems)
        first = stable(read_jsonl(os.path.join(config.output_dir, "per_problem.jsonl")))
        assert pending_work(problems, config) == (False, False)
        again = run_pipeline(problems, config, None, None, False)
        assert again["summary"][0]["Method"] == ev.METHOD_NAMES["greedy"]
        second = stable(read_jsonl(os.path.join(config.output_dir, "per_problem.jsonl")))
        assert first == second
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def test_pipeline_sequential_and_partial_resume():
    directory = tempfile.mkdtemp()
    try:
        config = make_config(directory, False)
        problems = load_problems(config.data_path)
        truths = {p["question"]: p["ground_truth_answer"] for p in problems}
        run_pipeline(problems[:3], config, make_generate_fn(truths), judge_fn, False)
        generation = os.path.join(config.checkpoint_dir, "generation.jsonl")
        judging = os.path.join(config.checkpoint_dir, "judging.jsonl")
        lines = open(judging, encoding="utf-8").read().splitlines()
        with open(judging, "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines[:2]) + "\n{broken")
        assert len(read_jsonl(generation)) == 3
        assert pending_work(problems, config) == (True, True)
        run_pipeline(problems, config, make_generate_fn(truths), judge_fn, True)
        check_artifacts(config, problems)
        assert len(read_jsonl(generation)) == PROBLEM_COUNT
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def test_audit_roundtrip():
    directory = tempfile.mkdtemp()
    try:
        config, problems, _, _ = run_once(directory, False)
        path = os.path.join(config.output_dir, "judge_audit", "audit_sample.csv")
        import csv

        with open(path, encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) >= 30
        assert sum(1 for r in rows if r["judge_verdict"] == "INVALID") >= 10
        for row in rows:
            row["human_label"] = row["judge_verdict"]
        with open(path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        result = finalize_audit(config.output_dir)
        assert result["FP"] == 0 and result["FN"] == 0 and result["precision_invalid"] == 1.0
        assert result["meets_minimum_30_steps"] and result["meets_minimum_10_judge_invalid"]
        run_pipeline(problems, config, None, None, False)
        with open(path, encoding="utf-8", newline="") as handle:
            assert all(r["human_label"] for r in csv.DictReader(handle))
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def test_configuration_mismatch_is_rejected():
    directory = tempfile.mkdtemp()
    try:
        config, problems, truths, _ = run_once(directory, False)
        config.temperature = 0.9
        try:
            pending_work(problems, config)
            assert False
        except RuntimeError:
            pass
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def run_all_tests() -> int:
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    failed = 0
    for test in tests:
        try:
            test()
            print(f"  [PASS] {test.__name__}")
        except Exception as error:
            import traceback

            traceback.print_exc()
            print(f"  [FAIL] {test.__name__}: {error}")
            failed += 1
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(run_all_tests())
