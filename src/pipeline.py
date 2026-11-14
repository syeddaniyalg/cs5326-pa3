import contextlib
import hashlib
import json
import os
import platform
import queue
import threading
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable

from tqdm import tqdm

from . import evaluation as ev
from .generation import PROMPT_TEMPLATE, compute_majority_vote, run_greedy, run_self_consistency
from .prm_judge import JUDGE_TEMPLATE
from .reporting import write_artifacts
from .search import run_compute_matched_self_consistency, run_prm_best_of_n
from .utils import (
    GENERATOR_ID,
    GLOBAL_SEED,
    GREEDY_GEN_KWARGS,
    JUDGE_GEN_KWARGS,
    JUDGE_ID,
    SAMPLING_GEN_KWARGS,
    append_jsonl,
    load_generator,
    load_judge,
    make_hf_generate_fn,
    make_hf_judge_fn,
    read_json,
    read_jsonl,
    resolve_devices,
    set_seed_for_problem,
    write_json,
)

GenerateFn = Callable[[str, int, dict], list[dict]]
JudgeFn = Callable[[str], dict]


@dataclass
class PipelineConfig:
    data_path: str = "data/gsm8k_test_50.jsonl"
    output_dir: str = "outputs"
    limit: int | None = None
    pool_size: int = ev.POOL_SIZE
    target_calls: int = ev.TARGET_CALLS
    temperature: float = 0.7
    generator_device: str = "cuda:0"
    judge_device: str = "cuda:1"
    parallel: bool = True

    @property
    def checkpoint_dir(self) -> str:
        return os.path.join(self.output_dir, "checkpoints")

    @property
    def generation_path(self) -> str:
        return os.path.join(self.checkpoint_dir, "generation.jsonl")

    @property
    def judging_path(self) -> str:
        return os.path.join(self.checkpoint_dir, "judging.jsonl")

    @property
    def meta_path(self) -> str:
        return os.path.join(self.checkpoint_dir, "meta.json")


def load_problems(path: str, limit: int | None = None) -> list[dict[str, Any]]:
    problems = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                problems.append(json.loads(line))
    if limit is not None:
        problems = problems[:limit]
    for index, problem in enumerate(problems):
        problem["index"] = index
    return problems


def checkpoint_signature(config: PipelineConfig) -> dict[str, Any]:
    digest = hashlib.sha256((PROMPT_TEMPLATE + JUDGE_TEMPLATE).encode("utf-8")).hexdigest()[:16]
    return {
        "generator": GENERATOR_ID,
        "judge": JUDGE_ID,
        "seed": GLOBAL_SEED,
        "pool_size": config.pool_size,
        "target_calls": config.target_calls,
        "temperature": config.temperature,
        "greedy": GREEDY_GEN_KWARGS,
        "sampling": SAMPLING_GEN_KWARGS,
        "judge_kwargs": JUDGE_GEN_KWARGS,
        "prompt_digest": digest,
    }


def verify_checkpoint(config: PipelineConfig) -> None:
    signature = checkpoint_signature(config)
    stored = read_json(os.path.join(config.checkpoint_dir, "signature.json"))
    if stored is None:
        write_json(signature, os.path.join(config.checkpoint_dir, "signature.json"))
    elif stored != signature:
        raise RuntimeError(
            f"Checkpoints in {config.checkpoint_dir} were produced with a different configuration; "
            "remove the directory to start a fresh run."
        )


def load_checkpoint(path: str, problems: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    valid = {problem["index"]: problem["id"] for problem in problems}
    records = {}
    for record in read_jsonl(path):
        if valid.get(record.get("index")) == record.get("id"):
            records[record["index"]] = record
    return records


def device_context(device: str | None):
    if device and device.startswith("cuda"):
        import torch

        index = int(device.split(":")[1]) if ":" in device else 0
        if torch.cuda.is_available() and index < torch.cuda.device_count():
            return torch.cuda.device(index)
    return contextlib.nullcontext()


def generate_problem(problem: dict[str, Any], generate_fn: GenerateFn, config: PipelineConfig) -> dict[str, Any]:
    question = problem["question"]
    set_seed_for_problem(problem["index"])
    greedy = run_greedy(None, None, question, generate_fn=generate_fn)
    shared = run_self_consistency(None, None, question, n=config.pool_size, temperature=config.temperature, generate_fn=generate_fn)
    matched = run_compute_matched_self_consistency(
        None,
        None,
        question,
        target_calls=config.target_calls,
        temperature=config.temperature,
        existing_candidates=shared["candidates"],
        generate_fn=generate_fn,
    )
    return {
        "index": problem["index"],
        "id": problem["id"],
        "question": question,
        "ground_truth_answer": problem["ground_truth_answer"],
        "greedy": greedy,
        "sc5_time_seconds": shared["time_seconds"],
        "sc15_extra_time_seconds": matched["time_seconds"],
        "candidates": matched["candidates"],
    }


def judge_problem(generation: dict[str, Any], judge_fn: JudgeFn, config: PipelineConfig) -> dict[str, Any]:
    pool = generation["candidates"][: config.pool_size]
    result = run_prm_best_of_n(
        None,
        None,
        None,
        None,
        generation["question"],
        n=config.pool_size,
        aggregation_strategy="min",
        candidates=pool,
        judge_fn=judge_fn,
    )
    scored = result["candidates"]
    scores = [c["step_scores"] for c in scored]
    probabilities = [c.get("step_probs", []) for c in scored]
    has_probabilities = any(probabilities) and all(len(p) == len(s) for p, s in zip(probabilities, scores))
    return {
        "index": generation["index"],
        "id": generation["id"],
        "step_scores": scores,
        "rationales": [c["rationales"] for c in scored],
        "step_probs": probabilities if has_probabilities else None,
        "judge_calls": result["judge_calls"],
        "judge_tokens": result["judge_tokens"],
        "judge_prompt_tokens": result["judge_prompt_tokens"],
        "time_seconds": result["time_seconds"],
    }


class GenerationMeter:
    def __init__(self) -> None:
        self.count = 0
        self.correct = {"greedy": 0, "sc5": 0, "sc15": 0}
        self.tokens = 0
        self.seconds = 0.0

    def update(self, record: dict[str, Any]) -> dict[str, str]:
        truth = record["ground_truth_answer"]
        answers = [c["answer"] for c in record["candidates"]]
        self.count += 1
        self.correct["greedy"] += ev.is_correct(record["greedy"]["answer"], truth)
        self.correct["sc5"] += ev.is_correct(compute_majority_vote(answers[: ev.POOL_SIZE]), truth)
        self.correct["sc15"] += ev.is_correct(compute_majority_vote(answers), truth)
        self.tokens += record["greedy"]["generated_tokens"]
        self.seconds += record["greedy"]["time_seconds"] + record["sc5_time_seconds"] + record["sc15_extra_time_seconds"]
        return {
            "greedy": f"{100 * self.correct['greedy'] / self.count:.1f}%",
            "sc5": f"{100 * self.correct['sc5'] / self.count:.1f}%",
            "sc15": f"{100 * self.correct['sc15'] / self.count:.1f}%",
            "greedy_tok": f"{self.tokens / self.count:.0f}",
            "s/prob": f"{self.seconds / self.count:.1f}",
        }


class JudgeMeter:
    def __init__(self) -> None:
        self.count = 0
        self.correct = {key: 0 for key in ev.PRM_KEYS}
        self.steps = 0
        self.invalid = 0
        self.paths = 0
        self.tokens = 0
        self.seconds = 0.0

    def update(self, generation: dict[str, Any], judging: dict[str, Any]) -> dict[str, str]:
        record = ev.build_problem_record(generation, judging)
        self.count += 1
        for key in ev.PRM_KEYS:
            self.correct[key] += record["correct"][key]
        for scores in judging["step_scores"]:
            self.paths += 1
            self.steps += len(scores)
            self.invalid += sum(1 for s in scores if s < 0)
        self.tokens += judging["judge_tokens"]
        self.seconds += judging["time_seconds"]
        return {
            "min": f"{100 * self.correct['min'] / self.count:.1f}%",
            "prod": f"{100 * self.correct['product'] / self.count:.1f}%",
            "pen": f"{100 * self.correct['step_penalty_l2.0'] / self.count:.1f}%",
            "K": f"{self.steps / max(self.paths, 1):.2f}",
            "invalid": f"{100 * self.invalid / max(self.steps, 1):.1f}%",
            "s/prob": f"{self.seconds / self.count:.1f}",
        }


def run_stages(
    problems: list[dict[str, Any]],
    config: PipelineConfig,
    generate_fn: GenerateFn | None,
    judge_fn: JudgeFn | None,
    parallel: bool,
) -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]]]:
    generated = load_checkpoint(config.generation_path, problems)
    judged = load_checkpoint(config.judging_path, problems)
    judged = {index: record for index, record in judged.items() if index in generated}
    pending_generation =[p for p in problems if p["index"] not in generated]
    pending_judging = [generated[i] for i in sorted(generated) if i not in judged]
    if pending_generation and generate_fn is None:
        raise RuntimeError("generate_fn is required for pending generation work")
    if (pending_generation or pending_judging) and judge_fn is None:
        raise RuntimeError("judge_fn is required for pending judging work")

    work: queue.Queue = queue.Queue()
    for record in pending_judging:
        work.put(record)
    failures: list[BaseException] = []
    stop = threading.Event()
    total = len(problems)

    generation_meter = GenerationMeter()
    for index in sorted(generated):
        generation_meter.update(generated[index])
    judge_meter = JudgeMeter()
    for index in sorted(judged):
        judge_meter.update(generated[index], judged[index])

    generation_bar = tqdm(total=total, initial=len(generated), desc="generate", unit="prob", position=0, leave=True)
    judge_bar = tqdm(total=total, initial=len(judged), desc="judge   ", unit="prob", position=1, leave=True)

    def generation_loop() -> None:
        try:
            with device_context(config.generator_device):
                for problem in pending_generation:
                    if stop.is_set():
                        break
                    record = generate_problem(problem, generate_fn, config)
                    append_jsonl(record, config.generation_path)
                    generated[problem["index"]] = record
                    generation_bar.update(1)
                    generation_bar.set_postfix(generation_meter.update(record))
                    work.put(record)
        except BaseException as error:
            failures.append(error)
            stop.set()
        finally:
            work.put(None)

    def judging_loop() -> None:
        try:
            with device_context(config.judge_device):
                while True:
                    item = work.get()
                    if item is None or stop.is_set():
                        break
                    record = judge_problem(item, judge_fn, config)
                    append_jsonl(record, config.judging_path)
                    judged[item["index"]] = record
                    judge_bar.update(1)
                    judge_bar.set_postfix(judge_meter.update(item, record))
        except BaseException as error:
            failures.append(error)
            stop.set()

    if parallel:
        threads = [
            threading.Thread(target=generation_loop, daemon=True),
            threading.Thread(target=judging_loop, daemon=True),
        ]
        for thread in threads:
            thread.start()
        try:
            while any(thread.is_alive() for thread in threads):
                for thread in threads:
                    thread.join(0.5)
        except KeyboardInterrupt:
            stop.set()
            raise
    else:
        generation_loop()
        judging_loop()
    generation_bar.close()
    judge_bar.close()
    if failures:
        raise failures[0]
    return generated, judged


def collect_environment(config: PipelineConfig) -> dict[str, Any]:
    from importlib import metadata

    info: dict[str, Any] = {"python": platform.python_version(), "platform": platform.platform()}
    for package in ("torch", "transformers", "accelerate", "bitsandbytes", "numpy", "matplotlib", "tqdm"):
        try:
            info[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            info[package] = None
    try:
        import torch

        info["cuda_available"] = torch.cuda.is_available()
        info["cuda_version"] = torch.version.cuda
        info["gpus"] = [
            {
                "name": torch.cuda.get_device_name(i),
                "memory_gb": round(torch.cuda.get_device_properties(i).total_memory / 1024**3, 2),
            }
            for i in range(torch.cuda.device_count())
        ]
    except Exception:
        info["gpus"] = []
    info["generator"] = {"model": GENERATOR_ID, "dtype": "float16", "device": config.generator_device}
    info["judge"] = {"model": JUDGE_ID, "quantization": "4-bit NF4 (double quant, fp16 compute)", "device": config.judge_device}
    info["seed_protocol"] = f"transformers.set_seed({GLOBAL_SEED} + problem_index) before each problem"
    return info


def run_pipeline(
    problems: list[dict[str, Any]],
    config: PipelineConfig,
    generate_fn: GenerateFn | None,
    judge_fn: JudgeFn | None,
    parallel: bool | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    verify_checkpoint(config)
    parallel = config.parallel if parallel is None else parallel
    generated, judged = run_stages(problems, config, generate_fn, judge_fn, parallel)
    records = [ev.build_problem_record(generated[p["index"]], judged[p["index"]]) for p in problems]
    artifacts = write_artifacts(records, config.output_dir)

    meta = read_json(config.meta_path, {"sessions": []})
    meta["sessions"].append({"elapsed_seconds": time.perf_counter() - started, "finished_at": time.strftime("%Y-%m-%d %H:%M:%S")})
    meta["elapsed_seconds_total"] = sum(s["elapsed_seconds"] for s in meta["sessions"])
    write_json(meta, config.meta_path)
    write_json(
        {
            "environment": collect_environment(config),
            "config": asdict(config),
            "checkpoint_signature": checkpoint_signature(config),
            "wall_clock_seconds_total": meta["elapsed_seconds_total"],
            "sessions": meta["sessions"],
            "stage_seconds": {
                "generation": sum(r["seconds"]["greedy"] + r["seconds"]["sc5"] + r["seconds"]["sc15_extra"] for r in records),
                "judging": sum(r["seconds"]["judge"] for r in records),
            },
        },
        os.path.join(config.output_dir, "run_meta.json"),
    )
    return artifacts


def prefetch_models(model_ids: list[str]) -> None:
    from huggingface_hub import snapshot_download

    patterns = ["*.json", "*.safetensors", "*.txt", "tokenizer*"]
    errors: list[BaseException] = []

    def fetch(model_id: str) -> None:
        try:
            snapshot_download(model_id, allow_patterns=patterns)
        except BaseException as error:
            errors.append(error)

    threads = [threading.Thread(target=fetch, args=(model_id,), daemon=True) for model_id in model_ids]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if errors:
        raise errors[0]


def load_runtime(config: PipelineConfig, need_generator: bool, need_judge: bool) -> tuple[GenerateFn | None, JudgeFn | None, bool]:
    generator_device, judge_device = resolve_devices(config.generator_device, config.judge_device)
    config.generator_device, config.judge_device = generator_device, judge_device
    wanted = []
    if need_generator:
        wanted.append(GENERATOR_ID)
    if need_judge:
        wanted.append(JUDGE_ID)
    if wanted:
        prefetch_models(wanted)
    loaded: dict[str, Any] = {}
    errors: list[BaseException] = []

    def load(name: str, loader: Callable[[], Any]) -> None:
        try:
            loaded[name] = loader()
        except BaseException as error:
            errors.append(error)

    threads = []
    if need_generator:
        threads.append(threading.Thread(target=load, args=("generator", lambda: load_generator(generator_device)), daemon=True))
    if need_judge:
        threads.append(threading.Thread(target=load, args=("judge", lambda: load_judge(judge_device)), daemon=True))
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if errors:
        raise errors[0]
    generate_fn = make_hf_generate_fn(*loaded["generator"]) if need_generator else None
    judge_fn = make_hf_judge_fn(*loaded["judge"]) if need_judge else None
    parallel = config.parallel and generator_device != judge_device
    return generate_fn, judge_fn, parallel


def pending_work(problems: list[dict[str, Any]], config: PipelineConfig) -> tuple[bool, bool]:
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    verify_checkpoint(config)
    generated = load_checkpoint(config.generation_path, problems)
    judged = load_checkpoint(config.judging_path, problems)
    need_generator = len(generated) < len(problems)
    need_judge = need_generator or len(judged) < len(generated)
    return need_generator, need_judge
