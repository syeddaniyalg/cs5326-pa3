import contextlib
import json
import os
import random
from typing import Any, Callable

GLOBAL_SEED = 5326
GENERATOR_ID = "Qwen/Qwen2.5-1.5B-Instruct"
JUDGE_ID = "Qwen/Qwen2.5-7B-Instruct"

GREEDY_GEN_KWARGS = {
    "do_sample": False,
    "temperature": None,
    "top_p": None,
    "top_k": None,
    "max_new_tokens": 512,
}

SAMPLING_GEN_KWARGS = {
    "do_sample": True,
    "temperature": 0.7,
    "top_p": 1.0,
    "top_k": 0,
    "repetition_penalty": 1.0,
    "max_new_tokens": 512,
}

JUDGE_GEN_KWARGS = {
    "do_sample": False,
    "temperature": None,
    "top_p": None,
    "top_k": None,
    "max_new_tokens": 96,
}


def set_seed_for_problem(problem_index: int) -> None:
    try:
        import transformers

        transformers.set_seed(GLOBAL_SEED + problem_index)
    except ImportError:
        random.seed(GLOBAL_SEED + problem_index)


def resolve_devices(generator_device: str = "cuda:0", judge_device: str = "cuda:1") -> tuple[str, str]:
    import torch

    count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if count >= 2:
        return generator_device, judge_device
    if count == 1:
        return "cuda:0", "cuda:0"
    return "cpu", "cpu"


def load_generator(device: str = "cuda:0", model_id: str = GENERATOR_ID):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if device == "cpu":
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32).to("cpu")
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map={"": device})
    model.eval()
    return model, tokenizer


def load_judge(device: str = "cuda:1", model_id: str = JUDGE_ID, quantize: bool = True):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    if device == "cpu":
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32).to("cpu")
    elif quantize:
        config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=config, device_map={"": device})
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map={"": device})
    model.eval()
    return model, tokenizer


def load_models(generator_device: str = "cuda:0", judge_device: str = "cuda:1"):
    generator_device, judge_device = resolve_devices(generator_device, judge_device)
    gen_model, gen_tokenizer = load_generator(generator_device)
    judge_model, judge_tokenizer = load_judge(judge_device)
    return gen_model, gen_tokenizer, judge_model, judge_tokenizer


def _eos_token_ids(model: Any, tokenizer: Any) -> set[int]:
    ids: set[int] = set()
    for source in (getattr(model, "generation_config", None), tokenizer):
        value = getattr(source, "eos_token_id", None) if source is not None else None
        if isinstance(value, int):
            ids.add(value)
        elif isinstance(value, (list, tuple, set)):
            ids.update(value)
    return ids


def _slice_to_first_eos(gen_tokens: Any, eos_ids: set[int]) -> Any:
    ids = gen_tokens.tolist() if hasattr(gen_tokens, "tolist") else list(gen_tokens)
    for index, token in enumerate(ids):
        if token in eos_ids:
            return gen_tokens[: index + 1]
    return gen_tokens


def _no_grad():
    try:
        import torch

        return torch.no_grad()
    except ImportError:
        return contextlib.nullcontext()


def _chat_text(tokenizer: Any, prompt: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
    )


def _to_device(inputs: Any, device: Any) -> dict:
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}


def make_hf_generate_fn(model: Any, tokenizer: Any) -> Callable[[str, int, dict], list[dict]]:
    def generate_fn(prompt: str, num_samples: int, gen_kwargs: dict) -> list[dict]:
        inputs = _to_device(tokenizer(_chat_text(tokenizer, prompt), return_tensors="pt"), model.device)
        prompt_len = inputs["input_ids"].shape[1]
        kwargs = dict(gen_kwargs)
        if not kwargs.get("do_sample", False):
            kwargs["temperature"] = None
            kwargs["top_p"] = None
            kwargs["top_k"] = None
        elif num_samples > 1:
            kwargs["num_return_sequences"] = num_samples
        kwargs.setdefault("pad_token_id", tokenizer.eos_token_id)
        with _no_grad():
            output_ids = model.generate(**inputs, **kwargs)
        eos_ids = _eos_token_ids(model, tokenizer)
        results = []
        for row in range(output_ids.shape[0]):
            tokens = _slice_to_first_eos(output_ids[row][prompt_len:], eos_ids)
            text = tokenizer.decode(tokens, skip_special_tokens=True)
            results.append({"text": text, "num_tokens": len(tokens)})
        return results

    return generate_fn


def _verdict_token_ids(tokenizer: Any) -> tuple[list[int], list[int]] | None:
    if not hasattr(tokenizer, "encode"):
        return None
    try:
        valid = sorted({tokenizer.encode(prefix + "VALID", add_special_tokens=False)[0] for prefix in ("", " ")})
        invalid = sorted({tokenizer.encode(prefix + "INVALID", add_special_tokens=False)[0] for prefix in ("", " ")})
    except Exception:
        return None
    if set(valid) & set(invalid):
        return None
    return valid, invalid


class VerdictRecorder:
    def __init__(self, token_ids: list[int]):
        self.token_ids = token_ids
        self.steps: list[Any] = []

    def __call__(self, input_ids: Any, scores: Any) -> Any:
        self.steps.append(scores[:, self.token_ids].detach().float())
        return scores


def _verdict_probability(recorded: Any, row: int, generated: list[int], ids: list[int], num_valid: int) -> float | None:
    import torch

    lookup = set(ids)
    for position, token in enumerate(generated):
        if token in lookup and position < recorded.shape[0]:
            logits = recorded[position, row]
            valid = torch.logsumexp(logits[:num_valid], dim=0)
            invalid = torch.logsumexp(logits[num_valid:], dim=0)
            return float(torch.sigmoid(valid - invalid))
    return None


def make_hf_judge_fn(judge_model: Any, judge_tokenizer: Any) -> Callable[[str], dict]:
    verdict_ids = _verdict_token_ids(judge_tokenizer)

    def run(prompts: list[str]) -> list[dict]:
        texts = [_chat_text(judge_tokenizer, prompt) for prompt in prompts]
        if len(texts) == 1:
            encoded = judge_tokenizer(texts[0], return_tensors="pt")
        else:
            judge_tokenizer.padding_side = "left"
            encoded = judge_tokenizer(texts, return_tensors="pt", padding=True)
        inputs = _to_device(encoded, judge_model.device)
        prompt_len = inputs["input_ids"].shape[1]
        kwargs = dict(JUDGE_GEN_KWARGS)
        kwargs.setdefault("pad_token_id", judge_tokenizer.eos_token_id)
        recorder = None
        if verdict_ids is not None:
            from transformers import LogitsProcessorList

            ids = verdict_ids[0] + verdict_ids[1]
            recorder = VerdictRecorder(ids)
            kwargs["logits_processor"] = LogitsProcessorList([recorder])
        with _no_grad():
            output_ids = judge_model.generate(**inputs, **kwargs)
        recorded = None
        if recorder is not None and recorder.steps:
            import torch

            recorded = torch.stack(recorder.steps).cpu()
        eos_ids = _eos_token_ids(judge_model, judge_tokenizer)
        mask = inputs.get("attention_mask")
        results = []
        for row in range(output_ids.shape[0]):
            tokens = _slice_to_first_eos(output_ids[row][prompt_len:], eos_ids)
            ids_list = tokens.tolist() if hasattr(tokens, "tolist") else list(tokens)
            probability = None
            if recorded is not None:
                probability = _verdict_probability(recorded, row, ids_list, ids, len(verdict_ids[0]))
            results.append(
                {
                    "text": judge_tokenizer.decode(tokens, skip_special_tokens=True),
                    "num_tokens": len(tokens),
                    "p_valid": probability,
                    "prompt_tokens": int(mask[row].sum()) if mask is not None else int(prompt_len),
                }
            )
        return results

    def judge_fn(judge_prompt: str) -> dict:
        return run([judge_prompt])[0]

    judge_fn.batch = run
    return judge_fn


def write_json(obj: Any, path: str) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, ensure_ascii=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def read_json(path: str, default: Any = None) -> Any:
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def append_jsonl(record: dict, path: str) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    records = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def write_jsonl(records: list[dict], path: str) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def save_results_summary(results: list[dict[str, Any]], filepath: str = "outputs/results_summary.json") -> None:
    write_json(results, filepath)


def save_aggregation_ablation(ablation: list[dict[str, Any]], filepath: str = "outputs/aggregation_ablation.json") -> None:
    write_json(ablation, filepath)


def save_sc_scaling_curve(curve: list[dict[str, Any]], filepath: str = "outputs/sc_scaling_curve.json") -> None:
    write_json(curve, filepath)
