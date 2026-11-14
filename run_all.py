import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.pipeline import PipelineConfig, load_problems, load_runtime, pending_work, run_pipeline
from src.reporting import finalize_audit, markdown_table


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CS5326 PA3: search over reasoning, full evaluation pipeline")
    parser.add_argument("--data", default="data/gsm8k_test_50.jsonl")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--generator-device", default="cuda:0")
    parser.add_argument("--judge-device", default="cuda:1")
    parser.add_argument("--no-parallel", action="store_true")
    parser.add_argument("--audit", action="store_true")
    return parser.parse_args(argv)


def print_results(summary: list[dict]) -> None:
    headers = list(summary[0].keys())
    print(markdown_table(headers, [[row[h] for h in headers] for row in summary]))


def main(argv: list[str] | None = None) -> dict | None:
    arguments = parse_arguments(argv)
    if arguments.audit:
        result = finalize_audit(arguments.output_dir)
        print(json.dumps(result, indent=2) if result else "No labelled audit rows found.")
        return result
    config = PipelineConfig(
        data_path=arguments.data,
        output_dir=arguments.output_dir,
        limit=arguments.limit,
        generator_device=arguments.generator_device,
        judge_device=arguments.judge_device,
        parallel=not arguments.no_parallel,
    )
    problems = load_problems(config.data_path, config.limit)
    need_generator, need_judge = pending_work(problems, config)
    generate_fn, judge_fn, parallel = load_runtime(config, need_generator, need_judge)
    artifacts = run_pipeline(problems, config, generate_fn, judge_fn, parallel)
    print_results(artifacts["summary"])
    return artifacts


if __name__ == "__main__":
    main()
