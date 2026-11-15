# Search over Reasoning: A Prompted Step Verifier versus Repeated Sampling on GSM8K

This repository contains the implementation for CS 5326 Programming Assignment 3. It compares greedy decoding, self-consistency (SC), and best-of-five search guided by a step-level verifier on 50 fixed GSM8K problems. All experiments are inference only and no model weights are updated.

## Overview

A Qwen2.5-1.5B-Instruct generator (FP16) writes step-structured solutions. A Qwen2.5-7B-Instruct judge (4-bit NF4) labels each step as valid or invalid under local validity, meaning that the preceding steps are assumed correct. Step labels are combined into a path score, and the best of five shared candidates is selected.

Six configurations are evaluated:

| Configuration | Candidates | Selection |
| --- | --- | --- |
| Greedy | 1 | None |
| SC-5 | 5 | Majority vote |
| SC-15 (compute matched) | 15 | Majority vote |
| PRM best-of-5, min | 5 | Highest minimum step score |
| PRM best-of-5, product | 5 | Highest product of step probabilities |
| PRM best-of-5, step penalty | 5 | Highest mean score minus a penalty per invalid step |

SC-5 and the three PRM variants use the same five candidates per problem, and the judge scores them once. SC-15 extends the same five candidates with ten more samples. An optional continuous variant, which uses the judge's verdict probability instead of a binary label, is also reported.

## Results

| Method | Accuracy (%) | Total LLM calls | Output tokens per problem |
| --- | --- | --- | --- |
| Greedy | 56.0 | 50 | 228 |
| SC-5 | 64.0 | 250 | 927 |
| SC-15 | 80.0 | 750 | 2,746 |
| PRM best-of-5 (min, product, step penalty) | 70.0 | 1,709 | 1,841 (927 generated, 914 judge) |

Full tables, confidence intervals, and analysis are in `outputs/` and in the report source under `paper/`.

## Repository Structure

```
.
|-- data/
|   `-- gsm8k_test_50.jsonl        Fixed 50-problem benchmark
|-- src/
|   |-- generation.py              Step prompt, greedy decoding, self-consistency
|   |-- extraction.py              Step and answer parsers
|   |-- prm_judge.py               Judge prompt, verdict parser, step scoring
|   |-- search.py                  Path scores, selection, best-of-N, compute-matched SC
|   |-- utils.py                   Model loading, generation and judge callables, file helpers
|   |-- evaluation.py              Metrics, bootstrap intervals, paired tests, budget analysis
|   |-- pipeline.py                Two-stage run with checkpoints and live progress
|   `-- reporting.py               Writes tables, figures, and audit files
|-- notebooks/
|   `-- kaggle_pa3.ipynb           Kaggle runner for a T4 x2 session
|-- outputs/                       Results of the reported run
|-- paper/
|   |-- main.tex                   Report source (NeurIPS 2026 style)
|   |-- neurips_2026.sty
|   `-- figures/
|-- run_all.py                     Command-line entry point
|-- requirements.txt
`-- pyproject.toml
```

## Setup

The code needs Python 3.10 or later, PyTorch, Transformers, Accelerate, NumPy, Matplotlib, tqdm, and `bitsandbytes` 0.45.5 or later. Kaggle images provide everything except `bitsandbytes`, which should be upgraded:

```bash
pip install -q -U bitsandbytes
```

The run was made on two NVIDIA T4 GPUs. The generator uses `cuda:0` and the judge uses `cuda:1`. With a single GPU, both models are placed on `cuda:0`. Model weights are downloaded from the Hugging Face Hub on first use.

## Running

On Kaggle, open `notebooks/kaggle_pa3.ipynb` with the T4 x2 accelerator and internet enabled, and run all cells. The notebook clones this repository, downloads the models, runs the pipeline, and packages `outputs/` into `pa3_outputs.zip`.

From a terminal, run the full evaluation with:

```bash
python run_all.py
```

Options:

| Option | Description |
| --- | --- |
| `--data PATH` | Benchmark file, default `data/gsm8k_test_50.jsonl` |
| `--output-dir PATH` | Output directory, default `outputs` |
| `--limit N` | Use only the first N problems |
| `--generator-device`, `--judge-device` | Devices for the two models |
| `--no-parallel` | Run generation and judging one after the other |
| `--audit` | Compute the judge audit confusion matrix from `outputs/judge_audit/audit_sample.csv` |

With two GPUs, generation and judging run in parallel threads: the judge scores the five-candidate pool of each problem while the generator moves on to the next one.

## Outputs

Files written to `outputs/`:

| File | Content |
| --- | --- |
| `results_summary.json` | Accuracy, calls, tokens, and time for the six configurations |
| `aggregation_ablation.json` | Min, product, and step penalty at several penalty weights, with tie rates |
| `sc_scaling_curve.json` | Self-consistency accuracy and tokens for 1 to 15 samples |
| `per_problem.jsonl` | Completions, candidates, step scores, rationales, and selections per problem |
| `analysis.json` | Bootstrap intervals, paired comparisons, budget and cost analysis, judge statistics |
| `failure_cases.json` | Candidate examples of judge false positives, false negatives, and shared errors |
| `bonus_product_logprob.json` | Result of the continuous verdict probability variant |
| `run_meta.json` | Hardware, library versions, configuration, and runtimes |
| `tables/report_tables.md` | Tables in Markdown form |
| `figures/` | Scaling, ablation, and path statistics plots |
| `judge_audit/` | Audit sample of 40 judged steps with hand labels and the resulting confusion matrix |

Per-problem checkpoints are written to `outputs/checkpoints/` during a run and are not tracked by git. A rerun resumes from them, and it refuses to resume if the prompts or sampling settings have changed.

## Reproducibility

Each problem is seeded with `5326 + problem_index` before generation, and bootstrap intervals use seed 5326. Greedy decoding and the judge use deterministic decoding. Sampling uses temperature 0.7, top-p 1.0, and top-k 0, with at most 512 new tokens for the generator and 96 for the judge. Judge prompts are batched in groups of at most four with left padding, which can change results slightly through numerical differences. The software versions of the reported run are recorded in `outputs/run_meta.json`.
