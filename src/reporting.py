import csv
import os
from typing import Any

from . import evaluation as ev
from .utils import write_json, write_jsonl

MARKERS = {"min": "s", "product": "^", "step_penalty_l2.0": "D"}

AUDIT_COLUMNS = [
    "audit_id",
    "problem_id",
    "candidate_index",
    "step_index",
    "question",
    "preceding_steps",
    "step",
    "judge_verdict",
    "judge_rationale",
    "candidate_answer",
    "ground_truth_answer",
    "human_label",
]


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def format_number(value: Any, digits: int = 2) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def build_tables(
    summary: list[dict[str, Any]], ablation: list[dict[str, Any]], stats: dict[str, Any], curve: list[dict[str, Any]]
) -> str:
    sections = []
    headers = list(summary[0].keys())
    sections.append("## Results\n\n" + markdown_table(headers, [[row[h] for h in headers] for row in summary]))

    interval_rows = [
        [name, format_number(v["accuracy"]), f"[{v['ci_low']:.2f}, {v['ci_high']:.2f}]"]
        for name, v in stats["accuracy_confidence_intervals"].items()
    ]
    sections.append(
        "## Accuracy with 95% bootstrap intervals (1000 resamples, seed 5326)\n\n"
        + markdown_table(["Method", "Accuracy (%)", "95% CI"], interval_rows)
    )

    paired_rows = []
    for name, v in stats["paired_comparisons"].items():
        paired_rows.append(
            [
                name,
                v["first_wins"],
                v["second_wins"],
                v["agreed"],
                format_number(v["accuracy_difference"]),
                f"[{v['difference_ci_low']:.2f}, {v['difference_ci_high']:.2f}]",
                format_number(v["sign_test_p_value"], 4),
            ]
        )
    sections.append(
        "## Paired win / loss counts\n\n"
        + markdown_table(["Comparison", "First wins", "Second wins", "Agreed", "Diff (pp)", "Diff 95% CI", "Sign-test p"], paired_rows)
    )

    ablation_rows = [
        [row["Strategy"], format_number(row["Lambda"], 1), row["Accuracy (%)"], row["Tie Rate (%)"]] for row in ablation
    ]
    sections.append("## Aggregation ablation\n\n" + markdown_table(["Strategy", "Lambda", "Accuracy (%)", "Tie Rate (%)"], ablation_rows))

    curve_rows = [[row["N"], row["Accuracy (%)"], row["Avg Gen Tokens / Problem"]] for row in curve]
    sections.append("## Self-consistency scaling curve\n\n" + markdown_table(["N", "Accuracy (%)", "Avg Gen Tokens / Problem"], curve_rows))

    budget = stats["budget"]
    budget_rows = [
        ["Greedy generated tokens", budget["greedy_generated_tokens"]],
        ["10x greedy budget (tokens)", budget["budget_tokens"]],
        ["PRM Best-of-5 total tokens (gen + judge)", budget["prm_total_tokens"]],
        ["PRM budget fraction", format_number(budget["prm_budget_fraction"], 3)],
        ["Largest SC N within budget", budget["largest_sc_n_within_budget"]],
        ["SC accuracy at that N (%)", budget["sc_accuracy_at_largest_n_within_budget"]],
        ["SC N=15 total tokens", budget["sc_max_n_total_tokens"]],
        ["SC N=15 budget fraction", format_number(budget["sc_max_n_budget_fraction"], 3)],
    ]
    sections.append("## 10x greedy compute budget\n\n" + markdown_table(["Quantity", "Value"], budget_rows))

    asymmetry = stats["cost_asymmetry"]
    asymmetry_rows = [[key, format_number(value, 3)] for key, value in asymmetry.items()]
    sections.append("## Generator vs judge cost\n\n" + markdown_table(["Quantity", "Value"], asymmetry_rows))

    judge_rows = [[key, format_number(value, 4)] for key, value in stats["judge"].items()]
    sections.append("## Judge statistics\n\n" + markdown_table(["Quantity", "Value"], judge_rows))

    candidate_rows = [[key, format_number(value, 4)] for key, value in stats["candidates"].items()]
    sections.append("## Candidate statistics\n\n" + markdown_table(["Quantity", "Value"], candidate_rows))
    return "\n\n".join(sections) + "\n"


def make_figures(summary: list[dict[str, Any]], curve: list[dict[str, Any]], stats: dict[str, Any], records: list[dict[str, Any]], ablation: list[dict[str, Any]], directory: str) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(directory, exist_ok=True)
    count = len(records)
    rows = {row["Method"]: row for row in summary}
    intervals = stats["sc_curve_confidence_intervals"]
    budget = stats["budget"]
    colors = {"greedy": "#4d4d4d", "min": "#c0392b", "product": "#e67e22", "step_penalty_l2.0": "#8e44ad"}
    paths = []

    sc_calls = [row["N"] * count for row in curve]
    sc_accuracy = [row["Accuracy (%)"] for row in curve]
    low = [item["ci_low"] for item in intervals]
    high = [item["ci_high"] for item in intervals]

    figure, axis = plt.subplots(figsize=(7.5, 5))
    axis.plot(sc_calls, sc_accuracy, marker="o", color="#2471a3", label="Self-consistency (N=1..15)")
    axis.fill_between(sc_calls, low, high, color="#2471a3", alpha=0.15, label="SC 95% bootstrap CI")
    greedy = rows[ev.METHOD_NAMES["greedy"]]
    axis.scatter([greedy["Total LLM Calls"]], [greedy["Accuracy (%)"]], marker="*", s=180, color=colors["greedy"], label="Greedy", zorder=5)
    span = max(sc_calls) - min(sc_calls)
    for offset, key in zip((-1, 0, 1), ev.PRM_KEYS):
        row = rows[ev.METHOD_NAMES[key]]
        interval = stats["accuracy_confidence_intervals"][ev.METHOD_NAMES[key]]
        axis.errorbar(
            [row["Total LLM Calls"] + offset * 0.01 * span],
            [row["Accuracy (%)"]],
            yerr=[[row["Accuracy (%)"] - interval["ci_low"]], [interval["ci_high"] - row["Accuracy (%)"]]],
            fmt=MARKERS[key],
            markersize=8,
            capsize=3,
            color=colors[key],
            label=ev.METHOD_NAMES[key],
        )
    axis.set_xlabel(f"Total LLM calls over {count} problems")
    axis.set_ylabel("Accuracy (%)")
    axis.set_title("Accuracy vs total LLM calls")
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8, loc="best")
    figure.tight_layout()
    path = os.path.join(directory, "accuracy_vs_calls.png")
    figure.savefig(path, dpi=200)
    plt.close(figure)
    paths.append(path)

    sc_tokens = [item["total_tokens"] for item in budget["sc_curve_tokens"]]
    figure, axis = plt.subplots(figsize=(7.5, 5))
    axis.plot(sc_tokens, sc_accuracy, marker="o", color="#2471a3", label="Self-consistency (N=1..15)")
    axis.fill_between(sc_tokens, low, high, color="#2471a3", alpha=0.15, label="SC 95% bootstrap CI")
    axis.scatter([budget["greedy_generated_tokens"]], [greedy["Accuracy (%)"]], marker="*", s=180, color=colors["greedy"], label="Greedy", zorder=5)
    span = max(sc_tokens) - min(sc_tokens)
    for offset, key in zip((-1, 0, 1), ev.PRM_KEYS):
        row = rows[ev.METHOD_NAMES[key]]
        interval = stats["accuracy_confidence_intervals"][ev.METHOD_NAMES[key]]
        axis.errorbar(
            [budget["prm_total_tokens"] + offset * 0.01 * span],
            [row["Accuracy (%)"]],
            yerr=[[row["Accuracy (%)"] - interval["ci_low"]], [interval["ci_high"] - row["Accuracy (%)"]]],
            fmt=MARKERS[key],
            markersize=8,
            capsize=3,
            color=colors[key],
            label=ev.METHOD_NAMES[key],
        )
    axis.axvline(budget["budget_tokens"], color="black", linestyle="--", linewidth=1, label="10x greedy tokens")
    axis.set_xlabel(f"Total output tokens (generated + judge) over {count} problems")
    axis.set_ylabel("Accuracy (%)")
    axis.set_title("Accuracy vs total output tokens")
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8, loc="best")
    figure.tight_layout()
    path = os.path.join(directory, "accuracy_vs_tokens.png")
    figure.savefig(path, dpi=200)
    plt.close(figure)
    paths.append(path)

    labels = [ev.variant_key(row["Strategy"], row["Lambda"]).replace("_l", "\nl=") for row in ablation]
    figure, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].bar(labels, [row["Accuracy (%)"] for row in ablation], color="#2471a3")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_title("Aggregation ablation: accuracy")
    axes[1].bar(labels, [row["Tie Rate (%)"] for row in ablation], color="#c0392b")
    axes[1].set_ylabel("Tie rate (%)")
    axes[1].set_title("Aggregation ablation: tie rate")
    for axis in axes:
        axis.tick_params(axis="x", labelsize=8)
        axis.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    path = os.path.join(directory, "aggregation_ablation.png")
    figure.savefig(path, dpi=200)
    plt.close(figure)
    paths.append(path)

    steps = [len(c["step_scores"]) for r in records for c in r["candidates"][: ev.POOL_SIZE]]
    invalid = [sum(1 for s in c["step_scores"] if s < 0) for r in records for c in r["candidates"][: ev.POOL_SIZE]]
    figure, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].hist(steps, bins=range(0, max(steps + [1]) + 2), color="#2471a3", align="left", rwidth=0.9)
    axes[0].set_xlabel("Steps per path")
    axes[0].set_ylabel("Paths")
    axes[0].set_title("Path length")
    axes[1].hist(invalid, bins=range(0, max(invalid + [1]) + 2), color="#c0392b", align="left", rwidth=0.9)
    axes[1].set_xlabel("Invalid steps per path")
    axes[1].set_ylabel("Paths")
    axes[1].set_title("Judge-flagged invalid steps")
    for axis in axes:
        axis.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    path = os.path.join(directory, "path_statistics.png")
    figure.savefig(path, dpi=200)
    plt.close(figure)
    paths.append(path)
    return paths


def write_audit_sample(rows: list[dict[str, Any]], path: str) -> bool:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8", newline="") as handle:
            if any((row.get("human_label") or "").strip() for row in csv.DictReader(handle)):
                return False
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=AUDIT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in AUDIT_COLUMNS})
    return True


def finalize_audit(output_dir: str) -> dict[str, Any] | None:
    path = os.path.join(output_dir, "judge_audit", "audit_sample.csv")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    result = ev.audit_confusion(rows)
    if result["labelled_steps"] == 0:
        return None
    write_json(result, os.path.join(output_dir, "judge_audit", "audit_confusion.json"))
    return result


def write_artifacts(records: list[dict[str, Any]], output_dir: str) -> dict[str, Any]:
    records = sorted(records, key=lambda r: r["index"])
    summary = ev.results_summary(records)
    ablation = ev.aggregation_ablation(records)
    curve = ev.sc_scaling_curve(records)
    stats = ev.analysis(records, curve)
    failures = ev.find_failure_cases(records)

    write_json(summary, os.path.join(output_dir, "results_summary.json"))
    write_json(ablation, os.path.join(output_dir, "aggregation_ablation.json"))
    write_json(curve, os.path.join(output_dir, "sc_scaling_curve.json"))
    write_jsonl(records, os.path.join(output_dir, "per_problem.jsonl"))
    write_json(stats, os.path.join(output_dir, "analysis.json"))
    write_json(failures, os.path.join(output_dir, "failure_cases.json"))
    write_json(stats["bonus_product_logprob"], os.path.join(output_dir, "bonus_product_logprob.json"))

    tables_path = os.path.join(output_dir, "tables", "report_tables.md")
    os.makedirs(os.path.dirname(tables_path), exist_ok=True)
    with open(tables_path, "w", encoding="utf-8") as handle:
        handle.write(build_tables(summary, ablation, stats, curve))

    figures = make_figures(summary, curve, stats, records, ablation, os.path.join(output_dir, "figures"))
    write_audit_sample(ev.sample_audit_steps(records), os.path.join(output_dir, "judge_audit", "audit_sample.csv"))
    audit = finalize_audit(output_dir)
    return {"summary": summary, "ablation": ablation, "curve": curve, "analysis": stats, "figures": figures, "audit": audit}

