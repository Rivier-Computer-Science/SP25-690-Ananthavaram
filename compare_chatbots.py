import argparse
import csv
import json
import time
from pathlib import Path

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from baseline_chatbot import baseline_chatbot
from emotion_chatbot import emotion_chatbot, get_available_model_sources, load_emotion_pipeline

DEFAULT_PROMPTS_PATH = Path("evaluation_prompts.json")
DEFAULT_OUTPUT_PATH = Path("comparison_results.csv")
DEFAULT_SUMMARY_PATH = Path("model_comparison_summary.csv")
DEFAULT_METRICS_PATH = Path("model_metric_comparison.csv")


def parse_args():
    parser = argparse.ArgumentParser(description="Compare chatbot models")
    parser.add_argument("--prompts-file", type=Path, default=DEFAULT_PROMPTS_PATH)
    parser.add_argument("--output-file", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--summary-file", type=Path, default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--metrics-file", type=Path, default=DEFAULT_METRICS_PATH)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--print-all", action="store_true")
    return parser.parse_args()


def load_prompts(path: Path):
    with path.open("r", encoding="utf-8") as f:
        prompts = json.load(f)
    return prompts if isinstance(prompts, list) else []


def resolve_requested_models(requested):
    available = get_available_model_sources()
    if requested:
        return [(m, available.get(m, m)) for m in requested]
    return list(available.items())


def write_csv(path: Path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def calculate_model_metrics(rows):
    labeled = [r for r in rows if isinstance(r["emotion_match"], bool)]
    if not labeled:
        return {
            "matched_labeled_prompts": 0,
            "labeled_prompts": 0,
            "accuracy": "0",
            "precision_weighted": "0",
            "recall_weighted": "0",
            "f1_weighted": "0",
            "f1_macro": "0",
        }

    expected = [r["expected_emotion"] for r in labeled]
    predicted = [r["predicted_emotion"] for r in labeled]
    correct = sum(1 for r in labeled if r["emotion_match"])
    total = len(labeled)

    return {
        "matched_labeled_prompts": correct,
        "labeled_prompts": total,
        "accuracy": f"{accuracy_score(expected, predicted):.4f}",
        "precision_weighted": f"{precision_score(expected, predicted, average='weighted', zero_division=0):.4f}",
        "recall_weighted": f"{recall_score(expected, predicted, average='weighted', zero_division=0):.4f}",
        "f1_weighted": f"{f1_score(expected, predicted, average='weighted', zero_division=0):.4f}",
        "f1_macro": f"{f1_score(expected, predicted, average='macro', zero_division=0):.4f}",
    }


def rank_metric_rows(metric_rows):
    def safe_float(x):
        try:
            return float(x)
        except:
            return 0.0

    ranked = sorted(
        metric_rows,
        key=lambda r: (
            safe_float(r["f1_weighted"]),
            safe_float(r["accuracy"]),
            safe_float(r["precision_weighted"]),
            safe_float(r["recall_weighted"]),
        ),
        reverse=True,
    )

    return [{**r, "rank": i + 1} for i, r in enumerate(ranked)]


def run_comparison(prompts, model_specs, output_path, summary_path, metrics_path):
    start = time.time()
    rows = []
    summary = []
    metrics_rows = []
    baseline_cache = {}

    for model_name, model_source in model_specs:
        print(f"Loading: {model_name}")
        try:
            classifier = load_emotion_pipeline(model_name=model_name)
        except Exception as e:
            summary.append(
                {
                    "model": model_name,
                    "model_source": model_source,
                    "status": "failed",
                    "error": str(e),
                }
            )
            continue

        model_rows = []
        t0 = time.time()

        for i, prompt in enumerate(prompts):
            text = prompt["text"]
            expected = prompt.get("expected_emotion", "")

            baseline_resp = baseline_cache.setdefault(text, baseline_chatbot(text))
            result = emotion_chatbot(text, classifier)

            predicted = result["emotion"]
            match = expected == predicted if expected and expected != "neutral" else ""

            row = {
                "model": model_name,
                "model_source": model_source,
                "text": text,
                "expected_emotion": expected,
                "predicted_emotion": predicted,
                "emotion_score": f"{result['score']:.4f}",
                "emotion_match": match,
                "baseline_response": baseline_resp,
                "emotion_aware_response": result["response"],
            }

            rows.append(row)
            model_rows.append(row)

        m = calculate_model_metrics(model_rows)
        elapsed = time.time() - t0

        summary.append(
            {
                "model": model_name,
                "status": "ok",
                "matched": m["matched_labeled_prompts"],
                "total": m["labeled_prompts"],
                "accuracy": m["accuracy"],
                "f1": m["f1_weighted"],
                "time": f"{elapsed:.2f}",
            }
        )

        metrics_rows.append(
            {
                "model": model_name,
                "accuracy": m["accuracy"],
                "precision_weighted": m["precision_weighted"],
                "recall_weighted": m["recall_weighted"],
                "f1_weighted": m["f1_weighted"],
                "f1_macro": m["f1_macro"],
                "matched": m["matched_labeled_prompts"],
                "total": m["labeled_prompts"],
                "time": f"{elapsed:.2f}",
            }
        )

    ranked = rank_metric_rows(metrics_rows)

    write_csv(output_path, rows)
    write_csv(summary_path, summary)
    write_csv(metrics_path, ranked)

    return rows, summary, ranked, time.time() - start


def print_summary(rows, summary, metrics, elapsed, print_all):
    print(f"\nDone in {elapsed:.2f}s")

    for s in summary:
        print(f"{s['model']} -> {s.get('accuracy')} / {s.get('f1')}")

    for m in metrics:
        print(f"#{m['rank']} {m['model']} f1={m['f1_weighted']}")

    sample = rows if print_all else rows[:3]
    for r in sample:
        print("\nUser:", r["text"])
        print("Baseline:", r["baseline_response"])
        print("Emotion:", r["emotion_aware_response"])


def main():
    args = parse_args()
    prompts = load_prompts(args.prompts_file)

    if args.quick:
        prompts = prompts[:3]

    models = resolve_requested_models(args.models)

    rows, summary, metrics, elapsed = run_comparison(
        prompts,
        models,
        args.output_file,
        args.summary_file,
        args.metrics_file,
    )

    print_summary(rows, summary, metrics, elapsed, args.print_all)


if __name__ == "__main__":
    main()
