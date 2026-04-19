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
    parser = argparse.ArgumentParser(
        description="Compare baseline and multiple emotion-aware chatbot models on a shared prompt set."
    )
    parser.add_argument(
        "--prompts-file",
        type=Path,
        default=DEFAULT_PROMPTS_PATH,
        help="Path to a JSON file containing evaluation prompts.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="CSV file where the comparison report will be written.",
    )
    parser.add_argument(
        "--summary-file",
        type=Path,
        default=DEFAULT_SUMMARY_PATH,
        help="CSV file where the per-model summary will be written.",
    )
    parser.add_argument(
        "--metrics-file",
        type=Path,
        default=DEFAULT_METRICS_PATH,
        help="CSV file where the mathematical model metrics will be written.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a smaller 3-prompt comparison for faster terminal feedback.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Model aliases to compare. Defaults to all configured models.",
    )
    parser.add_argument(
        "--print-all",
        action="store_true",
        help="Print every comparison result to the terminal instead of only a short sample.",
    )
    return parser.parse_args()


def load_prompts(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        prompts = json.load(handle)
    if not isinstance(prompts, list):
        raise ValueError("Prompts file must contain a JSON list.")
    return prompts


def resolve_requested_models(requested_models):
    available_models = get_available_model_sources()
    if requested_models:
        return [(model_name, available_models.get(model_name, model_name)) for model_name in requested_models]
    return list(available_models.items())


def write_csv(path: Path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def calculate_model_metrics(model_rows):
    labeled_rows = [row for row in model_rows if isinstance(row["emotion_match"], bool)]
    if not labeled_rows:
        return {
            "matched_labeled_prompts": 0,
            "labeled_prompts": 0,
            "accuracy": "",
            "precision_weighted": "",
            "recall_weighted": "",
            "f1_weighted": "",
            "f1_macro": "",
        }

    expected = [row["expected_emotion"] for row in labeled_rows]
    predicted = [row["predicted_emotion"] for row in labeled_rows]
    correct = sum(1 for row in labeled_rows if row["emotion_match"])
    total = len(labeled_rows)
    return {
        "matched_labeled_prompts": correct,
        "labeled_prompts": total,
        "accuracy": f"{accuracy_score(expected, predicted):.4f}",
        "precision_weighted": f"{precision_score(expected, predicted, average='weighted', zero_division=0):.4f}",
        "recall_weighted": f"{recall_score(expected, predicted, average='weighted', zero_division=0):.4f}",
        "f1_weighted": f"{f1_score(expected, predicted, average='weighted', zero_division=0):.4f}",
        "f1_macro": f"{f1_score(expected, predicted, average='macro', zero_division=0):.4f}",
    }


def run_comparison(prompts, model_specs, output_path: Path, summary_path: Path, metrics_path: Path):
    start_time = time.time()
    rows = []
    summary_rows = []
    metric_rows = []
    baseline_cache = {}

    for model_name, model_source in model_specs:
        print(f"Loading emotion classifier: {model_name} ({model_source})...")
        try:
            classifier = load_emotion_pipeline(model_name=model_name)
        except Exception as exc:
            print(f"  Failed to load model '{model_name}': {exc}\n")
            summary_rows.append(
                {
                    "model": model_name,
                    "model_source": model_source,
                    "status": "load_failed",
                    "matched_labeled_prompts": "",
                    "labeled_prompts": "",
                    "match_rate": "",
                    "accuracy": "",
                    "precision_weighted": "",
                    "recall_weighted": "",
                    "f1_weighted": "",
                    "f1_macro": "",
                    "elapsed_seconds": "",
                    "error": str(exc),
                }
            )
            continue

        print("Classifier loaded. Running prompt comparison...\n")
        model_start_time = time.time()
        model_rows = []

        for index, prompt in enumerate(prompts, start=1):
            prompt_id = prompt.get("id", "")
            text = prompt["text"]
            expected_emotion = prompt.get("expected_emotion", "")

            print(f"[{model_name} {index}/{len(prompts)}] Evaluating {prompt_id or 'prompt'}")

            baseline_response = baseline_cache.setdefault(text, baseline_chatbot(text))
            emotion_result = emotion_chatbot(text, classifier=classifier)
            predicted_emotion = emotion_result["emotion"]
            emotion_match = (
                expected_emotion == predicted_emotion if expected_emotion and expected_emotion != "neutral" else ""
            )

            row = {
                "model": model_name,
                "model_source": model_source,
                "id": prompt_id,
                "text": text,
                "expected_emotion": expected_emotion,
                "predicted_emotion": predicted_emotion,
                "raw_label": emotion_result["raw_label"],
                "emotion_score": f"{emotion_result['score']:.4f}",
                "emotion_match": emotion_match,
                "baseline_response": baseline_response,
                "emotion_aware_response": emotion_result["response"],
            }
            rows.append(row)
            model_rows.append(row)

            print(
                "  Predicted emotion: "
                f"{predicted_emotion} "
                f"(raw={emotion_result['raw_label']}, score={emotion_result['score']:.4f})"
            )
            print("  Comparison recorded.\n")

        metrics = calculate_model_metrics(model_rows)
        model_elapsed = time.time() - model_start_time
        summary_rows.append(
            {
                "model": model_name,
                "model_source": model_source,
                "status": "ok",
                "matched_labeled_prompts": metrics["matched_labeled_prompts"],
                "labeled_prompts": metrics["labeled_prompts"],
                "match_rate": (
                    f"{(metrics['matched_labeled_prompts'] / metrics['labeled_prompts']):.4f}"
                    if metrics["labeled_prompts"]
                    else ""
                ),
                "accuracy": metrics["accuracy"],
                "precision_weighted": metrics["precision_weighted"],
                "recall_weighted": metrics["recall_weighted"],
                "f1_weighted": metrics["f1_weighted"],
                "f1_macro": metrics["f1_macro"],
                "elapsed_seconds": f"{model_elapsed:.2f}",
                "error": "",
            }
        )
        metric_rows.append(
            {
                "model": model_name,
                "model_source": model_source,
                "accuracy": metrics["accuracy"],
                "precision_weighted": metrics["precision_weighted"],
                "recall_weighted": metrics["recall_weighted"],
                "f1_weighted": metrics["f1_weighted"],
                "f1_macro": metrics["f1_macro"],
                "matched_labeled_prompts": metrics["matched_labeled_prompts"],
                "labeled_prompts": metrics["labeled_prompts"],
                "elapsed_seconds": f"{model_elapsed:.2f}",
            }
        )
        print(
            f"Finished model '{model_name}': "
            f"{metrics['matched_labeled_prompts']}/{metrics['labeled_prompts']} labeled prompts matched "
            f"in {model_elapsed:.2f} seconds.\n"
        )

    write_csv(output_path, rows)
    write_csv(summary_path, summary_rows)
    write_csv(metrics_path, metric_rows)
    elapsed = time.time() - start_time
    return rows, summary_rows, metric_rows, elapsed


def print_summary(
    rows,
    summary_rows,
    metric_rows,
    output_path: Path,
    summary_path: Path,
    metrics_path: Path,
    elapsed: float,
    print_all: bool,
):
    print(f"Wrote comparison report to: {output_path}")
    print(f"Wrote model summary to: {summary_path}")
    print(f"Wrote mathematical metrics to: {metrics_path}")
    print(f"Comparison rows written: {len(rows)}")
    print(f"Elapsed time: {elapsed:.2f} seconds")

    print("\nModel summary:")
    for row in summary_rows:
        if row["status"] != "ok":
            print(f"- {row['model']}: failed to load")
            continue
        print(
            f"- {row['model']}: "
            f"{row['matched_labeled_prompts']}/{row['labeled_prompts']} "
            f"labeled prompts matched"
        )

    print("\nMathematical comparison:")
    for row in metric_rows:
        print(
            f"- {row['model']}: "
            f"accuracy={row['accuracy']}, "
            f"precision_w={row['precision_weighted']}, "
            f"recall_w={row['recall_weighted']}, "
            f"f1_w={row['f1_weighted']}, "
            f"f1_macro={row['f1_macro']}"
        )

    results_to_print = rows if print_all else rows[:3]
    heading = "All results:" if print_all else "Sample results:"
    print(f"\n{heading}")
    for row in results_to_print:
        print(f"- {row['model']} / {row['id'] or '[no id]'}")
        print(f"  User: {row['text']}")
        print(f"  Baseline: {row['baseline_response']}")
        print(
            "  Emotion-aware: "
            f"{row['emotion_aware_response']} "
            f"[emotion={row['predicted_emotion']}, raw={row['raw_label']}, score={row['emotion_score']}]"
        )


def main():
    args = parse_args()
    prompts = load_prompts(args.prompts_file)
    if args.quick:
        prompts = prompts[:3]
        print("Quick mode enabled: using the first 3 prompts.\n")
    model_specs = resolve_requested_models(args.models)
    print(f"Comparing models: {', '.join(model_name for model_name, _ in model_specs)}\n")
    rows, summary_rows, metric_rows, elapsed = run_comparison(
        prompts,
        model_specs,
        args.output_file,
        args.summary_file,
        args.metrics_file,
    )
    print_summary(
        rows,
        summary_rows,
        metric_rows,
        args.output_file,
        args.summary_file,
        args.metrics_file,
        elapsed,
        args.print_all,
    )


if __name__ == "__main__":
    main()
