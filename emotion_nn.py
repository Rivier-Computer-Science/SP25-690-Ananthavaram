import argparse
import inspect

import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

LABEL_NAMES = ["sadness", "joy", "love", "anger", "fear", "surprise"]


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")
    return {"accuracy": acc, "f1": f1}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a DistilBERT emotion classifier on dair-ai/emotion."
    )
    parser.add_argument(
        "--full-dataset",
        action="store_true",
        help="Use the full training and validation splits instead of a quicker CPU-friendly subset.",
    )
    parser.add_argument(
        "--epochs",
        type=float,
        default=1.0,
        help="Number of training epochs. Use a small value for quick testing.",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Limit the number of training examples.",
    )
    parser.add_argument(
        "--max-eval-samples",
        type=int,
        default=None,
        help="Limit the number of validation examples.",
    )
    return parser.parse_args()


def train_emotion_classifier(args):
    """
    Implements the first neural network stage for emotion classification
    using a transformer model (DistilBERT).
    """
    model_name = "distilbert-base-uncased"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # The full dataset is slow on CPU, so default to a smaller subset unless
    # the user explicitly opts into a full training run.
    max_train_samples = args.max_train_samples
    max_eval_samples = args.max_eval_samples
    if not args.full_dataset and max_train_samples is None:
        max_train_samples = 2000
    if not args.full_dataset and max_eval_samples is None:
        max_eval_samples = 500

    print(f"Using device: {device}")
    print(f"Loading tokenizer and model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print("Loading dataset...")
    dataset = load_dataset("dair-ai/emotion")

    if max_train_samples is not None:
        dataset["train"] = dataset["train"].select(
            range(min(max_train_samples, len(dataset["train"])))
        )
    if max_eval_samples is not None:
        dataset["validation"] = dataset["validation"].select(
            range(min(max_eval_samples, len(dataset["validation"])))
        )

    print(
        "Dataset sizes -> "
        f"train: {len(dataset['train'])}, validation: {len(dataset['validation'])}"
    )

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=128,
        )

    print("Tokenizing dataset...")
    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # 6 labels in dair-ai/emotion: sadness, joy, love, anger, fear, surprise
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=6,
        id2label={idx: label for idx, label in enumerate(LABEL_NAMES)},
        label2id={label: idx for idx, label in enumerate(LABEL_NAMES)},
    )

    training_args_kwargs = {
        "output_dir": "./results_emotion_model",
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "learning_rate": 2e-5,
        "per_device_train_batch_size": 8 if device == "cpu" else 16,
        "per_device_eval_batch_size": 8 if device == "cpu" else 16,
        "num_train_epochs": args.epochs,
        "weight_decay": 0.01,
        "report_to": "none",
        "fp16": torch.cuda.is_available(),
        "dataloader_pin_memory": torch.cuda.is_available(),
    }
    if "eval_strategy" in inspect.signature(TrainingArguments.__init__).parameters:
        training_args_kwargs["eval_strategy"] = "epoch"
    else:
        training_args_kwargs["evaluation_strategy"] = "epoch"

    training_args = TrainingArguments(**training_args_kwargs)

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": tokenized_datasets["train"],
        "eval_dataset": tokenized_datasets["validation"],
        "compute_metrics": compute_metrics,
    }
    trainer_signature = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in trainer_signature:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_signature:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = Trainer(**trainer_kwargs)

    print("Starting training of the emotion classifier...")
    trainer.train()

    model_save_path = "./saved_emotion_model"
    trainer.save_model(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    print(f"Model saved to {model_save_path}")


if __name__ == "__main__":
    train_emotion_classifier(parse_args())
