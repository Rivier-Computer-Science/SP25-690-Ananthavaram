import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')
    return {'accuracy': acc, 'f1': f1}

def train_emotion_classifier():
    """
    Implements the first neural network stage for emotion classification
    using a transformer model (DistilBERT).
    """
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Using 'dair-ai/emotion' dataset which contains text and emotion labels
    print("Loading dataset...")
    # 'trust_remote_code=True' might be needed depending on datasets version, but dair-ai/emotion usually works fine.
    dataset = load_dataset("dair-ai/emotion", trust_remote_code=True)
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)
    
    print("Tokenizing dataset...")
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    
    # 6 labels in dair-ai/emotion: sadness, joy, love, anger, fear, surprise
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=6)
    
    training_args = TrainingArguments(
        output_dir="./results_emotion_model",
        eval_strategy="epoch",  # eval_strategy instead of evaluation_strategy for newer transformers max compatibility
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        save_strategy="epoch",
        load_best_model_at_end=True,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        compute_metrics=compute_metrics,
        processing_class=tokenizer,
    )
    
    print("Starting training of the first neural network (Emotion Classifier)...")
    trainer.train()
    
    # Save the model
    model_save_path = "./saved_emotion_model"
    trainer.save_model(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    print(f"Model saved to {model_save_path}")

if __name__ == "__main__":
    train_emotion_classifier()
