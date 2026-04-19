# Emotion-Aware Chatbot Using Transformer Models

## Overview

This project explores how a chatbot can first identify a user's emotion and then produce a more appropriate reply. The main goal is to compare a simple baseline chatbot with an emotion-aware chatbot that uses transformer-based emotion classification.

## Problem Statement

Many chatbots respond in a generic way without recognizing how the user feels. If a user says they are sad, stressed, or angry, a standard chatbot may still reply with the same neutral tone. This project aims to make chatbot responses more natural and helpful by adding an emotion detection stage before response generation.

## Project Goal

The system is designed in two stages:

1. An emotion classifier predicts the emotion in a user's message.
2. A chatbot generates a reply that is guided by the detected emotion.

The expected outcome is that the emotion-aware chatbot will respond more appropriately than a baseline chatbot that does not use emotion information.

## Datasets

- `dair-ai/emotion` is used for emotion classification.
- A future extension can use `DailyDialog` for response-generation experiments.

## Current Implementation

This repository currently includes:

- `baseline_chatbot.py`: a simple baseline chatbot with general rule-based responses.
- `emotion_chatbot.py`: an emotion-aware chatbot that detects emotion and adapts its response.
- `emotion_nn.py`: a transformer training script for emotion classification using DistilBERT.
- `compare_chatbots.py`: a shared evaluation script that compares the baseline chatbot plus multiple emotion-classifier models on the same prompts and writes CSV reports.

## Method

### Stage 1: Emotion Classification

The project trains a transformer-based classifier using `distilbert-base-uncased` on the `dair-ai/emotion` dataset. The model predicts one of the supported emotion labels such as sadness, joy, anger, fear, love, or surprise.

### Stage 2: Emotion-Aware Response

The chatbot uses the predicted emotion to choose a more suitable response style. For example, a sad message should lead to a more supportive reply, while a joyful message should lead to a more positive reply.

## Baseline Method

The baseline chatbot does not use any emotion detection. It responds using simple general-purpose rules. This gives a comparison point for the emotion-aware version.

## Evaluation Plan

For emotion classification:

- Accuracy
- Weighted F1-score

For chatbot quality:

- Manual comparison between baseline and emotion-aware replies
- Shared prompt evaluation using `compare_chatbots.py`
- Future extension: BLEU or other text-generation metrics if a generative response model is added

## How To Run

### 1. Create or activate the environment

If you are using the local virtual environment already included in the project:

```powershell
.\.venv\Scripts\python.exe --version
```

### 2. Train the emotion classifier

Quick local test:

```powershell
.\.venv\Scripts\python.exe emotion_nn.py --epochs 0.01 --max-train-samples 64 --max-eval-samples 32
```

Smaller default CPU-friendly run:

```powershell
.\.venv\Scripts\python.exe emotion_nn.py
```

Full training run:

```powershell
.\.venv\Scripts\python.exe emotion_nn.py --full-dataset --epochs 3
```

### 3. Run the baseline chatbot

```powershell
.\.venv\Scripts\python.exe baseline_chatbot.py
```

### 4. Run the emotion-aware chatbot

```powershell
.\.venv\Scripts\python.exe emotion_chatbot.py
```

If `saved_emotion_model` exists, the chatbot will try to use that local model first. Otherwise it falls back to a public Hugging Face emotion model for demonstration.

### 5. Run the work dashboard

```powershell
.\.venv\Scripts\streamlit.exe run dashboard.py
```

The dashboard shows the current project files, supports manual refresh, and can auto-refresh on a timer so you can keep an eye on the workspace while you work.

### 6. Compare chatbot outputs on a shared test set

```powershell
.\.venv\Scripts\python.exe compare_chatbots.py
```

This runs the baseline chatbot plus all configured emotion models against the shared prompts in `evaluation_prompts.json` and writes detailed results to `comparison_results.csv` and per-model scores to `model_comparison_summary.csv`.
It also writes mathematical metrics such as accuracy, weighted precision, weighted recall, weighted F1, and macro F1 to `model_metric_comparison.csv`.

For a faster terminal check with only the first 3 prompts:

```powershell
.\.venv\Scripts\python.exe compare_chatbots.py --quick
```

To compare only specific models:

```powershell
.\.venv\Scripts\python.exe compare_chatbots.py --quick --models local hartmann bhadresh
```

To print every comparison result in the terminal:

```powershell
.\.venv\Scripts\python.exe compare_chatbots.py --quick --models local hartmann --print-all
```

## Example

Input:

```text
I feel stressed about my project submission.
```

Possible baseline response:

```text
That sounds important. Tell me what part you want help with.
```

Possible emotion-aware response:

```text
Detected emotion: fear
That sounds stressful. Let's take it one step at a time.
```

## Limitations

- The current chatbot response system is still template-based, not a full generative dialogue model.
- If the emotion classifier predicts the wrong label, the chatbot response may also be inappropriate.
- This system is for educational and research purposes only.

## Ethics and Responsible Use

This chatbot should not be used for medical, legal, crisis, or mental health decisions. It may misread emotions and should only be treated as a simple conversational support tool.

## Roadmap

- Improve the emotion classifier with more tuning and evaluation
- Add a true response-generation model
- Compare baseline and emotion-aware outputs on a shared test set
- Add automatic evaluation for chatbot responses

## References

1. Devlin, J., Chang, M. W., Lee, K., and Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. Proceedings of NAACL-HLT.
2. Li, Y., Su, H., Shen, X., Li, W., Cao, Z., and Niu, S. (2017). DailyDialog: A manually labelled multi-turn dialogue dataset. Proceedings of IJCNLP.
3. Papineni, K., Roukos, S., Ward, T., and Zhu, W. J. (2002). BLEU: a method for automatic evaluation of machine translation. Proceedings of ACL.
4. Wolf, T., Debut, L., Sanh, V., et al. (2020). Transformers: State-of-the-art natural language processing. Proceedings of EMNLP: System Demonstrations.
