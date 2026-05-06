from pathlib import Path
from transformers import pipeline

DEFAULT_REMOTE_MODEL = "j-hartmann/emotion-english-distilroberta-base"

LOCAL_MODEL_PATH = Path("saved_emotion_model")

MODEL_SOURCES = {
    "local": str(LOCAL_MODEL_PATH),
    "hartmann": "j-hartmann/emotion-english-distilroberta-base",
    "bhadresh": "bhadresh-savani/distilbert-base-uncased-emotion",
    "go_emotions": "SamLowe/roberta-base-go_emotions",
}

LOCAL_LABEL_MAP = {
    "label_0": "sadness",
    "label_1": "joy",
    "label_2": "love",
    "label_3": "anger",
    "label_4": "fear",
    "label_5": "surprise",
}

EMOTION_RESPONSES = {
    "sadness": "I understand this feels difficult. Would you like to talk more about what is bothering you?",
    "joy": "That sounds really good to hear. What made you feel this way?",
    "anger": "It seems something is frustrating you. Do you want to explain what happened?",
    "fear": "That sounds stressful. We can go through it step by step.",
    "love": "That sounds meaningful. It seems important to you.",
    "surprise": "That sounds unexpected. What happened after that?",
    "neutral": "I understand. Please tell me more so I can better understand.",
}

CANONICAL_EMOTION_MAP = {
    "admiration": "love",
    "amusement": "joy",
    "anger": "anger",
    "annoyance": "anger",
    "approval": "joy",
    "caring": "love",
    "confusion": "surprise",
    "curiosity": "surprise",
    "desire": "love",
    "disappointment": "sadness",
    "disapproval": "anger",
    "disgust": "anger",
    "embarrassment": "fear",
    "excitement": "joy",
    "fear": "fear",
    "gratitude": "love",
    "grief": "sadness",
    "joy": "joy",
    "love": "love",
    "nervousness": "fear",
    "optimism": "joy",
    "pride": "joy",
    "realization": "surprise",
    "relief": "joy",
    "remorse": "sadness",
    "sadness": "sadness",
    "surprise": "surprise",
    "neutral": "neutral",
}


def get_available_model_sources():
    available = dict(MODEL_SOURCES)
    if not LOCAL_MODEL_PATH.exists():
        available.pop("local", None)
    return available


def resolve_model_source(model_name=None):
    available = get_available_model_sources()
    if model_name:
        return available.get(model_name, model_name)
    if "local" in available:
        return available["local"]
    return DEFAULT_REMOTE_MODEL


def normalize_emotion_label(raw_label):
    label = raw_label.lower().strip()
    label = LOCAL_LABEL_MAP.get(label, label)
    return CANONICAL_EMOTION_MAP.get(label, label)


def load_emotion_pipeline(model_name=None):
    model_source = resolve_model_source(model_name=model_name)
    return pipeline(
        "text-classification",
        model=model_source,
        tokenizer=model_source,
        top_k=1,
        truncation=True
    )


def detect_emotion(user_input, classifier=None):
    classifier = classifier or load_emotion_pipeline()
    prediction = classifier(user_input)[0]
    raw_label = prediction["label"]
    emotion = normalize_emotion_label(raw_label)
    score = float(prediction["score"])
    return emotion, score, raw_label


def emotion_chatbot(user_input, classifier=None):
    if not user_input or not user_input.strip():
        return {
            "emotion": "neutral",
            "raw_label": "neutral",
            "score": 1.0,
            "response": "Please enter a message so I can respond.",
        }

    emotion, score, raw_label = detect_emotion(user_input, classifier=classifier)

    response = EMOTION_RESPONSES.get(
        emotion,
        "Thank you for sharing that. Please tell me more so I can understand better."
    )

    return {
        "emotion": emotion,
        "raw_label": raw_label,
        "score": score,
        "response": response,
    }


def run_chat():
    print("Emotion-aware chatbot")
    print("Type 'quit' to stop\n")

    try:
        classifier = load_emotion_pipeline()
    except Exception as e:
        print("Model loading failed")
        print(e)
        return

    while True:
        user_input = input("You: ")

        if user_input.strip().lower() in ["quit", "exit"]:
            print("Bot: Goodbye")
            break

        result = emotion_chatbot(user_input, classifier)

        print(
            f"Detected emotion: {result['emotion']} "
            f"(raw={result['raw_label']}, score={result['score']:.2f})"
        )
        print("Bot:", result["response"])


if __name__ == "__main__":
    run_chat()
