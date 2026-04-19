from pathlib import Path

from transformers import pipeline


DEFAULT_REMOTE_MODEL = "j-hartmann/emotion-english-distilroberta-base"
LOCAL_MODEL_PATH = Path("saved_emotion_model")
LOCAL_LABEL_MAP = {
    "label_0": "sadness",
    "label_1": "joy",
    "label_2": "love",
    "label_3": "anger",
    "label_4": "fear",
    "label_5": "surprise",
}

EMOTION_RESPONSES = {
    "sadness": "I'm sorry you're dealing with that. Do you want to share what has been hardest?",
    "joy": "That sounds really positive. What made you feel that way?",
    "anger": "I can tell this is frustrating. Do you want to talk through what happened?",
    "fear": "That sounds stressful. Let's take it one step at a time.",
    "love": "That sounds meaningful and important to you.",
    "surprise": "That sounds unexpected. What happened next?",
    "neutral": "I hear you. Tell me a little more.",
}


def load_emotion_pipeline():
    """
    Prefer a locally trained model if it exists. Otherwise fall back to a
    public emotion classification model for demonstration purposes.
    """
    model_source = str(LOCAL_MODEL_PATH) if LOCAL_MODEL_PATH.exists() else DEFAULT_REMOTE_MODEL
    return pipeline("text-classification", model=model_source, tokenizer=model_source)


def detect_emotion(user_input, classifier=None):
    classifier = classifier or load_emotion_pipeline()
    prediction = classifier(user_input)[0]
    raw_label = prediction["label"].lower()
    emotion = LOCAL_LABEL_MAP.get(raw_label, raw_label)
    return emotion, float(prediction["score"])


def emotion_chatbot(user_input, classifier=None):
    if not user_input.strip():
        return {
            "emotion": "neutral",
            "score": 1.0,
            "response": "Please type a message so I can respond.",
        }

    emotion, score = detect_emotion(user_input, classifier=classifier)
    response = EMOTION_RESPONSES.get(
        emotion, "Thanks for sharing that. Tell me a little more so I can understand better."
    )
    return {"emotion": emotion, "score": score, "response": response}


def run_chat():
    print("Emotion-aware chatbot")
    print("Type 'quit' to stop.\n")

    try:
        classifier = load_emotion_pipeline()
    except Exception as exc:
        print("Could not load the emotion model.")
        print(f"Error: {exc}")
        return

    while True:
        user_input = input("You: ")
        if user_input.strip().lower() in {"quit", "exit"}:
            print("Bot: Goodbye.")
            break

        result = emotion_chatbot(user_input, classifier=classifier)
        print(f"Detected emotion: {result['emotion']} ({result['score']:.2f})")
        print("Bot:", result["response"])


if __name__ == "__main__":
    run_chat()
