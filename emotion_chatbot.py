from transformers import pipeline

emotion_model = pipeline("text-classification",
                         model="j-hartmann/emotion-english-distilroberta-base")

def emotion_chatbot(user_input):
    emotion = emotion_model(user_input)[0]['label']

    if emotion == "sadness":
        return "I'm sorry you're feeling sad."
    elif emotion == "joy":
        return "That's great to hear!"
    elif emotion == "anger":
        return "I understand you're upset."
    else:
        return "Tell me more."

print(emotion_chatbot("I feel stressed"))