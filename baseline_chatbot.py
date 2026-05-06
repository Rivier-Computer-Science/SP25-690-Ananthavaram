import re

QUESTION_WORDS = {"what", "why", "how", "when", "where", "who", "can", "do", "is", "are", "should", "could", "would"}


def normalize_text(text):
    return re.sub(r"\s+", " ", text.strip())


def baseline_chatbot(user_input):
    message = normalize_text(user_input)

    if not message:
        return "Please type a message so I can respond."

    lowered = message.lower()
    words = lowered.split()
    first_word = words[0] if words else ""

    if any(greet in lowered for greet in ["hello", "hi", "hey"]):
        return "Hello. What would you like to talk about today?"

    if first_word in QUESTION_WORDS or message.endswith("?"):
        return "That is a thoughtful question. Please explain a bit more so I can give a better response."

    if any(k in lowered for k in ["school", "project", "assignment", "exam", "study"]):
        return "That sounds important. Tell me what part you need help with."

    if any(k in lowered for k in ["problem", "issue", "error", "stuck"]):
        return "I understand there is some difficulty. Can you describe the problem in more detail?"

    if any(k in lowered for k in ["thank", "thanks"]):
        return "You're welcome. Let me know if you need anything else."

    if any(k in lowered for k in ["happy", "good", "great", "nice"]):
        return "That sounds good. Tell me more about it."

    if any(k in lowered for k in ["sad", "bad", "upset", "tired"]):
        return "I understand. If you want, you can share more about what happened."

    return "I understand. Tell me more about that."


def run_chat():
    print("Baseline chatbot")
    print("Type 'quit' to stop\n")

    while True:
        user_input = input("You: ")

        if user_input.strip().lower() in ["quit", "exit"]:
            print("Bot: Goodbye")
            break

        print("Bot:", baseline_chatbot(user_input))


if __name__ == "__main__":
    run_chat()
