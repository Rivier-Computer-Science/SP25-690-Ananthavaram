import re


QUESTION_WORDS = {"what", "why", "how", "when", "where", "who", "can", "do", "is"}


def normalize_text(text):
    return re.sub(r"\s+", " ", text.strip())


def baseline_chatbot(user_input):
    """
    A simple baseline chatbot that responds without using emotion labels.
    This gives us a comparison point for the emotion-aware chatbot.
    """
    message = normalize_text(user_input)
    if not message:
        return "Please type a message so I can respond."

    lowered = message.lower()
    first_word = lowered.split()[0]

    if "hello" in lowered or "hi" in lowered:
        return "Hello. What would you like to talk about today?"
    if first_word in QUESTION_WORDS or message.endswith("?"):
        return "That is a thoughtful question. Tell me a little more so I can respond better."
    if "school" in lowered or "project" in lowered or "assignment" in lowered:
        return "That sounds important. Tell me what part you want help with."
    if "thank" in lowered:
        return "You're welcome. Let me know if you want to keep going."

    return "I understand. Tell me more about that."


def run_chat():
    print("Baseline chatbot")
    print("Type 'quit' to stop.\n")

    while True:
        user_input = input("You: ")
        if user_input.strip().lower() in {"quit", "exit"}:
            print("Bot: Goodbye.")
            break
        print("Bot:", baseline_chatbot(user_input))


if __name__ == "__main__":
    run_chat()
