def baseline_chatbot(user_input):
    return "I understand. Tell me more."

# Test
if __name__ == "__main__":
    user = input("You: ")
    print("Bot:", baseline_chatbot(user))