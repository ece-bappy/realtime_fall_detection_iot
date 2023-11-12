import requests

# Replace 'YOUR_BOT_TOKEN' with the actual token you received from BotFather
TOKEN = "6813924100:AAFO02QvPduuBTYKfOFNYVqV89xAHmIlDsM"
GROUP_CHAT_ID = (
    -4060429072
)  # Replace with your group ID (use a negative sign before the group ID)


def send_message(message_text):
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    params = {
        "chat_id": GROUP_CHAT_ID,
        "text": message_text,
    }

    response = requests.post(url, params=params)
    print(response.json())


def main():
    print("Bot is running. Send a message, and it will be forwarded to the group.")
    while True:
        message_text = input("Enter your message: ")
        send_message(message_text)


if __name__ == "__main__":
    main()
