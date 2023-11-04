import requests

# Replace 'YOUR_BOT_TOKEN' with the API token you received from BotFather
BOT_TOKEN = '6817863091:AAFne4OsI1jIVgL1ujL8Eel0fkGhaSxRz8U'

# Define the base URL for the Telegram Bot API
BASE_URL = f'https://api.telegram.org/bot{BOT_TOKEN}/'

# Function to send a message to the bot
def send_message(chat_id, text):
    url = BASE_URL + 'sendMessage'
    data = {
        'chat_id': chat_id,
        'text': text
    }
    response = requests.post(url, data=data)
    return response.json()

# Get the chat ID for your conversation with the bot
# You can send any message to your bot, and then use the message['chat']['id'] from the response to get your chat ID
chat_id =6050100984

while True:
    # Read input from the keyboard
    user_input = input("Enter a message for your bot (or type 'exit' to quit): ")

    if user_input.lower() == 'exit':
        break

    # Send the input as a message to the bot
    send_message(chat_id, user_input)
    print(f"Message sent: {user_input}")

print("Bot conversation ended.")
