#!/bin/bash
# Minimal test for Telegram
SCRIPT_DIR=$(dirname "$0")

# Load secrets
if [ -f "$SCRIPT_DIR/../secrets.sh" ]; then
    source "$SCRIPT_DIR/../secrets.sh"
elif [ -f "secrets.sh" ]; then
    source "secrets.sh"
fi

python3 -c "
import os
import urllib.request
import urllib.parse
import json

token = os.environ.get('TELEGRAM_BOT_TOKEN')
chat_id = os.environ.get('TELEGRAM_CHAT_ID')

print(f'Token: {token[:5]}...')
print(f'Chat ID: {chat_id}')

url = f'https://api.telegram.org/bot{token}/sendMessage'
payload = {
    'chat_id': chat_id,
    'text': 'Test: Simple Plain Text',
    'parse_mode': 'Markdown'
}
data = json.dumps(payload).encode('utf-8')
req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})

try:
    with urllib.request.urlopen(req) as response:
        print(f'Response Code: {response.status}')
        print(response.read().decode('utf-8'))
except urllib.error.HTTPError as e:
    print(f'HTTP Error: {e.code}')
    print(e.read().decode('utf-8'))
except Exception as e:
    print(e)
"
