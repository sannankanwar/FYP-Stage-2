#!/usr/bin/env python3
"""
Telegram Notification Script
----------------------------
Sends a message via Telegram Bot API using a bot token and chat ID from environment variables.
Supports attaching a log tail to the message.

Usage:
    python scripts/notify.py "Message Title" "Optional body text" --log-file /path/to/log.txt
"""

import os
import sys
import argparse
import urllib.request
import urllib.parse
import json

def send_telegram_message(token, chat_id, text):
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "Markdown"
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, 
        data=data, 
        headers={"Content-Type": "application/json"}
    )
    
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            return response.status == 200
    except Exception as e:
        print(f"Warning: Failed to send Telegram notification: {e}", file=sys.stderr)
        return False

def tail_file(filepath, n_lines=40):
    """Read the last n_lines of a file."""
    if not os.path.exists(filepath):
        return f"[Log file not found: {filepath}]"
    
    try:
        with open(filepath, 'r', errors='replace') as f:
            # Efficient implementation for large files would seek to end, but 
            # for typical training logs, reading lines is acceptable for now.
            lines = f.readlines()
            return "".join(lines[-n_lines:])
    except Exception as e:
        return f"[Error reading log file: {e}]"

def main():
    parser = argparse.ArgumentParser(description="Send Telegram notification.")
    parser.add_argument("title", help="Message title/header")
    parser.add_argument("body", nargs="?", default="", help="Message body")
    parser.add_argument("--log-file", help="Path to log file to attach tail of")
    
    args = parser.parse_args()

    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")

    if not token or not chat_id:
        print("Warning: TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set. Skipping notification.", file=sys.stdout)
        # We exit 0 so we don't crash the pipeline, just warn
        sys.exit(0)

    message = f"*{args.title}*\n\n{args.body}"
    
    if args.log_file:
        log_tail = tail_file(args.log_file)
        message += f"\n\n*Log Tail:*\n```\n{log_tail}\n```"

    success = send_telegram_message(token, chat_id, message)
    if not success:
        print("Failed to send notification.")

if __name__ == "__main__":
    main()
