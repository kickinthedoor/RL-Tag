import os
from datetime import datetime

def log(message, log_file=None, timestamp=True):
    """
    Logs a message to stdout and optionally to a log file.

    Args:
        message (str): The message to log.
        log_file (str or None): If provided, writes the message to this file.
        timestamp (bool): Whether to prepend a timestamp to the message.
    """
    if timestamp:
        time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        message = f"[{time_str}] {message}"

    print(message)

    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(message + "\n")
