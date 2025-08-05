import datetime

class Logs:
    def __init__(self, log_file="app.log"):
        self.log_file = log_file

    def write(self, message: str):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}\n"
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(log_message)
