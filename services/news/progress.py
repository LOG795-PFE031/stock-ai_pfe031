import sys
import threading
import time

class Spinner:
    def __init__(self, message="Loading..."):
        self.message = message
        self._running = False
        self._thread = None

    def start(self):
        self._running = True
        print(f"🔄 {self.message}", flush=True)
        self._thread = threading.Thread(target=self._spin)
        self._thread.start()

    def _spin(self):
        while self._running:
            time.sleep(0.1)

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join()
        print("✔️ Done", flush=True)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

def create_spinner(message):
    return Spinner(message)

def print_status(status, message, level="info", clear_previous=False):
    icons = {"success": "✔️", "error": "❌", "warning": "⚠️", "info": "ℹ️"}
    icon = icons.get(level, "ℹ️")
    print(f"{icon} {status}: {message}", flush=True)

def print_error(e):
    print(f"❌ Error: {e}", file=sys.stderr, flush=True) 
