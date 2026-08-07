import time
import json
from datetime import datetime

try:
    import psutil
except ImportError:
    print("Please install psutil first:")
    print("pip install psutil")
    exit()

filename = "telemetry_log.json"
interval = 600  # 10 minutes (change to 2 or 5 for quick testing)

print(f"Logging to {filename} every {interval} seconds. Press Ctrl+C to stop.")

try:
    while True:
        entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "cpu_usage": psutil.cpu_percent(interval=1),
            "memory_usage": psutil.virtual_memory().percent
        }

        # Read existing data or start fresh
        try:
            with open(filename, "r") as file:
                data = json.load(file)
                if not isinstance(data, list):
                    data = [data]
        except (FileNotFoundError, json.JSONDecodeError):
            data = []

        data.append(entry)

        # Write updated data back
        with open(filename, "w") as file:
            json.dump(data, file, indent=4)

        print(f"[{entry['timestamp']}] CPU: {entry['cpu_usage']}% | RAM: {entry['memory_usage']}% - Logged successfully.")
        time.sleep(interval)
except KeyboardInterrupt:
    print("\nLogging stopped by user.")
