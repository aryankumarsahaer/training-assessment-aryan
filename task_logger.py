# task_logger.py #task 2

import json
import os
import time
from datetime import datetime


FILE_NAME = "telemetry.json"

# Ask user for interval in seconds
interval = int(input("Enter logging interval in seconds: "))


# Load existing data if file exists
if os.path.exists(FILE_NAME):
    try:
        with open(FILE_NAME, "r") as file:
            data = json.load(file)

        if data:
            counter = data[-1]["counter"]
        else:
            counter = 0

    except json.JSONDecodeError:
        data = []
        counter = 0

else:
    data = []
    counter = 0


print(f"\nLogger started - logging every {interval} seconds")
print("Press Ctrl+C to stop.\n")


try:
    while True:

        counter += 10

        now = datetime.now()

        entry = {
            "time": now.strftime("%H:%M:%S"),
            "date": now.strftime("%Y-%m-%d"),
            "counter": counter
        }

        data.append(entry)

        # Update telemetry.json
        with open(FILE_NAME, "w") as file:
            json.dump(data, file, indent=4)

        print("Logged:", entry)

        # Wait for given seconds
        time.sleep(interval)


except KeyboardInterrupt:
    print("\nLogger stopped.")