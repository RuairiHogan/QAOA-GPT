import subprocess
import time
import sys

SCRIPT = "ADAPT_QAOA.py"

print("Starting continuous QAOA-GPT data generation...")
print("Press Ctrl+C to stop.\n")

run_count = 0

while True:
    run_count += 1
    print(f"=== Run {run_count} starting ===")

    try:
        subprocess.run(
            [sys.executable, SCRIPT],
            check=True
        )
        print(f"=== Run {run_count} finished normally ===\n")

    except subprocess.CalledProcessError as e:
        print(f"!!! Run {run_count} crashed (code {e.returncode})")
        print("Restarting in 5 seconds...\n")
        time.sleep(5)

    except KeyboardInterrupt:
        print("\nStopped by user.")
        break

    time.sleep(2)
