#!/usr/bin/env python3
import subprocess
import time
import argparse
import sys
import os

def get_free_vram():
    """Queries nvidia-smi for free GPU memory in MiB."""
    try:
        res = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"])
        # If there are multiple GPUs, this will return multiple lines. We'll take the first one (GPU 0).
        lines = res.decode().strip().split('\n')
        return int(lines[0])
    except Exception as e:
        print(f"Error checking VRAM: {e}")
        return 0

def main():
    parser = argparse.ArgumentParser(description="Wait for a specific amount of free GPU VRAM and then run a command.")
    parser.add_argument("--threshold", type=int, default=15360, help="Free VRAM threshold in MiB (default: 15360 for 15GB).")
    parser.add_argument("--interval", type=int, default=60, help="Polling interval in seconds (default: 60).")
    parser.add_argument("--command", type=str, required=True, help="The shell command to execute once memory is free.")
    parser.add_argument("--source-venv", action="store_true", help="Automatically source .venv/bin/activate before running the command.")

    args = parser.parse_args()

    # Prepare the final command
    final_command = args.command
    if args.source_venv:
        if os.path.exists(".venv/bin/activate"):
            final_command = f"source .venv/bin/activate && {final_command}"
        else:
            print("Warning: .venv/bin/activate not found. Skipping sourcing.")

    print(f"=== GPU Scheduler ===")
    print(f"Threshold: {args.threshold} MiB")
    print(f"Interval:  {args.interval} seconds")
    print(f"Command:   {final_command}")
    print(f"=====================")

    while True:
        free_vram = get_free_vram()
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        if free_vram >= args.threshold:
            print(f"[{timestamp}] SUCCESS: {free_vram} MiB free. Launching command...")
            # Use shell=True and executable='/bin/bash' to allow 'source' to work
            try:
                subprocess.Popen(final_command, shell=True, executable='/bin/bash', start_new_session=True)
                print(f"[{timestamp}] Process launched. Scheduler exiting.")
                break
            except Exception as e:
                print(f"[{timestamp}] Error launching command: {e}")
                sys.exit(1)
        else:
            print(f"[{timestamp}] Waiting: {free_vram} MiB free. (Need {args.threshold} MiB)")
            time.sleep(args.interval)

if __name__ == "__main__":
    main()
