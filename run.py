#!/usr/bin/env python3
import sys
import os
import subprocess


def run_step(script_name: str, folder: str):
    """
    Run a single step script like:
      python script_name folder
    using the same Python executable as this script.
    """
    cmd = [sys.executable, script_name, folder]
    print(f"\n=== Running: {' '.join(cmd)} ===")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\n[ERROR] Step {script_name} failed with code {result.returncode}. Aborting pipeline.")
        sys.exit(result.returncode)


def main():
    if len(sys.argv) < 2:
        print("Usage: python run_all.py <folder_path>")
        sys.exit(1)

    # Normalize folder path
    folder = os.path.abspath(sys.argv[1])

    if not os.path.isdir(folder):
        print(f"Error: {folder} is not a directory")
        sys.exit(1)

    print(f"Starting full pipeline for: {folder}")

    # Steps in order
    run_step("analyze_tracks.py", folder)
    run_step("build_audio.py", folder)
    run_step("build_video_list.py", folder)
    run_step("cleanup.py", folder)

    print("\n✅ Pipeline completed successfully!")


if __name__ == "__main__":
    main()
