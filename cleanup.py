# cleanup.py
import sys
import os


def main():
    if len(sys.argv) < 2:
        print("Usage: python cleanup.py <folder_path>")
        sys.exit(1)

    folder = sys.argv[1]
    mp3_path = os.path.join(folder, "_audio.mp3")

    if os.path.isfile(mp3_path):
        os.remove(mp3_path)
        print(f"Removed: {mp3_path}")
    else:
        print(f"No file to remove: {mp3_path}")


if __name__ == "__main__":
    main()
