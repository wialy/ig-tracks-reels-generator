#!/usr/bin/env python3
# build_description.py

import sys
import os
import json
from textwrap import dedent


def main():
    if len(sys.argv) < 2:
        print("Usage: python build_description.py <folder_path>")
        sys.exit(1)

    folder = sys.argv[1]
    analysis_path = os.path.join(folder, "analysis.json")

    if not os.path.isfile(analysis_path):
        print(f"Error: {analysis_path} not found. Run analyze_tracks.py first.")
        sys.exit(1)

    with open(analysis_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    tracks = data.get("tracks", [])
    if not tracks:
        print("No tracks found in analysis.json.")
        sys.exit(0)

    # Intro line
    lines = ["I dig through tracks so you don't have to.", ""]

    # Bullet list of tracks: "- Artist - Title"
    for t in tracks:
        artist = t.get("artist", "").strip() or "Unknown Artist"
        title = t.get("title", "").strip() or "Untitled"
        lines.append(f"- {artist} - {title}")

    lines.append("")  # blank line

    # Static description
    tail = dedent(
        """\
        Daily Drops is my daily reel series for anyone into electronic music. Every day, I share quick clips from three fresh tracks—house, techno, ambient, breaks, anything that stands out. It’s an easy way to discover new electronic music without spending hours digging for it.

        If you’re a DJ, producer, or just someone who likes staying on top of new releases, Daily Drops keeps your feed filled with new ideas and new artists. One reel a day, three new tracks to check out.
        """
    )

    lines.append(tail)

    description = "\n".join(lines)

    # Print to stdout
    print(description)

    # Save to file in the same folder
    out_path = os.path.join(folder, "description.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(description)

    print(f"\nDescription saved to: {out_path}")


if __name__ == "__main__":
    main()
