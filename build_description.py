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
    # (Usually this will be one track per reel, but we keep support for multiple.)
    for t in tracks:
        artist = t.get("artist", "").strip() or "Unknown Artist"
        title = t.get("title", "").strip() or "Untitled"
        lines.append(f"- {artist} - {title}")

    lines.append("")  # blank line

    # Updated SEO-friendly static description
    tail = dedent(
        """\
        Daily Drops is my daily reel series showcasing one standout electronic track per day, focused on the most impactful moment — the drop.
        I dig through new releases and hidden gems across house, techno, melodic, minimal, breaks, ambient and more to highlight tracks worth your attention.

        Whether you're a DJ looking for fresh tracks, a producer searching for inspiration, or just someone who loves discovering new electronic music, these short clips make it easy to stay updated without spending hours digging.

        Follow for daily underground finds, new releases and festival-ready drops — one reel a day, one new track to check out.

        #dailydrops 
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
