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

    # Updated SEO-friendly static description (with max 3 hashtags)
    tail = dedent(
        """\
        Welcome to Daily Drops — a fast, punchy reel series where each day I highlight one powerful drop from a fresh electronic track. Whether it’s house, techno, deep house, organic house, melodic techno, or anything underground that hits with energy, you get the best moment of the song in just a few seconds.

        If you're searching for DJ music, DJ tracks, DJ mixes, DJ sets, or that perfect beat drop for inspiration, this series keeps your feed full of new electronic music and undiscovered gems. Great for DJs, producers, and anyone exploring music discovery, fresh tracks, or club-ready moments that stand out.

        One reel. One track. One drop worth remembering.

        #dailydrops #dj #electronicmusic
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
