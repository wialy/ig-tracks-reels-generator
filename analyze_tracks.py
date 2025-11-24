# analyze_tracks.py
import sys
import os
import json
from typing import List, Dict

from pydub import AudioSegment
from PIL import Image

from media_utils import (
    SUPPORTED_EXT,
    TOTAL_TARGET_SEC,
    extract_metadata_and_cover,
    find_drop_centered_segment,
    make_square_cover_array,
    placeholder_cover_array,
)


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_tracks.py <folder_path>")
        sys.exit(1)

    folder = sys.argv[1]

    if not os.path.isdir(folder):
        print(f"Error: {folder} is not a directory")
        sys.exit(1)

    files = [
        os.path.join(folder, f)
        for f in sorted(os.listdir(folder))
        if f.lower().endswith(SUPPORTED_EXT)
    ]

    if not files:
        print("No supported audio files in folder.")
        sys.exit(0)

    print(f"Found {len(files)} candidate file(s):")
    for f in files:
        print("  -", os.path.basename(f))

    # Determine which tracks are actually loadable by pydub
    valid_files = []
    for filepath in files:
        basename = os.path.basename(filepath)
        try:
            _ = AudioSegment.from_file(filepath)
            valid_files.append(filepath)
        except Exception as e:
            print(f"[WARN] Skipping {basename}, pydub cannot open it: {e}")

    if not valid_files:
        print("No valid audio files after filtering. Exiting.")
        sys.exit(0)

    num_tracks = len(valid_files)
    per_segment_sec = TOTAL_TARGET_SEC / num_tracks

    print(f"\nValid tracks: {num_tracks}")
    print(f"Per-segment duration: {per_segment_sec:.2f} seconds")

    result: List[Dict] = []

    for filepath in valid_files:
        basename = os.path.basename(filepath)
        print(f"\nAnalyzing: {basename}")

        # 1) Metadata & cover
        artist, title, album, year, cover_img = extract_metadata_and_cover(filepath)

        if cover_img is None:
            cover_arr = placeholder_cover_array(size=800)
        else:
            cover_arr = make_square_cover_array(cover_img, size=800)

        # Save cover as a .jpg next to the track
        cover_name = os.path.splitext(basename)[0] + "_cover.jpg"
        cover_path = os.path.join(folder, cover_name)
        Image.fromarray(cover_arr).save(cover_path, format="JPEG")

        print(f"  Artist: {artist}")
        print(f"  Title:  {title}")
        if album:
            if year:
                print(f"  Album:  {album} ({year})")
            else:
                print(f"  Album:  {album}")
        print(f"  Cover saved to: {cover_path}")

        # 2) Best segment start
        start_sec = find_drop_centered_segment(
            filepath,
            target_sec=per_segment_sec,
            ignore_sec=10.0,
        )
        print(f"  Segment (drop-centered) start: {start_sec:.2f}s")

        track_entry = {
            "file_path": filepath,
            "artist": artist,
            "title": title,
            "album": album,
            "year": year,
            "cover_path": cover_path,
            "segment_start_sec": start_sec,
            "segment_duration_sec": per_segment_sec,
        }
        result.append(track_entry)

    # Save analysis.json
    out_json = os.path.join(folder, "analysis.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "total_target_sec": TOTAL_TARGET_SEC,
                "tracks": result,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"\nAnalysis saved to: {out_json}")


if __name__ == "__main__":
    main()
