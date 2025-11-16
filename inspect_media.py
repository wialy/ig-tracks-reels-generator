#!/usr/bin/env python3
import sys
import os
from io import BytesIO

from mutagen.easyid3 import EasyID3
from mutagen.id3 import ID3, APIC, ID3NoHeaderError
from PIL import Image


def print_easy_tags(path: str):
    print(f"=== File: {path} ===")
    if not os.path.isfile(path):
        print("ERROR: Path is not a regular file.")
        return

    print("\n[EasyID3 tags]")
    try:
        tags = EasyID3(path)
    except Exception as e:
        print(f"  Could not read EasyID3 tags: {e}")
        return

    if not tags:
        print("  No EasyID3 tags found.")
        return

    for key in sorted(tags.keys()):
        print(f"  {key}: {tags[key]}")


def print_raw_id3_keys(path: str):
    print("\n[Raw ID3 frame keys]")
    try:
        id3 = ID3(path)
    except ID3NoHeaderError:
        print("  No ID3 header found.")
        return
    except Exception as e:
        print(f"  Could not read ID3 tags: {e}")
        return

    keys = list(id3.keys())
    if not keys:
        print("  No ID3 frames.")
        return

    for k in keys:
        print(f"  {k}")


def extract_cover(path: str):
    print("\n[Cover art]")
    try:
        id3 = ID3(path)
    except ID3NoHeaderError:
        print("  No ID3 header found – no cover art.")
        return
    except Exception as e:
        print(f"  Could not read ID3 tags: {e}")
        return

    apic_frames = [f for f in id3.values() if isinstance(f, APIC)]

    if not apic_frames:
        print("  No APIC (cover art) frames found.")
        return

    apic = apic_frames[0]
    mime = apic.mime
    print(f"  Found APIC frame. MIME type: {mime}")

    ext = ".jpg"
    if mime == "image/png":
        ext = ".png"

    base, _ = os.path.splitext(path)
    out_path = base + "_cover" + ext

    try:
        img_data = apic.data
        img = Image.open(BytesIO(img_data)).convert("RGB")
        img.save(out_path)
        print(f"  Cover extracted and saved to: {out_path}")
    except Exception as e:
        print(f"  Failed to decode/save cover image: {e}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python inspect_id3.py /path/to/file.mp3")
        sys.exit(1)

    path = sys.argv[1]

    print_easy_tags(path)
    print_raw_id3_keys(path)
    extract_cover(path)


if __name__ == "__main__":
    main()
