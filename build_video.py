# build_video.py
import sys
import os
import json

import numpy as np
from PIL import Image
from moviepy.editor import (
    AudioFileClip,
    ImageClip,
    ColorClip,
    CompositeVideoClip,
    concatenate_videoclips,
    TextClip,
)

from media_utils import TOTAL_TARGET_SEC

VIDEO_SIZE = (1080, 1920)  # width, height


def main():
    if len(sys.argv) < 2:
        print("Usage: python build_video.py <folder_path>")
        sys.exit(1)

    folder = sys.argv[1]

    analysis_path = os.path.join(folder, "analysis.json")
    audio_path = os.path.join(folder, "combined_best15.mp3")

    if not os.path.isfile(analysis_path):
        print(f"Error: {analysis_path} not found. Run analyze_tracks.py first.")
        sys.exit(1)
    if not os.path.isfile(audio_path):
        print(f"Error: {audio_path} not found. Run build_audio.py first.")
        sys.exit(1)

    with open(analysis_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    tracks = data.get("tracks", [])
    if not tracks:
        print("No tracks in analysis.json.")
        sys.exit(0)

    num_tracks = len(tracks)
    per_segment_sec = TOTAL_TARGET_SEC / num_tracks

    print(f"Creating video with {num_tracks} segment(s), {per_segment_sec:.2f}s each.")

    segment_clips = []

    for idx, t in enumerate(tracks):
        artist = t["artist"]
        title = t["title"]
        album = t.get("album")
        year = t.get("year")
        cover_path = t["cover_path"]

        lines = [artist, title]
        if album:
            if year:
                lines.append(f"{album} ({year})")
            else:
                lines.append(album)
        label_text = "\n".join(lines)

        print(f"  Segment {idx+1}: {label_text.replace(chr(10), ' / ')}")

        # Load cover (already 800x800 from analyze step)
        img = Image.open(cover_path).convert("RGB")
        cover_arr = np.array(img)

        bg = ColorClip(size=VIDEO_SIZE, color=(5, 5, 5)).set_duration(per_segment_sec)

        cover_clip = (
            ImageClip(cover_arr)
            .set_duration(per_segment_sec)
            .set_position("center")
        )

        # Text overlay
        try:
            text_clip = (
                TextClip(
                    label_text,
                    fontsize=48,
                    color="white",
                    method="caption",
                    size=(VIDEO_SIZE[0] - 160, None),  # margin left/right
                )
                .set_duration(per_segment_sec)
                .set_position(("center", VIDEO_SIZE[1] - 260))
            )
            segment = CompositeVideoClip([bg, cover_clip, text_clip])
        except Exception as e:
            print(f"  [WARN] TextClip failed for segment {idx+1}: {e}")
            segment = CompositeVideoClip([bg, cover_clip])

        segment_clips.append(segment)

    final_video = concatenate_videoclips(segment_clips, method="compose")

    audio_clip = AudioFileClip(audio_path)
    final_video = final_video.set_audio(audio_clip)

    out_video_path = os.path.join(folder, "combined_best15_1080x1920.mp4")
    final_video.write_videofile(
        out_video_path,
        fps=30,
        codec="libx264",
        audio_codec="aac",
        threads=4,
        preset="medium",
    )

    print(f"\nVideo saved to: {out_video_path}")


if __name__ == "__main__":
    main()
