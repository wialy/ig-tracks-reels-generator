# build_video.py
import sys
import os
import json
import math

import numpy as np
from PIL import Image
from moviepy.editor import (
    AudioFileClip,
    ImageClip,
    VideoFileClip,
    CompositeVideoClip,
    concatenate_videoclips,
    TextClip,
)

from media_utils import TOTAL_TARGET_SEC

VIDEO_SIZE = (1080, 1920)  # (width, height)

def resize_video_clip_clip_safe(clip, target_size):
    """
    Fully PIL-safe resizing: manually resizes each frame using
    Pillow's modern LANCZOS (no MoviePy ANTIALIAS used).
    """
    tw, th = target_size

    def resize_frame(frame):
        # frame = numpy array, convert → PIL, resize → array
        img = Image.fromarray(frame)
        img = img.resize((tw, th), resample=Image.Resampling.LANCZOS)
        return np.array(img)

    return clip.fl_image(resize_frame)


def find_background_video(folder: str) -> str:
    """
    Return path to the first .mp4 file in the folder, or raise if none.
    """
    for name in sorted(os.listdir(folder)):
        if name.lower().endswith(".mp4"):
            return os.path.join(folder, name)
    raise FileNotFoundError("No .mp4 background video found in folder")


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

    # background video
    try:
        bg_video_path = find_background_video(folder)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    print(f"Using background video: {os.path.basename(bg_video_path)}")

    with open(analysis_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    tracks = data.get("tracks", [])
    if not tracks:
        print("No tracks in analysis.json.")
        sys.exit(0)

    num_tracks = len(tracks)
    per_segment_sec = TOTAL_TARGET_SEC / num_tracks

    print(f"Creating video with {num_tracks} segments, {per_segment_sec:.2f}s each.")

    # Load base background clip (no audio)
    base_bg = VideoFileClip(bg_video_path).without_audio()

    # Resize to 1080x1920 (simple stretch; tweak if you prefer crop)
    base_bg = resize_video_clip_clip_safe(base_bg, VIDEO_SIZE)

    # Loop the background to cover total duration (>= 15s)
    loops_needed = max(1, math.ceil(TOTAL_TARGET_SEC / base_bg.duration))
    bg_loop_clips = [base_bg] * loops_needed
    bg_loop = concatenate_videoclips(bg_loop_clips).subclip(0, TOTAL_TARGET_SEC)

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

        # Background slice for this segment
        seg_start = idx * per_segment_sec
        seg_end = (idx + 1) * per_segment_sec
        bg_seg = bg_loop.subclip(seg_start, seg_end)

        cover_clip = (
            ImageClip(cover_arr)
            .set_duration(per_segment_sec)
            .set_position("center")
        )

        # Text overlay: white with subtle black shadow
        # Note: TextClip usually needs ImageMagick installed
        try:
            text_clip = (
                TextClip(
                    label_text,
                    fontsize=48,
                    color="white",
                    stroke_color="black",      # subtle black outline / shadow
                    stroke_width=2,
                    method="caption",
                    size=(VIDEO_SIZE[0] - 160, None),  # side margins
                )
                .set_duration(per_segment_sec)
                .set_position(("center", VIDEO_SIZE[1] - 260))
            )
            segment = CompositeVideoClip([bg_seg, cover_clip, text_clip])
        except Exception as e:
            print(f"  [WARN] TextClip failed for segment {idx+1}: {e}")
            segment = CompositeVideoClip([bg_seg, cover_clip])

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
