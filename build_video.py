# build_video.py
import sys
import os
import json
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import (
    AudioFileClip,
    ImageClip,
    VideoFileClip,
    CompositeVideoClip,
    concatenate_videoclips,
)

from media_utils import TOTAL_TARGET_SEC

VIDEO_SIZE = (1080, 1920)  # (width, height)

# ---- FONT / LAYOUT CONSTANTS (tweak these) ----
CAPTION_FONT_SIZE = 64      # bottom captions (artist / title / album)
TITLE_FONT_SIZE = 64        # top caption (folder name)
CAPTION_OFFSET = 48          # distance between cover and each caption (px)


def find_background_video(folder: str) -> str:
    """
    Return path to the first .mp4 file in the folder, or raise if none.
    """
    for name in sorted(os.listdir(folder)):
        if name.lower().endswith(".mp4"):
            return os.path.join(folder, name)
    raise FileNotFoundError("No .mp4 background video found in folder")


def resize_video_clip_clip_safe(clip, target_size):
    """
    Fully PIL-safe resizing: manually resizes each frame using
    Pillow's modern LANCZOS (no MoviePy ANTIALIAS used).
    """
    tw, th = target_size

    def resize_frame(frame):
        img = Image.fromarray(frame)
        img = img.resize((tw, th), resample=Image.Resampling.LANCZOS)
        return np.array(img)

    return clip.fl_image(resize_frame)


def create_text_image(label_text: str, max_width: int, font_size: int) -> np.ndarray:
    """
    Render multiline text (white with subtle black shadow) into an RGBA image
    using a REAL TTF font on macOS.
    """

    # Try a guaranteed macOS system font
    mac_font_path = "/System/Library/Fonts/Supplemental/DIN Condensed Bold.ttf"

    try:
        font = ImageFont.truetype(mac_font_path, font_size)
    except Exception as e:
        print("[WARN] Failed to load system TTF font, falling back to default:", e)
        font = ImageFont.load_default()

    dummy = Image.new("RGBA", (max_width, 400), (0, 0, 0, 0))
    draw = ImageDraw.Draw(dummy)

    # measure multiline text
    bbox = draw.multiline_textbbox((0, 0), label_text, font=font, spacing=8)
    x0, y0, x1, y1 = bbox
    text_w = x1 - x0
    text_h = y1 - y0

    pad_x = 20
    pad_y = 20

    img_w = min(max_width, text_w + 2 * pad_x)
    img_h = text_h + 2 * pad_y

    img = Image.new("RGBA", (img_w, img_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    x_text = (img_w - text_w) // 2
    y_text = pad_y

    # shadow
    shadow_offset = (3, 3)
    draw.multiline_text(
        (x_text + shadow_offset[0], y_text + shadow_offset[1]),
        label_text,
        font=font,
        fill=(0, 0, 0, 200),
        spacing=10,
        align="center",
    )

    # main text
    draw.multiline_text(
        (x_text, y_text),
        label_text,
        font=font,
        fill=(255, 255, 255, 255),
        spacing=10,
        align="center",
    )

    return np.array(img)



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

    # Folder title for top caption (e.g. "./tracks/spoti/" -> "spoti")
    folder_title = os.path.basename(os.path.normpath(folder)) or folder

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

    # Load & resize background safely
    base_bg = VideoFileClip(bg_video_path).without_audio()
    base_bg = resize_video_clip_clip_safe(base_bg, VIDEO_SIZE)

    # Loop background to cover TOTAL_TARGET_SEC (typically 15s)
    loops_needed = max(1, math.ceil(TOTAL_TARGET_SEC / base_bg.duration))
    bg_loop_clips = [base_bg] * loops_needed
    bg_loop = concatenate_videoclips(bg_loop_clips).subclip(0, TOTAL_TARGET_SEC)

    segment_clips = []

    cover_size = VIDEO_SIZE[0] // 2

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

        # Load cover and resize to 1/3 width (Pillow)
        cover_pil = Image.open(cover_path).convert("RGB")
        cover_pil_resized = cover_pil.resize(
            (cover_size, cover_size),
            resample=Image.Resampling.LANCZOS,
        )
        cover_arr = np.array(cover_pil_resized)

        # Background slice for this segment
        seg_start = idx * per_segment_sec
        seg_end = (idx + 1) * per_segment_sec
        bg_seg = bg_loop.subclip(seg_start, seg_end)

        # Cover in the center
        cover_clip = (
            ImageClip(cover_arr)
            .set_duration(per_segment_sec)
            .set_position("center")
        )

        # Render bottom caption (track info)
        caption_arr = create_text_image(
            label_text,
            max_width=VIDEO_SIZE[0] - 160,
            font_size=CAPTION_FONT_SIZE,
        )
        caption_h = caption_arr.shape[0]

        # Render top title (folder name)
        title_arr = create_text_image(
            folder_title,
            max_width=VIDEO_SIZE[0] - 160,
            font_size=TITLE_FONT_SIZE,
        )
        title_h = title_arr.shape[0]

        # ---- Symmetric layout around cover ----

        # Vertical position of cover (centered)
        cover_top = (VIDEO_SIZE[1] - cover_size) // 2
        cover_bottom = cover_top + cover_size

        # Top caption: same distance above cover as bottom caption is below
        top_caption_y = cover_top - CAPTION_OFFSET - title_h
        bottom_caption_y = cover_bottom + CAPTION_OFFSET

        # Build text clips
        caption_clip = (
            ImageClip(caption_arr)
            .set_duration(per_segment_sec)
            .set_position(("center", bottom_caption_y))
        )

        title_clip = (
            ImageClip(title_arr)
            .set_duration(per_segment_sec)
            .set_position(("center", top_caption_y))
        )

        segment = CompositeVideoClip([bg_seg, cover_clip, caption_clip, title_clip])
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
