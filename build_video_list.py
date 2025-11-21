#!/usr/bin/env python3
# build_video_list.py
import sys
import os
import json
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import (
    AudioFileClip,
    ImageClip,
    VideoClip,
    VideoFileClip,
    CompositeVideoClip,
    concatenate_videoclips,
)

from media_utils import TOTAL_TARGET_SEC

VIDEO_SIZE = (1080, 1920)  # (width, height)

FADE_IN_DURATION = 0.5 

# ---- FONT / LAYOUT CONSTANTS (tweak these) ----
TITLE_FONT_SIZE = 64           # top caption (folder name)
LIST_FONT_SIZE = 48            # track list text
CTA_FONT_SIZE = 64            # track list text

# Safe zone / positioning
SAFE_LEFT = 128                # leftmost x for the whole list+bar block
LIST_VERTICAL_OFFSET = 0       # shift list block up/down after centering

ROW_COVER_SIZE = 200           # cover size in list rows (square)
ROW_SPACING = 48               # vertical spacing between rows

# Progress bar (vertical) constants
PROGRESS_BAR_WIDTH = 4        # px
PROGRESS_BAR_ALPHA_BG = 0.33   # background opacity (white)
PROGRESS_BAR_ALPHA_FILL = 1.0  # fill opacity (white)

# Distances between bar, cover, and text
BAR_TO_COVER_GAP = 16            # px
COVER_TO_TEXT_GAP = 24         # px

GAP = 96

# Font candidates (macOS system fonts)
FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/DIN Condensed Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
]


# ---------- Helpers ----------

def apply_background_fade(clip, fade_duration: float):
    """
    Returns a version of `clip` where, for t in [0, fade_duration],
    the frame is multiplied by (t / fade_duration), i.e. fades in from black.
    Overlays (titles, covers, text) stay unaffected in CompositeVideoClip.
    """
    def make_frame(get_frame, t):
        frame = get_frame(t)
        if t >= fade_duration:
            return frame
        alpha = max(0.0, min(1.0, t / fade_duration))
        # fade only background brightness
        return (frame.astype(np.float32) * alpha).astype(np.uint8)

    # fl takes a function (gf, t) -> frame
    return clip.fl(make_frame)

def find_background_video(folder: str) -> str:
    """
    Return path to the first .mp4 file in the folder, or raise if none.
    """
    for name in sorted(os.listdir(folder)):
        if(name.lower().startswith("reel")):
            continue
        if name.lower().endswith(".mp4"):
            return os.path.join(folder, name)
    raise FileNotFoundError("No .mp4 background video found in folder")


def load_font(font_size: int) -> ImageFont.FreeTypeFont:
    """
    Try known TTF fonts; if all fail, fall back to Pillow default.
    """
    for path in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, font_size)
        except Exception:
            continue
    print("[WARN] Could not load any TTF font, using default bitmap font.")
    return ImageFont.load_default()


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


def create_text_image(label_text: str,
                      max_width: int,
                      font_size: int,
                      align: str = "left") -> np.ndarray:
    """
    Render multiline text (white with subtle black shadow) into an RGBA image
    and return it as a numpy array.
    align: 'left' or 'center'
    """
    font = load_font(font_size)

    dummy = Image.new("RGBA", (max_width, 400), (0, 0, 0, 0))
    draw = ImageDraw.Draw(dummy)

    bbox = draw.multiline_textbbox((0, 0), label_text, font=font, spacing=12)
    x0, y0, x1, y1 = bbox
    text_w = x1 - x0
    text_h = y1 - y0

    pad_x = 20
    pad_y = LIST_FONT_SIZE // 2

    img_w = min(max_width, text_w + 2 * pad_x)
    img_h = text_h + 2 * pad_y

    img = Image.new("RGBA", (img_w, img_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    if align == "center":
        x_text = (img_w - text_w) // 2
    else:
        x_text = pad_x

    y_text = pad_y

    # shadow
    shadow_offset = (0, 2)
    draw.multiline_text(
        (x_text + shadow_offset[0], y_text + shadow_offset[1]),
        label_text,
        font=font,
        fill=(0, 0, 0, 200),
        spacing=LIST_FONT_SIZE // 2,
        align=align,
    )

    # main text
    draw.multiline_text(
        (x_text, y_text),
        label_text,
        font=font,
        fill=(255, 255, 255, 255),
        spacing=LIST_FONT_SIZE // 2,
        align=align,
    )

    return np.array(img)


def rgba_to_imageclip(rgba_arr: np.ndarray,
                      duration: float,
                      start: float = 0.0,
                      position=("center", "center")) -> ImageClip:
    """
    Convert an RGBA numpy array into a MoviePy ImageClip with alpha mask,
    so there is NO solid background (true transparency).
    """
    rgb = rgba_arr[..., :3]
    alpha = rgba_arr[..., 3] / 255.0

    img_clip = ImageClip(rgb)
    mask_clip = ImageClip(alpha, ismask=True)

    img_clip = img_clip.set_duration(duration).set_start(start).set_position(position)
    mask_clip = mask_clip.set_duration(duration).set_start(start).set_position(position)

    img_clip = img_clip.set_mask(mask_clip)
    return img_clip


def create_placeholder_cover(size: int) -> np.ndarray:
    """
    Create RGBA placeholder cover: transparent bg, white rectangle outline, '?' inside.
    """
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    margin = 6
    draw.rectangle(
        [margin, margin, size - margin, size - margin],
        outline=(255, 255, 255, 255),
        width=3,
    )

    font = load_font(int(size * 0.33))
    text = "?"
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    x = (size - tw) // 2
    y = (size - th) // 2

    # optional tiny shadow for '?'
    draw.text((x + 2, y + 2), text, font=font, fill=(0, 0, 0, 200))
    draw.text((x, y), text, font=font, fill=(255, 255, 255, 255))

    return np.array(img)


def make_vertical_progress_bar_clip(total_duration: float,
                                    num_tracks: int) -> VideoClip:
    """
    Create a vertical segmented progress bar for the entire video duration.

    - num_tracks segments stacked vertically
    - Each segment's height == ROW_COVER_SIZE
    - Vertical gaps == ROW_SPACING

    Returns an RGB VideoClip with a separate alpha mask (no 4-channel issue).
    """
    if num_tracks <= 0:
        num_tracks = 1

    width = PROGRESS_BAR_WIDTH
    seg_height = ROW_COVER_SIZE
    gap = ROW_SPACING

    total_bar_height = int(num_tracks * seg_height + (num_tracks - 1) * gap)
    seg_duration = total_duration / num_tracks

    def make_rgba(t):
        img = Image.new("RGBA", (width, total_bar_height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        # background segments (33% white)
        for i in range(num_tracks):
            y0 = int(i * (seg_height + gap))
            y1 = int(y0 + seg_height)
            draw.rectangle(
                [0, y0, width, y1],
                fill=(255, 255, 255, int(255 * PROGRESS_BAR_ALPHA_BG)),
            )

        # fill based on time
        for i in range(num_tracks):
            start_i = i * seg_duration
            end_i = (i + 1) * seg_duration

            if t >= end_i:
                fill_progress = 1.0
            elif t <= start_i:
                fill_progress = 0.0
            else:
                fill_progress = (t - start_i) / seg_duration

            if fill_progress <= 0:
                continue

            y0 = int(i * (seg_height + gap))
            y1 = int(y0 + seg_height * fill_progress)

            draw.rectangle(
                [0, y0, width, y1],
                fill=(255, 255, 255, int(255 * PROGRESS_BAR_ALPHA_FILL)),
            )

        return np.array(img)

    def make_rgb(t):
        rgba = make_rgba(t)
        return rgba[..., :3]

    def make_alpha(t):
        rgba = make_rgba(t)
        return rgba[..., 3] / 255.0

    color_clip = VideoClip(make_rgb, duration=total_duration)
    mask_clip = VideoClip(make_alpha, duration=total_duration, ismask=True)

    return color_clip.set_mask(mask_clip)


# ---------- Main ----------

def main():
    if len(sys.argv) < 2:
        print("Usage: python build_video_list.py <folder_path>")
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

    # Folder title (top text)
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
    total_duration = TOTAL_TARGET_SEC
    per_segment_sec = total_duration / num_tracks

    print(f"Creating list-style video with {num_tracks} tracks.")

    # Load & resize background safely
    base_bg = VideoFileClip(bg_video_path).without_audio()
    base_bg = resize_video_clip_clip_safe(base_bg, VIDEO_SIZE)

    # Loop background to cover total_duration
    loops_needed = max(1, math.ceil(total_duration / base_bg.duration))
    bg_loop_clips = [base_bg] * loops_needed
    bg_full = concatenate_videoclips(bg_loop_clips).subclip(0, total_duration)

    # Apply fade from black ONLY to background,
    # overlays (titles, covers, text, CTA) stay fully visible from t=0
    bg_full = apply_background_fade(bg_full, FADE_IN_DURATION)

    # Compute list total height and center it vertically
    list_total_height = num_tracks * ROW_COVER_SIZE + (num_tracks - 1) * ROW_SPACING
    list_top = int((VIDEO_SIZE[1] - list_total_height) / 2) + LIST_VERTICAL_OFFSET

    # Top title (folder name), centered, transparent background, above list
    title_arr = create_text_image(
        folder_title,
        max_width=VIDEO_SIZE[0] - 200,
        font_size=TITLE_FONT_SIZE,
        align="center",
    )
    title_h = title_arr.shape[0]
    title_y = list_top - title_h - GAP

    title_clip = rgba_to_imageclip(
        title_arr,
        duration=total_duration,
        start=0,
        position=("center", title_y),
    )

    # Horizontal layout: bar -> cover -> text
    bar_x = SAFE_LEFT
    cover_x = bar_x + PROGRESS_BAR_WIDTH + BAR_TO_COVER_GAP
    text_x = cover_x + ROW_COVER_SIZE + COVER_TO_TEXT_GAP
    text_max_width = VIDEO_SIZE[0] - text_x - SAFE_LEFT

    # Vertical progress bar, left of the list, aligned with list rows
    progress_bar_clip = make_vertical_progress_bar_clip(
        total_duration=total_duration,
        num_tracks=num_tracks,
    ).set_position((bar_x, list_top))

    # Row clips (covers + text + placeholders) with delayed appearance
    row_clips = []
    placeholder_arr = create_placeholder_cover(ROW_COVER_SIZE)

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

        print(f"  Row {idx+1}: {label_text.replace(chr(10), ' / ')}")

        # Real cover for row (small)
        cover_pil = Image.open(cover_path).convert("RGB")
        cover_pil_resized = cover_pil.resize(
            (ROW_COVER_SIZE, ROW_COVER_SIZE),
            resample=Image.Resampling.LANCZOS,
        )
        cover_arr = np.array(cover_pil_resized)

        # Text for row (left, justified)
        text_arr = create_text_image(
            label_text,
            max_width=text_max_width,
            font_size=LIST_FONT_SIZE,
            align="left",
        )

        row_y = list_top + idx * (ROW_COVER_SIZE + ROW_SPACING)

        # Reveal time: add a small global delay so at t=0 we only see placeholders
        reveal_time = idx * per_segment_sec
        # Clamp just in case
        if reveal_time > total_duration:
            reveal_time = total_duration

        visible_duration = max(0.0, total_duration - reveal_time)

        # Placeholder cover from t=0 until reveal_time (for EVERY track)
        if reveal_time > 0:
            placeholder_clip = rgba_to_imageclip(
                placeholder_arr,
                duration=reveal_time,
                start=0.0,
                position=(cover_x, row_y),
            )
            row_clips.append(placeholder_clip)

        # Real cover from reveal_time to end
        if visible_duration > 0:
            cover_clip = (
                ImageClip(cover_arr)
                .set_duration(visible_duration)
                .set_start(reveal_time)
                .set_position((cover_x, row_y))
            )

            # Text clip (transparent, appears at reveal_time)

            # --- vertical centering of text next to cover ---
            text_h = text_arr.shape[0]
            # cover height = ROW_COVER_SIZE, cover Y = row_y
            text_y = row_y + (ROW_COVER_SIZE // 2) - (text_h // 2) - LIST_FONT_SIZE // 4

            text_clip = rgba_to_imageclip(
                text_arr,
                duration=visible_duration,
                start=reveal_time,
                position=(text_x, text_y),
            )

            row_clips.extend([cover_clip, text_clip])
    
     # --- Bottom caption: "Next dose tomorrow" ---
    bottom_label = "MORE TRACKS TOMORROW"

    bottom_arr = create_text_image(
        bottom_label,
        max_width=VIDEO_SIZE[0] - 200,
        font_size=CTA_FONT_SIZE,
        align="center",
    )

    bottom_h = bottom_arr.shape[0]
    bottom_y = list_top + list_total_height + GAP

    bottom_clip = rgba_to_imageclip(
        bottom_arr,
        duration=total_duration,
        start=0,
        position=("center", bottom_y),
    )

    # Compose everything for the full duration
    final_video = CompositeVideoClip(
        [bg_full, title_clip, progress_bar_clip, *row_clips, bottom_clip],
        size=VIDEO_SIZE,
    ).set_duration(total_duration)

    # Add audio
    audio_clip = AudioFileClip(audio_path)
    final_video = final_video.set_audio(audio_clip)

    out_video_path = os.path.join(folder, "reel.mp4")
    final_video.write_videofile(
        out_video_path,
        fps=30,
        codec="libx264",
        audio_codec="aac",
        threads=4,
        preset="medium",
    )

    print(f"\nList-style video saved to: {out_video_path}")


if __name__ == "__main__":
    main()
