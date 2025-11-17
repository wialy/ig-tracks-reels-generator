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
    VideoClip,
    ColorClip,
)

from media_utils import TOTAL_TARGET_SEC

VIDEO_SIZE = (1080, 1920)  # (width, height)

# ---- FONT / LAYOUT CONSTANTS (tweak these) ----
CAPTION_FONT_SIZE = 64       # bottom captions (artist / title / album)
TITLE_FONT_SIZE = 64         # top caption (folder name)
TRACK_INDEX_FONT_SIZE = 48   # "Track X / N"
CAPTION_OFFSET = 60           # distance between elements (px)

# Progress bar
PROGRESS_BAR_HEIGHT = 2
PROGRESS_BAR_WIDTH_FACTOR = 0.5  # 50% of video width
PROGRESS_BAR_GAP = 8             # gap between segments (px)


# Try a few macOS system fonts; fall back to default if needed
FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/DIN Condensed Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
]


def find_background_video(folder: str) -> str:
    """
    Return path to the first .mp4 file in the folder, or raise if none.
    """
    for name in sorted(os.listdir(folder)):
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


def create_text_image(label_text: str, max_width: int, font_size: int,
                      line_spacing: int = 16) -> np.ndarray:
    """
    Render multiline text (white with subtle black shadow) into an RGBA image
    and return it as a numpy array.

    Added: adjustable line spacing (default: 16px).
    """
    font = load_font(font_size)

    dummy = Image.new("RGBA", (max_width, 400), (0, 0, 0, 0))
    draw = ImageDraw.Draw(dummy)

    # measure multiline text with spacing
    bbox = draw.multiline_textbbox(
        (0, 0),
        label_text,
        font=font,
        spacing=line_spacing,
    )
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
        spacing=line_spacing,
        align="center",
    )

    # main white text
    draw.multiline_text(
        (x_text, y_text),
        label_text,
        font=font,
        fill=(255, 255, 255, 255),
        spacing=line_spacing,
        align="center",
    )

    return np.array(img)



def make_progress_bar_clip(segment_index: int,
                           total_segments: int,
                           duration: float) -> VideoClip:
    """
    Create a VideoClip that draws a segmented progress bar over time.

    - total_segments: total number of tracks (N)
    - segment_index: current track index (0-based)
    - duration: duration of this segment in seconds

    Uses RGBA + a separate mask, so the bar background can be semi-transparent.
    """
    bar_width = int(VIDEO_SIZE[0] * PROGRESS_BAR_WIDTH_FACTOR)
    bar_height = PROGRESS_BAR_HEIGHT
    gap = PROGRESS_BAR_GAP

    if total_segments <= 0:
        total_segments = 1

    # width of one segment
    seg_width = (bar_width - gap * (total_segments - 1)) / total_segments

    def make_rgba_frame(t):
        # t in [0, duration]
        progress = max(0.0, min(1.0, t / duration))

        # RGBA for drawing (transparent background)
        img = Image.new("RGBA", (bar_width, bar_height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        # background segments: white with low alpha (e.g. ~33% visible)
        bg_alpha = int(255 * 0.33)
        for i in range(total_segments):
            x0 = int(i * (seg_width + gap))
            x1 = int(x0 + seg_width)
            draw.rectangle(
                [x0, 0, x1, bar_height],
                fill=(255, 255, 255, bg_alpha),
            )

        # fill segments: solid white (alpha 255)
        for i in range(total_segments):
            if i < segment_index:
                fill_progress = 1.0
            elif i == segment_index:
                fill_progress = progress
            else:
                fill_progress = 0.0

            if fill_progress <= 0:
                continue

            x0 = int(i * (seg_width + gap))
            x1 = int(x0 + seg_width * fill_progress)
            draw.rectangle(
                [x0, 0, x1, bar_height],
                fill=(255, 255, 255, 255),
            )

        return np.array(img)

    # Color clip (RGB only)
    def make_color_frame(t):
        rgba = make_rgba_frame(t)
        return rgba[..., :3]  # drop alpha

    # Mask clip (single channel float 0..1)
    def make_mask_frame(t):
        rgba = make_rgba_frame(t)
        alpha = rgba[..., 3]  # shape (H, W)
        return (alpha.astype("float32") / 255.0)

    color_clip = VideoClip(make_color_frame, duration=duration)
    mask_clip = VideoClip(make_mask_frame, ismask=True, duration=duration)

    return color_clip.set_mask(mask_clip)



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

    # Loop background to cover TOTAL_TARGET_SEC
    loops_needed = max(1, math.ceil(TOTAL_TARGET_SEC / base_bg.duration))
    bg_loop_clips = [base_bg] * loops_needed
    bg_loop = concatenate_videoclips(bg_loop_clips).subclip(0, TOTAL_TARGET_SEC)

    segment_clips = []

    # Cover size: 1/3 of video width (square)
    cover_size = VIDEO_SIZE[0] // 2

    for idx, t in enumerate(tracks):
        artist = t["artist"]
        title = t["title"]
        album = t.get("album")
        year = t.get("year")
        cover_path = t["cover_path"]

        # Build multiline caption text below cover
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

        # Bottom caption (track info)
        caption_arr = create_text_image(
            label_text,
            max_width=VIDEO_SIZE[0] - 160,
            font_size=CAPTION_FONT_SIZE,
        )
        caption_h = caption_arr.shape[0]

        # Top title (folder name, e.g. "spoti")
        title_arr = create_text_image(
            folder_title,
            max_width=VIDEO_SIZE[0] - 160,
            font_size=TITLE_FONT_SIZE,
        )
        title_h = title_arr.shape[0]

        # Track index caption: "Track X / N"
        track_index_text = f"Track {idx + 1} / {num_tracks}"
        track_index_arr = create_text_image(
            track_index_text,
            max_width=VIDEO_SIZE[0] - 160,
            font_size=TRACK_INDEX_FONT_SIZE,
        )
        track_index_h = track_index_arr.shape[0]

        # ---- Layout ----

        # Vertical position of cover (centered)
        cover_top = (VIDEO_SIZE[1] - cover_size) // 2
        cover_bottom = cover_top + cover_size

        # Track index: right above the cover
        track_index_y = cover_top - CAPTION_OFFSET - track_index_h

        # Folder title: above track index
        title_y = track_index_y - title_h

        bar_y = cover_bottom + PROGRESS_BAR_GAP

        # Bottom caption: below progress bar
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
            .set_position(("center", title_y))
        )

        track_index_clip = (
            ImageClip(track_index_arr)
            .set_duration(per_segment_sec)
            .set_position(("center", track_index_y))
        )

        # Progress bar clip
        progress_clip = make_progress_bar_clip(
            segment_index=idx,
            total_segments=num_tracks,
            duration=per_segment_sec,
        ).set_position(
            ("center", bar_y)
        )

        segment = CompositeVideoClip(
            [bg_seg, cover_clip, progress_clip, caption_clip, track_index_clip, title_clip]
        )
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
