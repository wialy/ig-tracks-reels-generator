#!/usr/bin/env python3
# build_video_single.py
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

SAFE_TOP_MARGIN = 300
SAFE_BOTTOM_MARGIN = 420
SAFE_LEFT_MARGIN = 160
SAFE_RIGHT_MARGIN = 160

# --- CARD / LAYOUT (as in the mockup) ---
CARD_LEFT = SAFE_LEFT_MARGIN
CARD_RIGHT = VIDEO_SIZE[0] - SAFE_RIGHT_MARGIN
CARD_TOP = SAFE_TOP_MARGIN
CARD_BOTTOM = VIDEO_SIZE[1] - SAFE_BOTTOM_MARGIN

STRIPE_WIDTH = 64  # vertical stripe with title on the right

# area inside the frame
INNER_LEFT = CARD_LEFT
INNER_RIGHT = CARD_RIGHT
INNER_TOP = CARD_TOP
INNER_BOTTOM = CARD_BOTTOM

# cover: square on the left, the rest is for the stripe
COVER_SIZE = INNER_RIGHT - INNER_LEFT
COVER_LEFT = INNER_LEFT
COVER_TOP = INNER_TOP + STRIPE_WIDTH

# radius of circular mask for vinyl_movie
COVER_CIRCLE_RADIUS = COVER_SIZE * 0.42

# text under the cover
TEXT_TOP_GAP = 16
TEXT_FONT_SIZE = 42

# waveform area (now used for _waveform.mp4 placement)
TEXT_TO_WAVEFORM_GAP = 0
WAVEFORM_HEIGHT = 120

# vertical title
TITLE_FONT_SIZE = 64

FONT_CANDIDATES = [
    "./DoppioOne-Regular.ttf",
    "/System/Library/Fonts/Supplemental/DIN Condensed Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
]

# text animation
TEXT_LINE_STAGGER = 0.4        # delay between lines
TEXT_LINE_ANIM_DURATION = 0.4  # single line animation duration (seconds)

WAVEFORM_TOP = COVER_TOP + COVER_SIZE + TEXT_TO_WAVEFORM_GAP
TEXT_TOP = WAVEFORM_TOP + WAVEFORM_HEIGHT + 1

TEXT_BG_COLOR = (22, 22, 29, 220)


# ---------- Helpers ----------

def find_background_video(folder: str) -> str:
    for name in sorted(os.listdir(folder)):
        if name.lower().startswith("_"):
            continue
        if name.lower().endswith(".mp4"):
            return os.path.join(folder, name)
    raise FileNotFoundError("No .mp4 background video found in folder")


def load_font(font_size: int) -> ImageFont.FreeTypeFont:
    for path in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, font_size)
        except Exception:
            continue
    print("[WARN] Could not load any TTF font, using default bitmap font.")
    return ImageFont.load_default()


def resize_video_clip_clip_safe(clip, target_size):
    tw, th = target_size

    def resize_frame(frame):
        img = Image.fromarray(frame)
        img = img.resize((tw, th), resample=Image.Resampling.LANCZOS)
        return np.array(img)

    return clip.fl_image(resize_frame)


def create_text_image(
    label_text: str,
    font_size: int,
    align: str = "center",
    max_width: int | None = None,
) -> np.ndarray:
    """
    Respects existing '\n'. If max_width is given and there is no '\n',
    performs a simple word-wrap. No shadow – plain text only.
    """
    font = load_font(font_size)

    # --- no max_width: use multiline, respect '\n' ---
    if max_width is None:
        dummy = Image.new("RGBA", (2000, 800), (0, 0, 0, 0))
        draw = ImageDraw.Draw(dummy)

        bbox = draw.multiline_textbbox((0, 0), label_text, font=font, spacing=10)
        x0, y0, x1, y1 = bbox
        text_w = x1 - x0
        text_h = y1 - y0

        pad_x = 20
        pad_y = 0
        img_w = text_w + 2 * pad_x
        img_h = text_h * 2

        img = Image.new("RGBA", (img_w, img_h), TEXT_BG_COLOR)
        draw = ImageDraw.Draw(img)

        if align == "center":
            x_text = (img_w - text_w) // 2
        else:
            x_text = pad_x
        y_text = pad_y

        draw.multiline_text(
            (x_text, y_text),
            label_text,
            font=font,
            fill=(255, 255, 255, 255),
            spacing=0,
            align=align,
        )
        return np.array(img)

    # --- max_width != None and already has '\n' -> use as-is ---
    if "\n" in label_text:
        text = label_text
        dummy = Image.new("RGBA", (max_width, 800), (0, 0, 0, 0))
        draw = ImageDraw.Draw(dummy)
        bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=10)
        x0, y0, x1, y1 = bbox
        text_w = x1 - x0
        text_h = y1 - y0

        pad_x = 20
        pad_y = 10
        img_w = min(max_width, text_w + 2 * pad_x)
        img_h = text_h + 2 * pad_y

        img = Image.new("RGBA", (img_w, img_h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        if align == "center":
            x_text = (img_w - text_w) // 2
        else:
            x_text = pad_x
        y_text = pad_y

        draw.multiline_text(
            (x_text, y_text),
            text,
            font=font,
            fill=(255, 255, 255, 255),
            spacing=10,
            align=align,
        )
        return np.array(img)

    # --- max_width != None and no '\n' -> word-wrap ---
    words = label_text.split()
    lines = []
    current = ""
    dummy = Image.new("RGBA", (max_width, 400), (0, 0, 0, 0))
    draw = ImageDraw.Draw(dummy)

    for w in words:
        test = (current + " " + w).strip()
        bbox = draw.textbbox((0, 0), test, font=font)
        w_test = bbox[2] - bbox[0]
        if w_test <= max_width - 40 or not current:
            current = test
        else:
            lines.append(current)
            current = w
    if current:
        lines.append(current)

    text = "\n".join(lines)
    bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=10, align="center")
    x0, y0, x1, y1 = bbox
    text_w = x1 - x0
    text_h = y1 - y0

    pad_x = 20
    pad_y = 10
    img_w = text_w + 2 * pad_x
    img_h = text_h + 2 * pad_y

    img = Image.new("RGBA", (img_w, img_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    if align == "center":
        x_text = (img_w - text_w) // 2
    else:
        x_text = pad_x
    y_text = pad_y

    draw.multiline_text(
        (x_text, y_text),
        text,
        font=font,
        fill=(255, 255, 255, 255),
        spacing=10,
        align=align,
    )
    return np.array(img)


def create_text_line_image(
    label_text: str,
    font_size: int,
    color: tuple[int, int, int],
    max_width: int,
) -> np.ndarray:
    """
    Single line of text (no '\n'), no shadow, in the given color.
    Used for 3 lines under the cover so they can have separate colors and clips.
    """
    font = load_font(font_size)

    dummy = Image.new("RGBA", (max_width, 400), (0, 0, 0, 0))
    draw = ImageDraw.Draw(dummy)

    bbox = draw.textbbox((0, 0), label_text, font=font)
    x0, y0, x1, y1 = bbox
    text_w = x1 - x0
    text_h = y1 - y0

    pad_x = 20
    pad_y = 0
    img_w = min(max_width, text_w + 2 * pad_x)
    img_h = int(text_h * 2)

    img = Image.new("RGBA", (img_w, img_h), TEXT_BG_COLOR)
    draw = ImageDraw.Draw(img)

    x_text = pad_x
    y_text = pad_y

    draw.text(
        (x_text, y_text),
        label_text,
        font=font,
        fill=(color[0], color[1], color[2], 255),
    )

    return np.array(img)


def rgba_to_imageclip(
    rgba_arr: np.ndarray,
    duration: float,
    start: float = 0.0,
    position=("center", "center"),
) -> ImageClip:
    rgb = rgba_arr[..., :3]
    alpha = rgba_arr[..., 3] / 255.0

    img_clip = ImageClip(rgb)
    mask_clip = ImageClip(alpha, ismask=True)

    img_clip = img_clip.set_duration(duration).set_start(start).set_position(position)
    mask_clip = mask_clip.set_duration(duration).set_start(start).set_position(position)

    img_clip = img_clip.set_mask(mask_clip)
    return img_clip


def make_card_bg_clip(duration: float) -> ImageClip:
    """
    Black card/frame (as in the mockup).
    """
    img = Image.new("RGBA", (VIDEO_SIZE[0], VIDEO_SIZE[1]), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # whole card
    draw.rectangle(
        [CARD_LEFT, CARD_TOP, CARD_RIGHT, CARD_BOTTOM],
        fill=(0, 0, 0, 200),
        outline=(0, 0, 0, 200),
        width=0,
    )

    return rgba_to_imageclip(np.array(img), duration=duration, start=0, position=(0, 0))


def make_title_clip(folder_title: str, duration: float) -> ImageClip:
    """
    Vertical text on the right edge of the card (inside STRIPE_WIDTH).
    """
    text = folder_title.upper()
    arr = create_text_image(text, TITLE_FONT_SIZE, align="center")
    img = Image.fromarray(arr)
    arr_rot = np.array(img)

    x = CARD_LEFT - img.width // 2 + (CARD_RIGHT - CARD_LEFT) // 2
    y = CARD_TOP

    return rgba_to_imageclip(arr_rot, duration=duration, start=0, position=(x, y))


def make_line_position_fn(
    line_index: int,
    y_final: int,
    line_height: int,
    x_pos: int,
):
    """
    Returns a position function (x, y) for a text line:
    - starts below the final position,
    - slides in upward over TEXT_LINE_ANIM_DURATION,
    - each line has a staggered delay of TEXT_LINE_STAGGER * index.
    """

    def pos(t: float):
        # local time for this line (including stagger)
        local_t = t - line_index * TEXT_LINE_STAGGER

        if local_t <= 0:
            # before animation start – hidden below
            y = y_final + line_height
        elif local_t >= TEXT_LINE_ANIM_DURATION:
            # after animation – in place
            y = y_final
        else:
            # during animation – linear slide up
            alpha = local_t / TEXT_LINE_ANIM_DURATION
            # start: y_final + line_height, end: y_final
            y = (y_final + line_height) - line_height * alpha

        return (x_pos, int(y))

    return pos


def make_circle_mask_array(size: int, radius: int) -> np.ndarray:
    """
    Create a 2D circular mask (float 0..1) with the given radius, centered.
    """
    cx = size / 2.0
    cy = size / 2.0
    Y, X = np.ogrid[:size, :size]
    dist_sq = (X - cx) ** 2 + (Y - cy) ** 2
    mask = (dist_sq <= radius**2).astype("float32")
    return mask


def make_vinyl_video_clip(
    vinyl_path: str,
    total_duration: float,
) -> VideoClip:
    """
    Load video from vinyl_movie, loop it to total_duration,
    crop to a square COVER_SIZE x COVER_SIZE and apply a circular mask.
    """
    base = VideoFileClip(vinyl_path).without_audio()

    if base.duration <= 0:
        raise ValueError(f"vinyl_movie has non-positive duration: {vinyl_path}")

    loops_needed = max(1, math.ceil(total_duration / base.duration))
    loop_clips = [base] * loops_needed
    full = concatenate_videoclips(loop_clips).subclip(0, total_duration)

    def crop_to_square(frame):
        img = Image.fromarray(frame)
        w, h = img.size
        scale = COVER_SIZE / min(w, h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = img.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)

        cx = new_w // 2
        cy = new_h // 2
        half = COVER_SIZE // 2
        left = max(0, cx - half)
        top = max(0, cy - half)
        right = left + COVER_SIZE
        bottom = top + COVER_SIZE
        img = img.crop((left, top, right, bottom))
        return np.array(img)

    square_clip = full.fl_image(crop_to_square)

    # circular mask
    circle_mask_arr = make_circle_mask_array(COVER_SIZE, COVER_CIRCLE_RADIUS)
    mask_clip = ImageClip(circle_mask_arr, ismask=True).set_duration(total_duration)

    square_clip = square_clip.set_duration(total_duration)
    square_clip = square_clip.set_mask(mask_clip)

    # position inside the card
    square_clip = square_clip.set_position((COVER_LEFT, COVER_TOP))

    return square_clip


# ---------- Main ----------

def main():
    if len(sys.argv) < 2:
        print("Usage: python build_video_single.py <folder_path>")
        sys.exit(1)

    folder = sys.argv[1]

    analysis_path = os.path.join(folder, "analysis.json")
    audio_path = os.path.join(folder, "_audio.mp3")
    waveform_video_path = os.path.join(folder, "_waveform.mp4")

    if not os.path.isfile(analysis_path):
        print(f"Error: {analysis_path} not found. Run analyze_tracks.py first.")
        sys.exit(1)
    if not os.path.isfile(audio_path):
        print(f"Error: {audio_path} not found. Run build_audio.py first.")
        sys.exit(1)
    if not os.path.isfile(waveform_video_path):
        print(f"Error: {waveform_video_path} not found. Generate _waveform.mp4 first.")
        sys.exit(1)

    folder_title = os.path.basename(os.path.normpath(folder)) or folder

    try:
        bg_video_path = find_background_video(folder)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    print(f"Using background video: {os.path.basename(bg_video_path)}")
    print(f"Using waveform video: {os.path.basename(waveform_video_path)}")

    with open(analysis_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    tracks = data.get("tracks", [])
    if not tracks:
        print("No tracks in analysis.json.")
        sys.exit(0)

    t0 = tracks[0]
    artist = t0["artist"]
    title = t0["title"]
    album = t0.get("album")
    year = t0.get("year")

    # old static cover kept as a fallback
    cover_path = t0.get("vinyl_art")
    vinyl_movie_path = t0.get("vinyl_movie")

    audio_clip = AudioFileClip(audio_path)
    total_duration = min(TOTAL_TARGET_SEC, audio_clip.duration)

    print(f"Creating single-track video ({total_duration:.2f}s).")

    base_bg = VideoFileClip(bg_video_path).without_audio()
    base_bg = resize_video_clip_clip_safe(base_bg, VIDEO_SIZE)
    loops_needed = max(1, math.ceil(total_duration / base_bg.duration))
    bg_loop_clips = [base_bg] * loops_needed
    bg_full = concatenate_videoclips(bg_loop_clips).subclip(0, total_duration)

    # card + vertical title
    title_clip = make_title_clip(folder_title, total_duration)

    # --- cover: now we use vinyl_movie with circular mask ---
    if vinyl_movie_path:
        # if path is relative – join with folder
        if not os.path.isabs(vinyl_movie_path):
            vinyl_movie_full = os.path.join(folder, vinyl_movie_path)
        else:
            vinyl_movie_full = vinyl_movie_path

        if not os.path.isfile(vinyl_movie_full):
            raise FileNotFoundError(f"vinyl_movie not found: {vinyl_movie_full}")

        cover_clip = make_vinyl_video_clip(vinyl_movie_full, total_duration)
    else:
        # fallback: old static cover
        if not cover_path:
            raise ValueError("Neither vinyl_movie nor vinyl_art provided in analysis.json")
        if not os.path.isabs(cover_path):
            cover_path_full = os.path.join(folder, cover_path)
        else:
            cover_path_full = cover_path

        cover_pil = Image.open(cover_path_full).convert("RGB")
        cw, ch = cover_pil.size
        scale = COVER_SIZE / min(cw, ch)
        new_w = int(cw * scale)
        new_h = int(ch * scale)
        cover_resized = cover_pil.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)

        cx, cy = new_w // 2, new_h // 2
        half = COVER_SIZE // 2
        left = max(0, cx - half)
        top = max(0, cy - half)
        right = left + COVER_SIZE
        bottom = top + COVER_SIZE
        cover_cropped = cover_resized.crop((left, top, right, bottom))

        cover_arr = np.array(cover_cropped)
        cover_clip = (
            ImageClip(cover_arr)
            .set_duration(total_duration)
            .set_start(0)
            .set_position((COVER_LEFT, COVER_TOP))
        )

    # --- text under the cover: separate lines, colors, and slide-up animation ---

    lines = [artist, title]
    if album and year:
        lines.append(year)

    text_top = TEXT_TOP
    max_text_width = INNER_RIGHT - INNER_LEFT

    line_clips = []
    line_heights = []
    line_x_positions = []

    for idx, line_text in enumerate(lines):
        if idx == 0:
            # first line – full white
            color = (255, 255, 255)
        else:
            # 2nd and 3rd lines – slightly darker
            color = (196, 196, 196)
        arr = create_text_line_image(
            line_text,
            TEXT_FONT_SIZE,
            color,
            max_width=max_text_width,
        )
        line_height = arr.shape[0]
        line_width = arr.shape[1]
        line_heights.append(line_height)

        # horizontally center inside [INNER_LEFT, INNER_RIGHT]
        x_centered = INNER_LEFT + (max_text_width - line_width) // 2
        line_x_positions.append(x_centered)

        clip = rgba_to_imageclip(
            arr,
            duration=total_duration,
            start=0,
            position=(x_centered, text_top),  # overridden by animated function later
        )

        line_clips.append(clip)

    # final positions of lines in the text block (vertical)
    LINE_GAP = 0
    y_positions = []
    current_y = text_top
    for h in line_heights:
        y_positions.append(current_y)
        current_y += h + LINE_GAP

    # override positions with functions that animate from below, keeping centered X
    for i, clip in enumerate(line_clips):
        y_final = y_positions[i]
        line_h = line_heights[i]
        x_pos = line_x_positions[i]
        pos_fn = make_line_position_fn(i, y_final, TEXT_FONT_SIZE * 6, x_pos)
        line_clips[i] = clip.set_position(pos_fn)

    # total height of the text block – used for cropping
    if line_heights:
        text_block_height = (y_positions[-1] + line_heights[-1]) - text_top
    else:
        text_block_height = 0

    # --- MASKED / CROPPED TEXT BLOCK CLIP ---
    if line_clips:
        text_block_clip_full = CompositeVideoClip(
            line_clips,
            size=VIDEO_SIZE,
        ).set_duration(total_duration)

        text_block_bottom = text_top + text_block_height

        text_block_cropped = text_block_clip_full.crop(
            x1=INNER_LEFT,
            y1=text_top,
            x2=INNER_RIGHT,
            y2=text_block_bottom,
        )

        text_block_cropped = text_block_cropped.set_position((INNER_LEFT, text_top))
    else:
        text_block_cropped = None

    # --- WAVEFORM VIDEO (_waveform.mp4) IN PLACE OF OLD WAVEFORM ---

    wave_top = WAVEFORM_TOP
    wave_height = WAVEFORM_HEIGHT
    wave_left = INNER_LEFT
    wave_width = INNER_RIGHT - INNER_LEFT

    waveform_raw = VideoFileClip(waveform_video_path).without_audio()

    # loop or trim waveform video to match total_duration
    if waveform_raw.duration < total_duration:
        wf_loops = max(1, math.ceil(total_duration / waveform_raw.duration))
        wf_loop_clips = [waveform_raw] * wf_loops
        waveform_full = concatenate_videoclips(wf_loop_clips).subclip(0, total_duration)
    else:
        waveform_full = waveform_raw.subclip(0, total_duration)

    waveform_resized = resize_video_clip_clip_safe(
        waveform_full, (wave_width, wave_height)
    ).set_position((wave_left, wave_top))

    # --- final composition ---

    overlay_clips = [
        bg_full,
        cover_clip,
        title_clip,
        waveform_resized,
    ]
    if text_block_cropped is not None:
        overlay_clips.append(text_block_cropped)

    final_video = CompositeVideoClip(
        overlay_clips,
        size=VIDEO_SIZE,
    ).set_duration(total_duration)

    final_video = final_video.set_audio(audio_clip.subclip(0, total_duration))

    out_video_path = os.path.join(folder, "_video_single.mp4")
    final_video.write_videofile(
        out_video_path,
        fps=30,
        codec="libx264",
        audio_codec="aac",
        threads=4,
        preset="medium",
    )

    print(f"\nSingle-track video saved to: {out_video_path}")


if __name__ == "__main__":
    main()
