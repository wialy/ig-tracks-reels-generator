#!/usr/bin/env python3
# build_video_carousel.py
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

# --- ANIM CONSTANTS ---
COVER_TRANSITION = 0.6       # sekundy: czas animacji przesunięcia okładek między segmentami
COVER_PREVIEW_SCALE = 0.8    # skala okładki w podglądzie (lewa/prawa)
TEXT_SLIDE_DURATION = 0.3    # jak długo linia tekstu „wjeżdża” z prawej
TEXT_LINE_STAGGER = 0.1      # opóźnienie między liniami tekstu (druga vs pierwsza itd.)

# --- LAYOUT ---
CENTER_COVER_SIZE = 640      # bazowy rozmiar okładki w centrum (px)
COVER_Y = VIDEO_SIZE[1] // 2.2  # środek ekranu w pionie

# Odległości od okładki w górę
GAP_INDEX_TO_COVER = 40      # gap między BOTTOM track-index a TOP okładki
GAP_TITLE_TO_INDEX = 32      # odstęp między folder name a Track X / N

# Okładki: pozycje środków (x) dla lewa / środek / prawa
CENTER_X = VIDEO_SIZE[0] // 2
COVER_OFFSET_X = 640         # odległość lewa/prawa okładka od środka

# Teksty pod okładką
TEXT_LINE_SPACING = 80       # odstęp między liniami tekstu

# FONTY
TITLE_FONT_SIZE = 80          # folder name
TRACK_INDEX_FONT_SIZE = 60    # "Track 1 / 3"
TEXT_FONT_SIZE = 70           # linie pod okładką

FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/DIN Condensed Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
]


# ---------- Helpers ----------

def find_background_video(folder: str) -> str:
    """
    Zwraca ścieżkę do pierwszego .mp4 w folderze (poza plikami zaczynającymi się od 'reel').
    """
    for name in sorted(os.listdir(folder)):
        if name.lower().startswith("reel"):
            continue
        if name.lower().endswith(".mp4"):
            return os.path.join(folder, name)
    raise FileNotFoundError("No .mp4 background video found in folder")


def load_font(font_size: int) -> ImageFont.FreeTypeFont:
    """
    Ładujemy TTF, albo fallback na domyślny font Pillow.
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
    Bezpieczne resize tła za pomocą Pillow (LANCZOS).
    """
    tw, th = target_size

    def resize_frame(frame):
        img = Image.fromarray(frame)
        img = img.resize((tw, th), resample=Image.Resampling.LANCZOS)
        return np.array(img)

    return clip.fl_image(resize_frame)


def create_text_image(label_text: str,
                      font_size: int) -> np.ndarray:
    """
    Renderuje tekst (biały + cień) jako RGBA (bez tła).
    """
    font = load_font(font_size)
    dummy = Image.new("RGBA", (2000, 400), (0, 0, 0, 0))
    draw = ImageDraw.Draw(dummy)

    bbox = draw.textbbox((0, 0), label_text, font=font)
    x0, y0, x1, y1 = bbox
    text_w = x1 - x0
    text_h = y1 - y0

    pad_x = 20
    pad_y = 10

    img_w = text_w + 2 * pad_x
    img_h = text_h + 2 * pad_y

    img = Image.new("RGBA", (img_w, img_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    x_text = pad_x
    y_text = pad_y

    # cień
    shadow_offset = (2, 3)
    draw.text(
        (x_text + shadow_offset[0], y_text + shadow_offset[1]),
        label_text,
        font=font,
        fill=(0, 0, 0, 200),
    )
    # główny tekst
    draw.text(
        (x_text, y_text),
        label_text,
        font=font,
        fill=(255, 255, 255, 255),
    )

    return np.array(img)


def rgba_to_imageclip(rgba_arr: np.ndarray,
                      duration: float,
                      start: float = 0.0,
                      position=("center", "center")) -> ImageClip:
    """
    RGBA -> ImageClip z maską alpha.
    """
    rgb = rgba_arr[..., :3]
    alpha = rgba_arr[..., 3] / 255.0

    img_clip = ImageClip(rgb)
    mask_clip = ImageClip(alpha, ismask=True)

    img_clip = img_clip.set_duration(duration).set_start(start).set_position(position)
    mask_clip = mask_clip.set_duration(duration).set_start(start).set_position(position)

    img_clip = img_clip.set_mask(mask_clip)
    return img_clip


def role_in_segment(track_idx: int, seg_idx: int, num_tracks: int) -> str:
    """
    Określa rolę okładki w danym segmencie:
    center / left / right / hidden
    """
    if seg_idx < 0 or seg_idx >= num_tracks:
        return "hidden"
    if track_idx == seg_idx:
        return "center"
    if track_idx == seg_idx - 1:
        return "left"
    if track_idx == seg_idx + 1:
        return "right"
    return "hidden"


def role_to_pos_scale_context(role: str,
                              base_size: int,
                              prev_role_for_hidden: str):
    """
    Zwraca ((cx, cy), scale) dla danej roli.
    Dla 'hidden' bierzemy pod uwagę poprzednią rolę:
    - jeśli schodzi z lewej, chowa się poza lewą krawędzią
    - w innych przypadkach poza prawą.
    """
    if role == "hidden":
        if prev_role_for_hidden == "left":
            # schowaj poza lewą
            off_x = CENTER_X - COVER_OFFSET_X - base_size * 1.5
        else:
            # schowaj poza prawą
            off_x = CENTER_X + COVER_OFFSET_X + base_size * 1.5
        return (off_x, COVER_Y), COVER_PREVIEW_SCALE

    if role == "center":
        return (CENTER_X, COVER_Y), 1.0
    if role == "left":
        return (CENTER_X - COVER_OFFSET_X, COVER_Y), COVER_PREVIEW_SCALE
    if role == "right":
        return (CENTER_X + COVER_OFFSET_X, COVER_Y), COVER_PREVIEW_SCALE

    # fallback
    return (CENTER_X, COVER_Y), COVER_PREVIEW_SCALE


def make_cover_clip(track_idx: int,
                    cover_arr: np.ndarray,
                    num_tracks: int,
                    total_duration: float,
                    seg_duration: float) -> ImageClip:
    """
    Tworzy ImageClip dla jednej okładki z pozycją i skalą zależną od czasu (karuzela),
    z własnym resize (bez MoviePy.resize i Image.ANTIALIAS).
    """
    base_size = CENTER_COVER_SIZE

    def role_and_progress(t: float):
        if t < 0:
            return "hidden", "hidden", 0.0

        seg_idx = int(t // seg_duration)
        if seg_idx >= num_tracks:
            seg_idx = num_tracks - 1
        local = t - seg_idx * seg_duration

        # jeśli to pierwszy segment lub jesteśmy poza oknem animacji – statycznie
        if seg_idx == 0 or local >= COVER_TRANSITION:
            r = role_in_segment(track_idx, seg_idx, num_tracks)
            return r, r, 0.0

        # początek segmentu -> przejście z poprzedniego układu
        prev_seg = seg_idx - 1
        prev_role = role_in_segment(track_idx, prev_seg, num_tracks)
        curr_role = role_in_segment(track_idx, seg_idx, num_tracks)
        p = max(0.0, min(1.0, local / COVER_TRANSITION))
        return prev_role, curr_role, p

    def scale_fn(t: float):
        prev_role, curr_role, p = role_and_progress(t)
        (_, _), s1 = role_to_pos_scale_context(prev_role, base_size, prev_role)
        (_, _), s2 = role_to_pos_scale_context(curr_role, base_size, prev_role)
        return s1 + (s2 - s1) * p

    def pos_center_fn(t: float):
        """
        Zwraca (cx, cy) – środek okładki dla danego czasu.
        """
        prev_role, curr_role, p = role_and_progress(t)
        (cx1, cy1), _ = role_to_pos_scale_context(prev_role, base_size, prev_role)
        (cx2, cy2), _ = role_to_pos_scale_context(curr_role, base_size, prev_role)
        cx = cx1 + (cx2 - cx1) * p
        cy = cy1 + (cy2 - cy1) * p
        return cx, cy

    base_clip = ImageClip(cover_arr).set_duration(total_duration)

    def fl(gf, t):
        frame = gf(0)  # (H, W, 3) – cover statyczny
        scale = scale_fn(t)
        img = Image.fromarray(frame)
        new_w = max(1, int(frame.shape[1] * scale))
        new_h = max(1, int(frame.shape[0] * scale))
        img = img.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)
        return np.array(img)

    resized_clip = base_clip.fl(fl, apply_to=[])

    def pos_topleft_fn(t):
        cx, cy = pos_center_fn(t)
        scale = scale_fn(t)
        size = base_size * scale
        x = cx - size / 2
        y = cy - size / 2
        return (x, y)

    return resized_clip.set_position(pos_topleft_fn)


def make_text_line_clip(text_arr: np.ndarray,
                        line_index: int,
                        seg_index: int,
                        seg_duration: float,
                        total_duration: float,
                        base_y: int) -> ImageClip:
    """
    Pojedyncza linia tekstu (artist / title / album) dla danego tracka:
    - start: na początku segmentu + stagger (0, 0.1, 0.2...)
    - wjeżdża z DOŁU na swoją docelową pozycję
    - znika po końcu segmentu (wyrzucona poza ekran w lewo)
    """
    line_start = seg_index * seg_duration + line_index * TEXT_LINE_STAGGER
    line_end = (seg_index + 1) * seg_duration

    text_w = text_arr.shape[1]
    text_h = text_arr.shape[0]

    # docelowa pozycja X/Y
    final_x = CENTER_X - text_w / 2
    final_y = base_y + line_index * TEXT_LINE_SPACING

    # start z dołu ekranu
    start_y = VIDEO_SIZE[1] + 50

    # po zakończeniu segmentu – wyrzuć w lewo (jak wcześniej)
    off_left_x = -text_w - 200

    def pos_fn(t: float):
        if t < line_start:
            # jeszcze przed startem – trzymaj pod ekranem
            return (final_x, start_y)
        if t >= line_end:
            # po końcu segmentu – wyrzuć poza ekran w lewo
            return (off_left_x, final_y)

        local = t - line_start
        if local >= TEXT_SLIDE_DURATION:
            # po zakończeniu animacji – na docelowej pozycji
            return (final_x, final_y)

        # animacja: z dołu w górę
        p = max(0.0, min(1.0, local / TEXT_SLIDE_DURATION))
        y = start_y + (final_y - start_y) * p
        return (final_x, y)

    clip = rgba_to_imageclip(
        text_arr,
        duration=total_duration,
        start=0,
        position=(0, 0),
    )
    clip = clip.set_position(pos_fn)
    return clip



# ---------- Main ----------

def main():
    if len(sys.argv) < 2:
        print("Usage: python build_video_carousel.py <folder_path>")
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
    seg_duration = total_duration / num_tracks

    print(f"Creating carousel-style video with {num_tracks} tracks.")

    # Background
    base_bg = VideoFileClip(bg_video_path).without_audio()
    base_bg = resize_video_clip_clip_safe(base_bg, VIDEO_SIZE)
    loops_needed = max(1, math.ceil(total_duration / base_bg.duration))
    bg_loop_clips = [base_bg] * loops_needed
    bg_full = concatenate_videoclips(bg_loop_clips).subclip(0, total_duration)

    # --- Layout zależny od okładki ---
    cover_top = COVER_Y - CENTER_COVER_SIZE // 2

    # Folder name (top), Track index nad okładką z GAPami

    # 1) Folder name obraz
    folder_arr = create_text_image(folder_title.upper(), TITLE_FONT_SIZE)
    folder_h = folder_arr.shape[0]

    # 2) Szablon do obliczenia wysokości track index (żeby nie wchodził w okładkę)
    idx_template_arr = create_text_image(f"Track 1 / {num_tracks}", TRACK_INDEX_FONT_SIZE)
    idx_h = idx_template_arr.shape[0]

    # Track index Y: tak, żeby BOTTOM indeksu był GAP_INDEX_TO_COVER nad górą okładki
    idx_y = cover_top - GAP_INDEX_TO_COVER - idx_h

    # Folder name Y: GAP_TITLE_TO_INDEX nad TOP indeksu
    folder_y = idx_y - GAP_TITLE_TO_INDEX - folder_h

    folder_clip = rgba_to_imageclip(
        folder_arr,
        duration=total_duration,
        start=0,
        position=("center", folder_y),
    )

    # Track index text (np. "Track 1 / 3"), osobny clip per segment
    track_index_clips = []
    for i in range(num_tracks):
        idx_label = f"Track {i + 1} / {num_tracks}"
        idx_arr = create_text_image(idx_label, TRACK_INDEX_FONT_SIZE)

        def make_index_clip(arr, seg_idx):
            seg_start = seg_idx * seg_duration
            seg_end = (seg_idx + 1) * seg_duration
            dur = seg_end - seg_start
            c = rgba_to_imageclip(
                arr,
                duration=dur,
                start=seg_start,
                position=("center", idx_y),
            )
            return c

        track_index_clips.append(make_index_clip(idx_arr, i))

    # Cover clips (karuzela)
    cover_clips = []
    for idx, t in enumerate(tracks):
        cover_path = t["cover_path"]
        cover_pil = Image.open(cover_path).convert("RGB")
        cover_pil_resized = cover_pil.resize(
            (CENTER_COVER_SIZE, CENTER_COVER_SIZE),
            resample=Image.Resampling.LANCZOS,
        )
        cover_arr = np.array(cover_pil_resized)

        clip = make_cover_clip(
            track_idx=idx,
            cover_arr=cover_arr,
            num_tracks=num_tracks,
            total_duration=total_duration,
            seg_duration=seg_duration,
        )
        cover_clips.append(clip)

    # Text (artist / title / album (year)) – osobny zestaw linii dla każdego tracka
    text_clips = []
    base_text_y = COVER_Y + CENTER_COVER_SIZE // 2 + 80  # pod okładką

    for i, t in enumerate(tracks):
        artist = t["artist"]
        title = t["title"]
        album = t.get("album")
        year = t.get("year")

        lines = [artist, title]
        if album:
            if year:
                lines.append(f"{album} ({year})")
            else:
                lines.append(album)

        for line_idx, line in enumerate(lines):
            line_arr = create_text_image(line, TEXT_FONT_SIZE)
            line_clip = make_text_line_clip(
                text_arr=line_arr,
                line_index=line_idx,
                seg_index=i,
                seg_duration=seg_duration,
                total_duration=total_duration,
                base_y=base_text_y,
            )
            text_clips.append(line_clip)

    # Składamy wszystko
    final_video = CompositeVideoClip(
        [
            bg_full,
            folder_clip,
            *track_index_clips,
            *cover_clips,
            *text_clips,
        ],
        size=VIDEO_SIZE,
    ).set_duration(total_duration)

    # Audio
    audio_clip = AudioFileClip(audio_path)
    final_video = final_video.set_audio(audio_clip)

    out_video_path = os.path.join(folder, "reel_carousel.mp4")
    final_video.write_videofile(
        out_video_path,
        fps=30,
        codec="libx264",
        audio_codec="aac",
        threads=4,
        preset="medium",
    )

    print(f"\nCarousel-style video saved to: {out_video_path}")


if __name__ == "__main__":
    main()
