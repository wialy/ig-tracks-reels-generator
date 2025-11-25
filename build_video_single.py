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
import librosa

from media_utils import TOTAL_TARGET_SEC

VIDEO_SIZE = (1080, 1920)  # (width, height)

SAFE_TOP_MARGIN = 420
SAFE_BOTTOM_MARGIN = 420
SAFE_LEFT_MARGIN = 160
SAFE_RIGHT_MARGIN = 160

# --- CARD / LAYOUT (jak na mocku) ---
CARD_LEFT = SAFE_LEFT_MARGIN
CARD_RIGHT = VIDEO_SIZE[0] - SAFE_RIGHT_MARGIN
CARD_TOP = SAFE_TOP_MARGIN
CARD_BOTTOM = VIDEO_SIZE[1] - SAFE_BOTTOM_MARGIN
CARD_BORDER_WIDTH = 12

STRIPE_WIDTH = 64  # pionowy pasek z tytułem po prawej

# obszar wewnątrz ramki
INNER_LEFT = CARD_LEFT + CARD_BORDER_WIDTH
INNER_RIGHT = CARD_RIGHT - CARD_BORDER_WIDTH
INNER_TOP = CARD_TOP + CARD_BORDER_WIDTH
INNER_BOTTOM = CARD_BOTTOM - CARD_BORDER_WIDTH

# okładka: kwadrat po lewej, reszta na stripe
COVER_SIZE = INNER_RIGHT - INNER_LEFT - STRIPE_WIDTH
COVER_LEFT = INNER_LEFT
COVER_TOP = INNER_TOP

# tekst pod okładką
TEXT_TOP_GAP = 16
TEXT_FONT_SIZE = 28

# waveform
TEXT_TO_WAVEFORM_GAP = 32
WAVEFORM_HEIGHT = CARD_BOTTOM - CARD_TOP - COVER_SIZE - TEXT_TOP_GAP - TEXT_TO_WAVEFORM_GAP * 3 - TEXT_FONT_SIZE * 3

# tytuł pionowy
TITLE_FONT_SIZE = 32

# waveform analysis
WAVEFORM_BINS = 256
WAVEFORM_NFFT = 2048
WAVEFORM_HOP = 512

FONT_CANDIDATES = [
    './Instagram Sans Bold.ttf',
    './DoppioOne-Regular.ttf',
    "/System/Library/Fonts/Supplemental/DIN Condensed Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
]


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


def create_text_image(label_text: str,
                      font_size: int,
                      align: str = "center",
                      max_width: int | None = None) -> np.ndarray:
    """
    Szanuje istniejące '\n'. Jeśli max_width podany i nie ma '\n', robi prosty word-wrap.
    """
    font = load_font(font_size)

    # --- bez max_width: używamy multiline, szanujemy \n ---
    if max_width is None:
        dummy = Image.new("RGBA", (2000, 800), (0, 0, 0, 0))
        draw = ImageDraw.Draw(dummy)

        bbox = draw.multiline_textbbox((0, 0), label_text, font=font, spacing=10)
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
            label_text,
            font=font,
            fill=(255, 255, 255, 255),
            spacing=10,
            align=align,
        )
        return np.array(img)

    # --- max_width != None i already has '\n' -> użyj jak jest ---
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

        shadow_offset = (2, 3)
        draw.multiline_text(
            (x_text + shadow_offset[0], y_text + shadow_offset[1]),
            text,
            font=font,
            fill=(0, 0, 0, 200),
            spacing=10,
            align=align,
        )
        draw.multiline_text(
            (x_text, y_text),
            text,
            font=font,
            fill=(255, 255, 255, 255),
            spacing=10,
            align=align,
        )
        return np.array(img)

    # --- max_width != None i brak '\n' -> word-wrap ---
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
    bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=10)
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

    shadow_offset = (2, 3)
    draw.multiline_text(
        (x_text + shadow_offset[0], y_text + shadow_offset[1]),
        text,
        font=font,
        fill=(0, 0, 0, 200),
        spacing=10,
        align=align,
    )
    draw.multiline_text(
        (x_text, y_text),
        text,
        font=font,
        fill=(255, 255, 255, 255),
        spacing=10,
        align=align,
    )
    return np.array(img)


def rgba_to_imageclip(rgba_arr: np.ndarray,
                      duration: float,
                      start: float = 0.0,
                      position=("center", "center")) -> ImageClip:
    rgb = rgba_arr[..., :3]
    alpha = rgba_arr[..., 3] / 255.0

    img_clip = ImageClip(rgb)
    mask_clip = ImageClip(alpha, ismask=True)

    img_clip = img_clip.set_duration(duration).set_start(start).set_position(position)
    mask_clip = mask_clip.set_duration(duration).set_start(start).set_position(position)

    img_clip = img_clip.set_mask(mask_clip)
    return img_clip


def compute_waveform_bands(audio_path: str,
                           total_duration: float,
                           sr_target: int | None = None):
    y, sr = librosa.load(audio_path, sr=sr_target, mono=True)
    max_samples = int(total_duration * sr)
    if len(y) > max_samples:
        y = y[:max_samples]

    S = np.abs(librosa.stft(y, n_fft=WAVEFORM_NFFT, hop_length=WAVEFORM_HOP))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=WAVEFORM_NFFT)

    low_max = 200.0
    mid_min = 200.0
    mid_max = 2000.0
    high_min = 2000.0

    low_idx = np.where(freqs <= low_max)[0]
    mid_idx = np.where((freqs > mid_min) & (freqs <= mid_max))[0]
    high_idx = np.where(freqs > high_min)[0]

    def band_mean(mag, idx):
        if len(idx) == 0:
            return np.zeros(mag.shape[1])
        return mag[idx, :].mean(axis=0)

    low = band_mean(S, low_idx)
    mid = band_mean(S, mid_idx)
    high = band_mean(S, high_idx)

    def norm_band(b):
        b = b.astype(float)
        if b.max() > 0:
            b = b / b.max()
        return b

    low = norm_band(low)
    mid = norm_band(mid)
    high = norm_band(high)

    bands = np.stack([low, mid, high], axis=1)

    n_frames = bands.shape[0]
    if n_frames == 0:
        return np.zeros((WAVEFORM_BINS, 3), dtype=float)

    xs = np.linspace(0, n_frames - 1, WAVEFORM_BINS)
    bands_resampled = np.zeros((WAVEFORM_BINS, 3), dtype=float)
    for c in range(3):
        bands_resampled[:, c] = np.interp(xs, np.arange(n_frames), bands[:, c])

    bands_resampled = np.clip(bands_resampled, 0.0, 1.0)
    return bands_resampled


def make_waveform_clip(bands: np.ndarray,
                       total_duration: float,
                       wave_left: int,
                       wave_top: int,
                       wave_width: int,
                       wave_height: int) -> VideoClip:
    """
    Przezroczysty RGB waveform + czerwona linia progresu.
    Linijki blisko aktualnego progresu są powiększane (do 1.5x),
    z liniową interpolacją w zależności od odległości.
    """
    BAR_WIDTH = 4
    BAR_GAP = 1

    MAX_SCALE = 1.5             # wysokość dokładnie pod wskaźnikiem progresu
    HIGHLIGHT_RADIUS_BINS = 3   # w ilu "słupkach" wygasa efekt (liniowo)

    n_bins = bands.shape[0]
    if n_bins <= 0:
        n_bins = 1
        bands = np.zeros((1, 3), dtype=float)

    total_bar_width = n_bins * (BAR_WIDTH + BAR_GAP)
    if total_bar_width > wave_width:
        scale = wave_width / total_bar_width
        new_bins = max(1, int(n_bins * scale))
        xs = np.linspace(0, n_bins - 1, new_bins)
        bands = np.stack([
            np.interp(xs, np.arange(n_bins), bands[:, 0]),
            np.interp(xs, np.arange(n_bins), bands[:, 1]),
            np.interp(xs, np.arange(n_bins), bands[:, 2]),
        ], axis=1)
        n_bins = new_bins
        total_bar_width = n_bins * (BAR_WIDTH + BAR_GAP)

    H, W = VIDEO_SIZE[1], VIDEO_SIZE[0]
    y_center = wave_top + wave_height // 2

    # precompute bazowe amplitudy i pozycje słupków (bez skalowania)
    base_low = np.zeros(n_bins, dtype=int)
    base_mid = np.zeros(n_bins, dtype=int)
    base_high = np.zeros(n_bins, dtype=int)
    x_starts = np.zeros(n_bins, dtype=int)
    x_ends = np.zeros(n_bins, dtype=int)

    for i in range(n_bins):
        L, M, Hband = bands[i]
        base_low[i] = int(L * (wave_height / 2))
        base_mid[i] = int(M * (wave_height / 2))
        base_high[i] = int(Hband * (wave_height / 2))

        x_start = wave_left + i * (BAR_WIDTH + BAR_GAP)
        x_end = x_start + BAR_WIDTH
        x_starts[i] = x_start
        x_ends[i] = x_end

    def draw_band(img: np.ndarray, height: int, x_start: int, x_end: int, color: np.ndarray):
        """Rysuje pojedynczy pasek (góra+dół) o danej wysokości."""
        if height <= 0:
            return

        y0_top = max(wave_top, y_center - height)
        y1_top = y_center
        y0_bot = y_center
        y1_bot = min(wave_top + wave_height, y_center + height)

        if y0_top < y1_top:
            img[y0_top:y1_top, x_start:x_end, :] = np.maximum(
                img[y0_top:y1_top, x_start:x_end, :],
                color
            )
        if y0_bot < y1_bot:
            img[y0_bot:y1_bot, x_start:x_end, :] = np.maximum(
                img[y0_bot:y1_bot, x_start:x_end, :],
                color
            )

    def make_rgb_frame(t: float):
        # startujemy od całkowicie czarnego obrazu (tło przezroczyste po masce)
        img = np.zeros((H, W, 3), dtype=np.uint8)

        # aktualny progres 0..1
        if total_duration <= 0:
            progress = 0.0
        else:
            progress = max(0.0, min(1.0, t / total_duration))

        # pozycja piksela czerwonej linii
        px = wave_left + int(progress * wave_width)
        px = max(wave_left, min(px, wave_left + wave_width - 1))

        # który bin jest "pod" wskaźnikiem
        progress_bin = progress * (n_bins - 1)

        for i in range(n_bins):
            # skalowanie wysokości w zależności od odległości od progress_bin
            dist = abs(i - progress_bin)
            if dist >= HIGHLIGHT_RADIUS_BINS:
                scale = 1.0
            else:
                scale = 1.0 + (MAX_SCALE - 1.0) * (1.0 - dist / HIGHLIGHT_RADIUS_BINS)

            h_low = int(base_low[i] * scale)
            h_mid = int(base_mid[i] * scale)
            h_high = int(base_high[i] * scale)

            x_start = x_starts[i]
            x_end = x_ends[i]

            # rysujemy 3 pasma w RGB
            draw_band(img, h_low,  x_start, x_end, np.array([255,   0,   0], dtype=np.uint8))
            draw_band(img, h_mid,  x_start, x_end, np.array([  0, 255,   0], dtype=np.uint8))
            draw_band(img, h_high, x_start, x_end, np.array([  0,   0, 255], dtype=np.uint8))

        # czerwona linia progresu na wierzchu
        img[wave_top:wave_top + wave_height, px:px + 2, :] = np.array([255, 80, 80], dtype=np.uint8)

        return img

    # Clip RGB
    color_clip = VideoClip(make_rgb_frame, duration=total_duration)

    # Zbuduj maskę z jasności pikseli (czarne = 0 → w pełni przezroczyste)
    mask_clip = color_clip.to_mask()

    return color_clip.set_mask(mask_clip)


def make_card_bg_clip(duration: float) -> ImageClip:
    """
    Czarna karta/ramka jak na mocku.
    """
    img = Image.new("RGBA", (VIDEO_SIZE[0], VIDEO_SIZE[1]), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # cała karta
    draw.rectangle(
        [CARD_LEFT, CARD_TOP, CARD_RIGHT, CARD_BOTTOM],
        fill=(0, 0, 0, 200),
        outline=(0, 0, 0, 200),
        width=CARD_BORDER_WIDTH,
    )

    return rgba_to_imageclip(np.array(img), duration=duration, start=0, position=(0, 0))


def make_vertical_title_clip(folder_title: str, duration: float) -> ImageClip:
    """
    Pionowy napis na prawej krawędzi karty (w pasie STRIPE_WIDTH).
    """
    text = folder_title.upper()
    arr = create_text_image(text, TITLE_FONT_SIZE, align="center")
    img = Image.fromarray(arr).rotate(90, expand=True)
    arr_rot = np.array(img)

    
    x = CARD_RIGHT - STRIPE_WIDTH * 1.1
    y = CARD_TOP

    return rgba_to_imageclip(arr_rot, duration=duration, start=0, position=(x, y))


# ---------- Main ----------

def main():
    if len(sys.argv) < 2:
        print("Usage: python build_video_single.py <folder_path>")
        sys.exit(1)

    folder = sys.argv[1]

    analysis_path = os.path.join(folder, "analysis.json")
    audio_path = os.path.join(folder, "_audio.mp3")

    if not os.path.isfile(analysis_path):
        print(f"Error: {analysis_path} not found. Run analyze_tracks.py first.")
        sys.exit(1)
    if not os.path.isfile(audio_path):
        print(f"Error: {audio_path} not found. Run build_audio.py first.")
        sys.exit(1)

    folder_title = os.path.basename(os.path.normpath(folder)) or folder

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

    t0 = tracks[0]
    artist = t0["artist"]
    title = t0["title"]
    album = t0.get("album")
    year = t0.get("year")
    cover_path = t0["cover_path"]

    audio_clip = AudioFileClip(audio_path)
    total_duration = min(TOTAL_TARGET_SEC, audio_clip.duration)

    print(f"Creating single-track video ({total_duration:.2f}s).")

    base_bg = VideoFileClip(bg_video_path).without_audio()
    base_bg = resize_video_clip_clip_safe(base_bg, VIDEO_SIZE)
    loops_needed = max(1, math.ceil(total_duration / base_bg.duration))
    bg_loop_clips = [base_bg] * loops_needed
    bg_full = concatenate_videoclips(bg_loop_clips).subclip(0, total_duration)

    # karta + pionowy tytuł
    card_bg_clip = make_card_bg_clip(total_duration)
    vertical_title_clip = make_vertical_title_clip(folder_title, total_duration)

    # okładka w kwadracie po lewej
    cover_pil = Image.open(cover_path).convert("RGB")
    cw, ch = cover_pil.size
    scale = COVER_SIZE / min(cw, ch)
    new_w = int(cw * scale)
    new_h = int(ch * scale)
    cover_resized = cover_pil.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)

    # przytnij do kwadratu COVER_SIZE x COVER_SIZE (środek)
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

    # tekst pod okładką
    lines = [artist, title]
    if album:
        if year:
            lines.append(f"{album} ({year})")
        else:
            lines.append(album)
    text_label = "\n".join(lines)

    text_arr = create_text_image(
        text_label,
        TEXT_FONT_SIZE,
        align="left",
        max_width=INNER_RIGHT - INNER_LEFT,
    )
    text_h = text_arr.shape[0]
    text_top = COVER_TOP + COVER_SIZE + TEXT_TOP_GAP

    text_clip = rgba_to_imageclip(
        text_arr,
        duration=total_duration,
        start=0,
        position=(INNER_LEFT, text_top),
    )

    # waveform w dolnej części karty
    wave_top = text_top + text_h + TEXT_TO_WAVEFORM_GAP
    wave_height = WAVEFORM_HEIGHT
    wave_left = INNER_LEFT
    wave_width = INNER_RIGHT - INNER_LEFT

    print("Computing waveform bands...")
    bands = compute_waveform_bands(audio_path, total_duration)

    # szare tło pod waveformem (prostokąt)
    wave_bg_img = Image.new("RGBA", (VIDEO_SIZE[0], VIDEO_SIZE[1]), (0, 0, 0, 0))
    draw = ImageDraw.Draw(wave_bg_img)
    draw.rectangle(
        [wave_left, wave_top, wave_left + wave_width, wave_top + wave_height],
        fill=(0, 0, 0, 0),
    )
    wave_bg_clip = rgba_to_imageclip(
        np.array(wave_bg_img),
        duration=total_duration,
        start=0,
        position=(0, 0),
    )

    waveform_clip = make_waveform_clip(
        bands=bands,
        total_duration=total_duration,
        wave_left=wave_left,
        wave_top=wave_top,
        wave_width=wave_width,
        wave_height=wave_height,
    )

    final_video = CompositeVideoClip(
        [
            bg_full,
            card_bg_clip,
            wave_bg_clip,
            waveform_clip,
            cover_clip,
            text_clip,
            vertical_title_clip,
        ],
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
