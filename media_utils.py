# media_utils.py
import os
from io import BytesIO
from typing import Tuple, Optional, List

import numpy as np
from PIL import Image
from pydub import AudioSegment
import librosa

from mutagen.easyid3 import EasyID3
from mutagen.id3 import ID3, APIC, ID3NoHeaderError

SUPPORTED_EXT = (".mp3", ".wav", ".flac", ".m4a", ".ogg")
TOTAL_TARGET_SEC = 3.0


# ---------- AUDIO ANALYSIS (SAFE) ----------

def find_best_segment(filepath: str, target_sec: float = 5.0,
                      ignore_sec: float = 10.0) -> float:
    """
    Try to analyze the track with librosa and return the best segment start (seconds).
    If analysis fails, fall back to the middle of the track using pydub.
    """
    try:
        # Primary path: librosa analysis
        y, sr = librosa.load(filepath, sr=44100, mono=True)

        frame_length = int(0.05 * sr)  # 50 ms
        hop_length = frame_length      # no overlap

        rms = librosa.feature.rms(
            y=y,
            frame_length=frame_length,
            hop_length=hop_length
        )[0]

        onset_env = librosa.onset.onset_strength(
            y=y, sr=sr, hop_length=hop_length
        )

        L = min(len(rms), len(onset_env))
        rms = rms[:L]
        onset_env = onset_env[:L]

        frames_per_sec = sr / hop_length
        win_frames = int(target_sec * frames_per_sec)
        if win_frames <= 0:
            return 0.0

        ignore_frames = int(ignore_sec * frames_per_sec)
        start = ignore_frames
        end = L - ignore_frames - win_frames

        if end <= start:
            start = 0
            end = L - win_frames

        if end <= start:
            return 0.0

        def norm(x: np.ndarray) -> np.ndarray:
            mn = x.min()
            mx = x.max()
            if mx == mn:
                return x
            return (x - mn) / (mx - mn)

        rms_n = norm(rms)
        onset_n = norm(onset_env)

        alpha = 0.7
        beta = 0.3

        best_score = -1.0
        best_frame = start

        for i in range(start, end):
            j = i + win_frames
            mean_rms = rms_n[i:j].mean()
            mean_onset = onset_n[i:j].mean()
            score = alpha * mean_rms + beta * mean_onset
            if score > best_score:
                best_score = score
                best_frame = i

        best_start_sec = best_frame * hop_length / sr
        return float(best_start_sec)

    except Exception as e:
        print(f"  [WARN] librosa failed on {filepath}: {e}")
        print("  [WARN] Falling back to middle-of-track segment.")
        # Fallback: pydub + middle segment
        try:
            audio_full = AudioSegment.from_file(filepath)
        except Exception as e2:
            print(f"  [ERROR] pydub also failed on {filepath}: {e2}")
            return 0.0

        duration_sec = len(audio_full) / 1000.0
        if duration_sec <= target_sec:
            return 0.0
        start_sec = max(0.0, (duration_sec - target_sec) / 2.0)
        return float(start_sec)


# ---------- METADATA & COVER (ID3 ONLY) ----------

def extract_metadata_and_cover(filepath: str):
    """
    Extract:
      - artist (str)
      - title (str)
      - album (Optional[str])
      - year (Optional[str])
      - cover_image (PIL.Image RGB or None)

    Uses only ID3 / EasyID3, no audio parsing.
    """
    basename = os.path.splitext(os.path.basename(filepath))[0]
    artist = "Unknown Artist"
    title = basename
    album = None
    year = None
    cover_image = None

    # EasyID3 (text tags)
    try:
        tags = EasyID3(filepath)

        if "artist" in tags and tags["artist"]:
            artist = tags["artist"][0]

        if "title" in tags and tags["title"]:
            title = tags["title"][0]

        if "album" in tags and tags["album"]:
            album = tags["album"][0]

        # Try to resolve year
        for key in ("date", "originaldate", "year"):
            if key in tags and tags[key]:
                raw = tags[key][0]
                digits = "".join(ch for ch in raw if ch.isdigit())
                if len(digits) >= 4:
                    year = digits[:4]
                    break

    except Exception as e:
        print(f"  [WARN] Could not read EasyID3 tags for {basename}: {e}")

    # ID3 for cover art (APIC)
    try:
        id3 = ID3(filepath)
        apic_frames = [f for f in id3.values() if isinstance(f, APIC)]
        if apic_frames:
            apic = apic_frames[0]
            img_data = apic.data
            cover_image = Image.open(BytesIO(img_data)).convert("RGB")
    except ID3NoHeaderError:
        print(f"  [INFO] No ID3 header in {basename} (no cover art).")
    except Exception as e:
        print(f"  [WARN] Could not read cover art for {basename}: {e}")

    return artist, title, album, year, cover_image


def make_square_cover_array(img: Image.Image, size: int = 800) -> np.ndarray:
    """
    Crop image to square, resize using Pillow (Pillow>=10-safe),
    return as a numpy array (for ImageClip).
    """
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2

    img_cropped = img.crop((left, top, left + side, top + side))
    img_resized = img_cropped.resize(
        (size, size),
        resample=Image.Resampling.LANCZOS
    )
    return np.array(img_resized)


def placeholder_cover_array(size: int = 800) -> np.ndarray:
    img = Image.new("RGB", (size, size), color=(30, 30, 30))
    return np.array(img)
