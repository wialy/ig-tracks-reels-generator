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
TOTAL_TARGET_SEC = 18.0


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

# -- DROP DETECTED --

def _smooth(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return x
    kernel = np.ones(win) / win
    return np.convolve(x, kernel, mode="same")


def _zscore(x: np.ndarray) -> np.ndarray:
    m = np.mean(x)
    s = np.std(x) + 1e-9
    return (x - m) / s


def detect_main_drop_time(
    path: str,
    frame_length: int = 4096,
    hop_length: int = 1024,
    smooth_rms_win: int = 6,
    smooth_flux_win: int = 4,
    score_z_thresh: float = 1.8,
    min_intro_sec: float = 15.0,
    min_time_between_drops_sec: float = 6.0,
    min_break_sec: float = 2.0,
    pre_window_sec: float = 8.0,
) -> float | None:
    """
    Zwraca czas (w sekundach) głównego dropu w utworze albo None,
    jeśli nic sensownego nie udało się znaleźć.
    """
    # 1. Audio
    y, sr = librosa.load(path, sr=None, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)

    # 2. RMS
    rms = librosa.feature.rms(
        y=y,
        frame_length=frame_length,
        hop_length=hop_length,
    )[0]
    rms_smooth = _smooth(rms, smooth_rms_win)

    # 3. Spectral flux (zmiana widma)
    S = librosa.stft(y, n_fft=2048, hop_length=hop_length)
    mag = np.abs(S)
    flux = np.sqrt(np.sum(np.diff(mag, axis=1).clip(min=0) ** 2, axis=0))
    flux = np.concatenate([[flux[0]], flux])
    flux_smooth = _smooth(flux, smooth_flux_win)

    # 4. Wyrównanie długości
    L = min(len(rms_smooth), len(flux_smooth))
    rms_smooth = rms_smooth[:L]
    flux_smooth = flux_smooth[:L]

    # 5. Skoki RMS + poziom flux
    rms_diff = np.diff(rms_smooth, prepend=rms_smooth[0])
    rms_diff_z = _zscore(rms_diff)
    flux_z = _zscore(flux_smooth)

    score = rms_diff_z + flux_z

    frames = np.arange(L)
    times = librosa.frames_to_time(frames, sr=sr, hop_length=hop_length)

    # 6. Kandydaci
    raw_idxs = np.where(score > score_z_thresh)[0]
    if len(raw_idxs) == 0:
        return None

    min_break_frames = int(min_break_sec * sr / hop_length)
    pre_window_frames = int(pre_window_sec * sr / hop_length)
    min_time_between_frames = int(min_time_between_drops_sec * sr / hop_length)

    median_rms = np.median(rms_smooth)
    low_energy_threshold = 0.9 * median_rms

    candidate_times: list[float] = []
    candidate_scores: list[float] = []
    last_drop_frame = -10**9

    for idx in raw_idxs:
        t = times[idx]
        if t < min_intro_sec:
            continue

        if idx - last_drop_frame < min_time_between_frames:
            continue

        start_pre = max(0, idx - pre_window_frames)
        end_pre_break = max(0, idx - min_break_frames)
        if end_pre_break <= start_pre:
            continue

        pre_window = rms_smooth[start_pre:idx]
        pre_break = rms_smooth[start_pre:end_pre_break]
        if len(pre_window) < 4 or len(pre_break) < 4:
            continue

        mean_pre = np.mean(pre_window)
        mean_break = np.mean(pre_break)

        if mean_break < low_energy_threshold and mean_pre > mean_break * 1.15:
            candidate_times.append(t)
            candidate_scores.append(score[idx])
            last_drop_frame = idx

    # 7. Wybór głównego dropu
    if candidate_times:
        best_idx = int(np.argmax(candidate_scores))
        return float(candidate_times[best_idx])

    # 8. Fallback – jak nie spełniły się warunki breaku,
    # bierzemy globalne maksimum score (poza intrem)
    valid_mask = times >= min_intro_sec
    if not np.any(valid_mask):
        return None

    idxs_valid = np.where(valid_mask)[0]
    best_idx = idxs_valid[np.argmax(score[idxs_valid])]
    t_best = float(times[best_idx])

    # jak drop jest praktycznie na samym końcu, to też słabo
    if duration - t_best < 5.0:
        return None

    return t_best


def find_drop_centered_segment(
    filepath: str,
    target_sec: float,
    ignore_sec: float = 10.0,
) -> float:
    """
    Zwraca start wycinka tak, żeby główny drop (jeśli wykryty)
    był w jego okolicach środka. Jeśli dropu nie ma, fallback
    do find_best_segment.
    """
    try:
        drop_time = detect_main_drop_time(filepath)
    except Exception:
        drop_time = None

    if drop_time is None:
        # stara logika jako backup
        return find_best_segment(filepath, target_sec=target_sec, ignore_sec=ignore_sec)

    # długość utworu
    duration = librosa.get_duration(path=filepath)

    # drop w środku wycinka
    start = drop_time - target_sec / 1.5

    # nie wchodźmy w ignorowane intro
    start = max(start, ignore_sec)

    # nie wychodźmy poza koniec utworu
    if start + target_sec > duration:
        start = max(ignore_sec, duration - target_sec)

    # i na wszelki wypadek >= 0
    return float(max(0.0, start))
