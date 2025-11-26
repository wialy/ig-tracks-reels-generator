#!/usr/bin/env python3
"""
analyze_track_character.py

Analiza:
- Punkt 2: cechy muzyczno-produkcyjne (energia, bas, jasność/timbre, perkusja, groove)
- Punkt 3: flow / energia w czasie, dramaturgia, "emocjonalny charakter"

Użycie:
    python analyze_track_character.py /sciezka/do/utworu.wav
"""

import sys
import json
from typing import Dict, Any, List, Tuple

import numpy as np
import librosa


# ---------------------------------------------------------
# Pomocnicze
# ---------------------------------------------------------

def _smooth(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return x
    kernel = np.ones(win) / win
    return np.convolve(x, kernel, mode="same")


def _norm01(x: np.ndarray) -> np.ndarray:
    mn = np.min(x)
    mx = np.max(x)
    if mx - mn < 1e-9:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn)


def _zscore(x: np.ndarray) -> np.ndarray:
    m = np.mean(x)
    s = np.std(x) + 1e-9
    return (x - m) / s


# ---------------------------------------------------------
# Analiza punkt 2 – cechy muzyczno-produkcyjne
# ---------------------------------------------------------

def analyze_technical_features(y: np.ndarray, sr: int) -> Dict[str, Any]:
    """
    Zwraca:
    - bpm
    - energy_mean (0-1)
    - energy_variation (0-1)
    - low_end_intensity (0-1)
    - brightness (0-1)
    - percussive_density (0-1)
    - groove_swing (0-1)
    """

    # RMS / energia
    hop_length = 1024
    frame_length = 4096
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    rms_smooth = _smooth(rms, 8)
    energy_curve = _norm01(rms_smooth)

    energy_mean = float(np.mean(energy_curve))
    energy_var = float(np.var(energy_curve))
    # przeskaluj wariancję do 0-1 (heurystycznie)
    energy_variation = float(np.tanh(3.0 * energy_var))

    # BPM
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    bpm = float(tempo)

    # Jasność (spectral centroid)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
    centroid_norm = _norm01(centroid)
    brightness = float(np.mean(centroid_norm))

    # HPSS – percussive vs harmonic
    harmonic, percussive = librosa.effects.hpss(y)
    # percussive RMS vs whole
    perc_rms = librosa.feature.rms(y=percussive, frame_length=frame_length, hop_length=hop_length)[0]
    perc_rms_norm = _norm01(perc_rms)
    percussive_density = float(np.mean(perc_rms_norm))

    # Bass – share of energy in the low band (e.g. < 150 Hz)
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop_length))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    low_band = S[(freqs >= 20) & (freqs <= 150), :]
    full_energy = np.sum(S ** 2)
    low_energy = np.sum(low_band ** 2)
    if full_energy < 1e-9:
        low_end_intensity = 0.0
    else:
        # slightly boost the scale -> tanh
        low_end_intensity = float(np.tanh(3.0 * (low_energy / full_energy)))

    # Groove / swing – look at the variability of intervals between onsets
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, hop_length=hop_length)
    groove_swing = 0.0
    if len(onset_frames) > 4:
        onsets_time = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)
        ioi = np.diff(onsets_time)
        if len(ioi) > 1:
            # normalized variability between intervals
            ioi_norm = ioi / (np.mean(ioi) + 1e-9)
            groove_swing_raw = float(np.std(ioi_norm))
            # scale to 0-1
            groove_swing = float(np.tanh(2.0 * groove_swing_raw))

    return {
        "bpm": bpm,
        "energy_mean": energy_mean,
        "energy_variation": energy_variation,
        "low_end_intensity": low_end_intensity,
        "brightness": brightness,
        "percussive_density": percussive_density,
        "groove_swing": groove_swing,
    }


# ---------------------------------------------------------
# Analiza punkt 3 – flow, dramaturgia, emocja
# ---------------------------------------------------------

def compute_energy_curve(y: np.ndarray, sr: int, hop_length: int = 2048) -> Tuple[np.ndarray, np.ndarray]:
    """Zwraca (times, energy_curve_norm)."""
    frame_length = 4096
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    rms_smooth = _smooth(rms, 16)
    energy_curve = _norm01(rms_smooth)
    times = librosa.frames_to_time(np.arange(len(energy_curve)), sr=sr, hop_length=hop_length)
    return times, energy_curve


def segment_dramaturgy(times: np.ndarray, energy_curve: np.ndarray) -> List[Dict[str, Any]]:
    """
    Bardzo prosta heurystyka:
    - intro: początkowy odcinek o niskiej energii
    - outro: końcowy odcinek o niskiej energii
    - lokalne maxima = "peaki" (dropy / kulminacje)
    - fragmenty przed peakami = build-up
    - fragmenty między peakami o niższej energii = breakdown
    Zwraca listę segmentów z labelami.
    """
    duration = float(times[-1]) if len(times) > 0 else 0.0
    if duration == 0:
        return []

    # wykrywanie lokalnych maksimów energii
    e = energy_curve
    # proste lokalne maxima
    peaks = []
    for i in range(1, len(e) - 1):
        if e[i] > e[i - 1] and e[i] > e[i + 1]:
            peaks.append(i)
    peaks = np.array(peaks, dtype=int)

    # filtr – tylko "wysokie" peaki
    if len(peaks) > 0:
        thr = np.quantile(e, 0.75)
        peaks = peaks[e[peaks] >= thr]

    segments: List[Dict[str, Any]] = []

    # intro – szukamy pierwszego momentu, gdzie energia wyraźniej rośnie
    intro_end_idx = 0
    if len(e) > 0:
        # średnia z pierwszych 10% utworu jako "tło"
        n10 = max(1, int(0.1 * len(e)))
        base = np.mean(e[:n10])
        for i in range(n10, len(e)):
            # jeżeli energia stabilnie powyżej tła
            if e[i] > base + 0.15:
                intro_end_idx = i
                break
        else:
            intro_end_idx = n10

    intro_start_t = float(times[0])
    intro_end_t = float(times[intro_end_idx])
    if intro_end_t - intro_start_t > 5.0:
        segments.append({
            "type": "intro",
            "start": intro_start_t,
            "end": intro_end_t,
        })

    # jeśli nie ma peaków, robimy tylko intro / body / outro
    if len(peaks) == 0:
        body_start = intro_end_t
        body_end = duration
        if body_end - body_start > 5.0:
            segments.append({
                "type": "body",
                "start": body_start,
                "end": body_end,
            })
        return segments

    # Sortujemy peaki po czasie
    peaks = np.sort(peaks)

    # Wyznaczamy outro jako końcowy odcinek o niskiej energii
    outro_start_idx = len(e) - 1
    n90 = max(1, int(0.9 * len(e)))
    tail_base = np.mean(e[n90:])
    for i in range(len(e) - 1, n90, -1):
        if e[i] > tail_base + 0.1:
            outro_start_idx = i
            break

    outro_start_t = float(times[outro_start_idx])
    if duration - outro_start_t > 5.0:
        # wcześniej jeszcze wstawimy segmenty główne
        main_region_start = intro_end_t
        main_region_end = outro_start_t

        # budujemy prostą strukturę build-up -> peak -> breakdown wokół każdego peaku
        last_end = main_region_start
        for p in peaks:
            peak_t = float(times[p])

            # pomijamy peaki w intro/outro
            if peak_t <= intro_end_t or peak_t >= outro_start_t:
                continue

            # build-up: od poprzedniego "end" do chwili tuż przed peakiem
            build_start = last_end
            build_end = max(build_start, peak_t - 4.0)  # zostaw 4 sekundy na sam peak

            if build_end - build_start > 3.0:
                segments.append({
                    "type": "build_up",
                    "start": build_start,
                    "end": build_end,
                })

            # peak
            peak_end = min(peak_t + 6.0, main_region_end)
            segments.append({
                "type": "peak",
                "start": peak_t,
                "end": peak_end,
            })

            # breakdown: od końca peaku do następnego build-up
            last_end = peak_end

        # cokolwiek zostało między ostatnim segmentem a outro
        if main_region_end - last_end > 3.0:
            segments.append({
                "type": "breakdown",
                "start": last_end,
                "end": main_region_end,
            })

        # outro
        segments.append({
            "type": "outro",
            "start": outro_start_t,
            "end": duration,
        })
    else:
        # bez wyraźnego outro – po prostu segmenty peakowe
        last_end = intro_end_t
        for p in peaks:
            peak_t = float(times[p])
            if peak_t <= intro_end_t:
                continue
            build_start = last_end
            build_end = max(build_start, peak_t - 4.0)
            if build_end - build_start > 3.0:
                segments.append({
                    "type": "build_up",
                    "start": build_start,
                    "end": build_end,
                })
            peak_end = min(peak_t + 6.0, duration)
            segments.append({
                "type": "peak",
                "start": peak_t,
                "end": peak_end,
            })
            last_end = peak_end

    return segments


def classify_emotional_character(
    bpm: float,
    brightness: float,
    energy_mean: float,
    energy_variation: float,
    low_end_intensity: float,
) -> str:
    """
    Bardzo uproszczona heurystyka:
    - dark / driving
    - uplifting
    - melancholic
    - neutral
    """
    fast = bpm >= 126
    slow = bpm <= 120

    bright = brightness > 0.55
    darkish = brightness < 0.45

    high_energy = energy_mean > 0.6
    low_energy = energy_mean < 0.4

    high_var = energy_variation > 0.15
    strong_low = low_end_intensity > 0.6

    # dark / driving
    if fast and darkish and strong_low and high_energy:
        return "dark_driving"

    # uplifting
    if bright and high_energy and not darkish:
        return "uplifting"

    # melancholic – raczej ciemniejszy i średnio-wolny, umiarkowana energia
    if darkish and not fast and not high_energy and not low_energy:
        return "melancholic"

    # deep / dubby
    if darkish and strong_low and low_energy:
        return "deep_dubby"

    # fallback
    return "neutral"


def summarize_flow(energy_curve: np.ndarray) -> Dict[str, Any]:
    """
    Prosty opis przebiegu energii:
    - rośnie / spada / łuk / falowanie itd.
    """
    if len(energy_curve) == 0:
        return {
            "shape": "unknown",
            "trend": 0.0,
        }

    # trend globalny
    x = np.arange(len(energy_curve))
    y = energy_curve
    # regresja liniowa y ~ ax + b
    a, b = np.polyfit(x, y, 1)
    trend = float(a)

    # porównanie energii w 3 tercylach
    n = len(y)
    thirds = np.array_split(y, 3)
    means = [float(np.mean(t)) for t in thirds]

    shape = "flat"
    if means[0] < means[1] > means[2]:
        shape = "arc_peak_middle"
    elif means[0] < means[1] < means[2]:
        shape = "rising"
    elif means[0] > means[1] > means[2]:
        shape = "falling"
    elif means[0] > means[1] < means[2]:
        shape = "valley"

    return {
        "shape": shape,
        "trend": trend,
        "energy_terciles": means,
    }


# ---------------------------------------------------------
# Główna funkcja analizy jednego pliku
# ---------------------------------------------------------

def analyze_track(path: str) -> Dict[str, Any]:
    y, sr = librosa.load(path, sr=None, mono=True)
    duration = float(librosa.get_duration(y=y, sr=sr))

    tech = analyze_technical_features(y, sr)
    times, energy_curve = compute_energy_curve(y, sr)
    segments = segment_dramaturgy(times, energy_curve)
    flow = summarize_flow(energy_curve)

    emotional = classify_emotional_character(
        bpm=tech["bpm"],
        brightness=tech["brightness"],
        energy_mean=tech["energy_mean"],
        energy_variation=tech["energy_variation"],
        low_end_intensity=tech["low_end_intensity"],
    )

    return {
        "file": path,
        "duration_sec": duration,
        "technical": tech,
        "flow": {
            "energy_curve_len": len(energy_curve),
            "summary": flow,
            "segments": segments,
        },
        "emotional_character": emotional,
    }


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print("Użycie: python analyze_track_character.py /sciezka/do/utworu.wav")
        sys.exit(1)

    path = sys.argv[1]
    result = analyze_track(path)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
