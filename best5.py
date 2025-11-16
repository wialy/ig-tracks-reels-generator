# best_folder_15s.py

import sys
import os
import librosa
import numpy as np
from pydub import AudioSegment

SUPPORTED_EXT = (".mp3", ".wav", ".flac", ".m4a", ".ogg")

TOTAL_TARGET_SEC = 15.0  # final length always 15 seconds


def find_best_segment(filepath, target_sec=5.0, ignore_sec=10.0):
    """
    Analyze the track and return the best segment start time (in seconds),
    of length target_sec, using a mix of loudness (RMS) + rhythmic punch (onset strength).
    """
    y, sr = librosa.load(filepath, sr=44100, mono=True)

    # Frame settings (50 ms)
    frame_length = int(0.05 * sr)
    hop_length = frame_length  # no overlap for simplicity

    # RMS (loudness)
    rms = librosa.feature.rms(
        y=y,
        frame_length=frame_length,
        hop_length=hop_length
    )[0]

    # Onset strength (rhythmic punch)
    onset_env = librosa.onset.onset_strength(
        y=y, sr=sr, hop_length=hop_length
    )

    L = min(len(rms), len(onset_env))
    rms = rms[:L]
    onset_env = onset_env[:L]

    # convert target_sec -> frames
    frames_per_sec = sr / hop_length
    win_frames = int(target_sec * frames_per_sec)
    if win_frames <= 0:
        return 0.0

    # ignore intro/outro
    ignore_frames = int(ignore_sec * frames_per_sec)
    start = ignore_frames
    end = L - ignore_frames - win_frames

    if end <= start:
        # track too short or ignore window too big → search full range
        start = 0
        end = L - win_frames

    if end <= start:
        # still too short, fallback to start
        return 0.0

    def norm(x: np.ndarray) -> np.ndarray:
        mn = x.min()
        mx = x.max()
        if mx == mn:
            return x
        return (x - mn) / (mx - mn)

    rms_n = norm(rms)
    onset_n = norm(onset_env)

    alpha = 0.7  # loudness weight
    beta = 0.3   # punch weight

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

    # convert best frame to seconds
    best_start_sec = best_frame * hop_length / sr
    return float(best_start_sec)


def main():
    if len(sys.argv) < 2:
        print("Usage: python best_folder_15s.py <folder_path>")
        sys.exit(1)

    folder = sys.argv[1]

    if not os.path.isdir(folder):
        print(f"Error: {folder} is not a directory")
        sys.exit(1)

    # collect audio files
    files = [
        os.path.join(folder, f)
        for f in sorted(os.listdir(folder))
        if f.lower().endswith(SUPPORTED_EXT)
    ]

    if not files:
        print("No supported audio files found in folder.")
        sys.exit(0)

    num_tracks = len(files)
    print(f"Found {num_tracks} file(s):")
    for f in files:
        print("  -", os.path.basename(f))

    # dynamic segment duration per track, total always 15s
    per_segment_sec = TOTAL_TARGET_SEC / num_tracks
    per_segment_ms = int(per_segment_sec * 1000)

    print(f"\nTotal target length: {TOTAL_TARGET_SEC:.2f}s")
    print(f"Per-track segment:   {per_segment_sec:.2f}s")

    segments = []

    # fade: up to 250ms, but at most 1/4 of the segment
    fade_ms = min(250, per_segment_ms // 4)

    for filepath in files:
        print(f"\nProcessing: {os.path.basename(filepath)}")
        start_sec = find_best_segment(
            filepath,
            target_sec=per_segment_sec,
            ignore_sec=10.0
        )
        print(f"  Best segment start: {start_sec:.2f}s (len ~ {per_segment_sec:.2f}s)")

        audio = AudioSegment.from_file(filepath)
        start_ms = int(start_sec * 1000)
        end_ms = start_ms + per_segment_ms

        # if near the end, shift window back
        if end_ms > len(audio):
            end_ms = len(audio)
            start_ms = max(0, end_ms - per_segment_ms)

        clip = audio[start_ms:end_ms]

        # if still not exactly per_segment_ms (e.g., very short file),
        # we can pad with silence to keep lengths consistent
        if len(clip) < per_segment_ms:
            pad = AudioSegment.silent(duration=per_segment_ms - len(clip))
            clip += pad
        elif len(clip) > per_segment_ms:
            clip = clip[:per_segment_ms]

        # apply fade in/out
        if len(clip) > 2 * fade_ms and fade_ms > 0:
            clip = clip.fade_in(fade_ms).fade_out(fade_ms)
        elif fade_ms > 0 and len(clip) > 0:
            half = len(clip) // 2
            clip = clip.fade_in(half).fade_out(half)

        segments.append(clip)

    if not segments:
        print("No segments created.")
        sys.exit(0)

    combined = segments[0]
    for seg in segments[1:]:
        combined += seg

    # enforce exact total length = TOTAL_TARGET_SEC
    target_ms = int(TOTAL_TARGET_SEC * 1000)
    if len(combined) < target_ms:
        pad = AudioSegment.silent(duration=target_ms - len(combined))
        combined += pad
    elif len(combined) > target_ms:
        combined = combined[:target_ms]

    out_path = os.path.join(folder, "combined_best15.mp3")
    combined.export(out_path, format="mp3")

    total_sec = len(combined) / 1000.0
    print(f"\nFinal combined length: {total_sec:.2f}s")
    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    main()
