# best_folder_15s.py

import sys
import os
import librosa
import numpy as np
from pydub import AudioSegment

SUPPORTED_EXT = (".mp3", ".wav", ".flac", ".m4a", ".ogg")


def find_best_5_seconds(filepath, target_sec=5, ignore_sec=10):
    """
    Analyze the track and return the best 5-second start time (in seconds).
    Heuristic: mix of loudness (RMS) + rhythmic punch (onset strength).
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

    # we only need up to 3 tracks for 15 seconds total
    files = files[:3]

    print("Found files:")
    for f in files:
        print("  -", os.path.basename(f))

    segments = []
    fade_ms = 250  # 0.25 sec fade-in/out
    target_sec = 5

    for filepath in files:
        print(f"\nProcessing: {os.path.basename(filepath)}")
        start_sec = find_best_5_seconds(filepath, target_sec=target_sec, ignore_sec=10)
        print(f"  Best 5-second segment starts at: {start_sec:.2f}s")

        audio = AudioSegment.from_file(filepath)
        start_ms = int(start_sec * 1000)
        end_ms = start_ms + target_sec * 1000

        if end_ms > len(audio):
            # if near the end, shift window back
            end_ms = len(audio)
            start_ms = max(0, end_ms - target_sec * 1000)

        clip = audio[start_ms:end_ms]

        # apply fade in/out (non-destructive if track < 500ms)
        if len(clip) > 2 * fade_ms:
            clip = clip.fade_in(fade_ms).fade_out(fade_ms)
        else:
            # if extremely short, just fade what we can
            half = len(clip) // 2
            clip = clip.fade_in(half).fade_out(half)

        segments.append(clip)

    if not segments:
        print("No segments created.")
        sys.exit(0)

    # concatenate segments
    combined = segments[0]
    for seg in segments[1:]:
        combined += seg

    out_path = os.path.join(folder, "combined_best15.mp3")
    combined.export(out_path, format="mp3")

    total_sec = len(combined) / 1000.0
    print(f"\nCombined track length: {total_sec:.2f}s")
    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    main()
