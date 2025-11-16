# best5.py
import sys
import librosa
import numpy as np
from pydub import AudioSegment

def find_best_5_seconds(filepath):
    # Load audio
    y, sr = librosa.load(filepath, sr=44100, mono=True)

    # Frame settings (50 ms)
    frame_length = int(0.05 * sr)
    hop_length = frame_length

    # Compute RMS (loudness)
    rms = librosa.feature.rms(
        y=y, 
        frame_length=frame_length, 
        hop_length=hop_length
    )[0]

    # Compute simple "onset strength" (rhythmic punch)
    onset_env = librosa.onset.onset_strength(
        y=y, sr=sr, hop_length=hop_length
    )

    L = min(len(rms), len(onset_env))
    rms = rms[:L]
    onset_env = onset_env[:L]

    # Convert 5 seconds → frames
    frames_per_sec = sr / hop_length
    win_frames = int(5 * frames_per_sec)

    # Avoid intros/outros
    ignore_sec = 10
    ignore_frames = int(ignore_sec * frames_per_sec)
    start = ignore_frames
    end = L - ignore_frames - win_frames
    if end <= start:
        start = 0
        end = L - win_frames

    # Normalize
    def norm(x):
        if x.max() == x.min():
            return x
        return (x - x.min()) / (x.max() - x.min())

    rms_n = norm(rms)
    onset_n = norm(onset_env)

    # Slide window
    alpha = 0.7
    beta = 0.3

    best_score = -1
    best_frame = 0

    for i in range(start, end):
        j = i + win_frames
        score = (
            alpha * rms_n[i:j].mean() +
            beta  * onset_n[i:j].mean()
        )
        if score > best_score:
            best_score = score
            best_frame = i

    start_sec = best_frame / frames_per_sec
    end_sec = start_sec + 5

    return start_sec, end_sec


def export_clip(filepath, start_sec, out_path):
    audio = AudioSegment.from_file(filepath)
    clip = audio[start_sec * 1000 : (start_sec + 5) * 1000]
    clip.export(out_path, format="mp3")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python best5.py <track.mp3>")
        sys.exit(1)

    filepath = sys.argv[1]
    start_sec, end_sec = find_best_5_seconds(filepath)

    print(f"Best 5 seconds: {start_sec:.2f}s – {end_sec:.2f}s")

    out_path = filepath.replace(".mp3", "_best5.mp3")
    export_clip(filepath, start_sec, out_path)

    print(f"Saved to: {out_path}")
