#!/usr/bin/env python3
"""
waveform_video.py

Generate a video with a vertically centered RGB waveform and a moving
progress line that scales nearby bars.

- Waveform consists of bars (BAR_WIDTH px width, BAR_GAP px gap).
- Each bar is composed of 3 overlapping channels: low/mid/high (RGB).
- Background color is (22, 22, 29, 220) [alpha ignored, we use RGB].
- Video duration matches the audio duration.
- Video has NO audio.
"""

import argparse
import os

import numpy as np
import librosa
from moviepy.editor import VideoClip


# ------------------------- CONSTANTS ---------------------------------

# Visual layout
BAR_WIDTH = 3                      # width of each bar in pixels
BAR_GAP = 1                        # gap between bars in pixels
BG_COLOR = (22, 22, 29, 220)       # RGBA-like, but we only use RGB
LINE_COLOR = (255, 255, 255)       # RGB color for progress line
LINE_WIDTH = 1                     # width of the vertical line

FPS = 30                           # frames per second

# Highlighting around the progress line
HIGHLIGHT_MAX_SCALE = 1.2          # scale factor directly under the progress line
HIGHLIGHT_RADIUS_BARS = 3          # how many bars on each side get scaled (linear falloff)

# Audio analysis
TARGET_SR = 44100                  # target sample rate for analysis
N_FFT = 2048                       # FFT window size
HOP_LENGTH = 1024                  # hop length between STFT frames

# Frequency bands in Hz: low / mid / high
BASS_MAX_HZ = 200.0
MID_MAX_HZ = 2000.0
HIGH_MAX_HZ = 8000.0               # up to this; above is ignored


# --------------------- AUDIO ANALYSIS --------------------------------

def analyze_audio_bands(audio_path: str, n_bars: int):
    """
    Analyze the audio and return:
      - amplitudes: (n_bars, 3) array of normalized band intensities [low, mid, high]
      - duration: audio duration in seconds

    Each bar corresponds to a chunk of time in the track; within that chunk we
    average energy in low/mid/high frequency bands.
    """
    # Load audio (mono)
    y, sr = librosa.load(audio_path, sr=TARGET_SR, mono=True)
    duration = len(y) / float(sr)

    # STFT magnitude
    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
    n_frames = S.shape[1]

    # Frequency band masks
    bass_mask = freqs <= BASS_MAX_HZ
    mid_mask = (freqs > BASS_MAX_HZ) & (freqs <= MID_MAX_HZ)
    high_mask = (freqs > MID_MAX_HZ) & (freqs <= HIGH_MAX_HZ)

    # Avoid empty masks (edge cases)
    if not bass_mask.any():
        bass_mask[0] = True
    if not mid_mask.any():
        mid_mask[bass_mask.argmax()] = True
    if not high_mask.any():
        high_mask[mid_mask.argmax()] = True

    amplitudes = np.zeros((n_bars, 3), dtype=np.float32)

    for i in range(n_bars):
        start_frame = int(i * n_frames / n_bars)
        end_frame = int((i + 1) * n_frames / n_bars)
        if end_frame <= start_frame:
            end_frame = min(start_frame + 1, n_frames)

        chunk = S[:, start_frame:end_frame]

        # Mean magnitude in each band
        low_energy = chunk[bass_mask].mean() if chunk[bass_mask].size > 0 else 0.0
        mid_energy = chunk[mid_mask].mean() if chunk[mid_mask].size > 0 else 0.0
        high_energy = chunk[high_mask].mean() if chunk[high_mask].size > 0 else 0.0

        amplitudes[i, 0] = low_energy
        amplitudes[i, 1] = mid_energy
        amplitudes[i, 2] = high_energy

    # Normalize each band independently to [0, 1] (controls HEIGHT only)
    for b in range(3):
        band_max = amplitudes[:, b].max()
        if band_max > 0:
            amplitudes[:, b] /= band_max

    return amplitudes, duration


# ---------------------- FRAME RENDERING -------------------------------

def make_waveform_frame_generator(
    amplitudes: np.ndarray,
    duration: float,
    video_width: int,
    video_height: int,
):
    """
    Returns a function f(t) suitable for MoviePy VideoClip(make_frame=f).
    - amplitudes: (n_bars, 3) normalized band intensities.
    - duration: video duration in seconds.

    IMPORTANT: amplitude ONLY affects bar height.
               Color is always full-intensity R/G/B.
    """
    n_bars = amplitudes.shape[0]

    # Geometry
    total_bar_width = n_bars * (BAR_WIDTH + BAR_GAP)
    if total_bar_width > video_width:
        # If bars don't fit, scale down bar width+gap proportionally.
        scale = video_width / total_bar_width
    else:
        scale = 1.0

    scaled_bar_width = max(1, int(round(BAR_WIDTH * scale)))
    scaled_bar_gap = max(1, int(round(BAR_GAP * scale)))
    bar_stride = scaled_bar_width + scaled_bar_gap

    # Actual number of bars that fit (should be n_bars by design, but be safe)
    max_bars_fit = min(n_bars, video_width // bar_stride)

    mid_y = video_height / 2.0
    waveform_height = video_height * 0.8  # use 80% of height for waveform

    max_highlight_distance_px = HIGHLIGHT_RADIUS_BARS * bar_stride

    bg_rgb = np.array(BG_COLOR[:3], dtype=np.uint8)

    def make_frame(t: float) -> np.ndarray:
        # Background RGB frame
        frame = np.zeros((video_height, video_width, 3), dtype=np.uint8)
        frame[:, :] = bg_rgb

        # Current progress (0..1)
        progress = t / duration if duration > 0 else 0.0
        progress = max(0.0, min(1.0, progress))
        progress_x = progress * (video_width - 1)

        # Draw waveform bars
        for i in range(max_bars_fit):
            # x position for this bar
            x = i * bar_stride
            bar_center_x = x + scaled_bar_width / 2.0

            # Highlight scaling factor based on distance to progress line
            distance_px = abs(bar_center_x - progress_x)
            if distance_px <= max_highlight_distance_px and max_highlight_distance_px > 0:
                # Linear falloff from HIGHLIGHT_MAX_SCALE at center to 1.0 at edge
                frac = 1.0 - (distance_px / max_highlight_distance_px)
                scale_factor = 1.0 + (HIGHLIGHT_MAX_SCALE - 1.0) * frac
            else:
                scale_factor = 1.0

            # Channel amplitudes for this bar: [low, mid, high]
            low_amp, mid_amp, high_amp = amplitudes[i]

            # Separate additive bars with constant RGB colors.
            # Amplitude ONLY affects height.
            channel_data = [
                (low_amp,  np.array([255, 0,   0], dtype=np.uint8)),  # Red  = low
                (mid_amp,  np.array([0,   255, 0], dtype=np.uint8)),  # Green = mid
                (high_amp, np.array([0,   0, 255], dtype=np.uint8)),  # Blue  = high
            ]

            for amp, base_color in channel_data:
                if amp <= 0.0:
                    continue

                # Height of this channel bar
                bar_h = max(1.0, amp * waveform_height * scale_factor)

                top = int(round(mid_y - bar_h / 2.0))
                bottom = int(round(mid_y + bar_h / 2.0))

                top = max(0, top)
                bottom = min(video_height, bottom)
                if bottom <= top:
                    continue

                # Constant, full-intensity color for this channel
                color = base_color

                # Slice for this bar
                x_start = x
                x_end = min(video_width, x + scaled_bar_width)
                if x_end <= x_start:
                    continue

                region = frame[top:bottom, x_start:x_end, :]
                # Additive blending so overlaps mix colors
                region = np.clip(region.astype(np.int16) + color, 0, 255).astype(np.uint8)
                frame[top:bottom, x_start:x_end, :] = region

        # Draw vertical progress line on top
        line_x = int(round(progress_x))
        x0 = max(0, line_x - LINE_WIDTH // 2)
        x1 = min(video_width, line_x + (LINE_WIDTH - LINE_WIDTH // 2))
        if x1 > x0:
            frame[:, x0:x1, 0] = LINE_COLOR[0]
            frame[:, x0:x1, 1] = LINE_COLOR[1]
            frame[:, x0:x1, 2] = LINE_COLOR[2]

        return frame

    return make_frame


# --------------------------- MAIN ------------------------------------

def build_waveform_video(audio_path: str, width: int, height: int, output_path: str):
    """
    Build a waveform video:

    - Video size = (width, height)
    - Duration = audio duration
    - No audio in output
    """
    # Compute number of bars that fit horizontally
    total_bar_span = BAR_WIDTH + BAR_GAP
    n_bars = max(1, width // total_bar_span)

    amplitudes, duration = analyze_audio_bands(audio_path, n_bars)

    make_frame = make_waveform_frame_generator(
        amplitudes=amplitudes,
        duration=duration,
        video_width=width,
        video_height=height,
    )

    clip = VideoClip(make_frame, duration=duration)
    # Important: ensure no audio in the final video
    clip = clip.set_audio(None)

    clip.write_videofile(
        output_path,
        fps=FPS,
        codec="libx264",
        audio=False,
        threads=4,
        preset="medium",
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a waveform video (RGB bands + moving progress line) from an audio file."
    )
    parser.add_argument("audio_path", help="Path to the input audio file")
    parser.add_argument("width", type=int, help="Waveform/video width in pixels")
    parser.add_argument("height", type=int, help="Waveform/video height in pixels")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_waveform_video(
        audio_path=args.audio_path,
        width=args.width,
        height=args.height,
        output_path=os.path.abspath(args.audio_path).replace("_audio.mp3", "_waveform.mp4")
    )
