#!/usr/bin/env python3
import sys
import os
import json

import numpy as np
from PIL import Image as PILImage
from moviepy.editor import VideoFileClip, CompositeVideoClip, VideoClip

# ------------------- CONFIG -------------------

ANALYSIS_JSON_FILENAME = "analysis.json"
OUTPUT_FILENAME = "_vinyl_movie.mp4"

# Percentage of video width that the cover should occupy (0.0 - 1.0)
COVER_WIDTH_FRACTION = 0.5


# ------------------- HELPER FUNCTIONS -------------------


def load_analysis_json(folder_path: str):
    json_path = os.path.join(folder_path, ANALYSIS_JSON_FILENAME)
    if not os.path.isfile(json_path):
        raise FileNotFoundError(f"File {ANALYSIS_JSON_FILENAME} not found in {folder_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data, json_path


def get_script_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def create_vinyl_video(template_video_path: str, cover_path: str, total_target_sec: float, output_path: str) -> str:

    if not os.path.isfile(template_video_path):
        raise FileNotFoundError(f"Video file not found: {template_video_path}")

    if not os.path.isfile(cover_path):
        raise FileNotFoundError(f"Cover file not found: {cover_path}")

    # --- BASE VIDEO ---
    base_clip = VideoFileClip(template_video_path)

    # Trim to total_target_sec (if shorter, take the whole video)
    duration = min(total_target_sec, base_clip.duration) if total_target_sec else base_clip.duration
    base_clip = base_clip.subclip(0, duration)

    # --- Load and scale cover in PIL ---
    pil_img = PILImage.open(cover_path).convert("RGB")
    orig_w, orig_h = pil_img.size

    # target cover width
    new_width = int(base_clip.w * COVER_WIDTH_FRACTION)
    if new_width <= 0:
        raise ValueError("COVER_WIDTH_FRACTION resulted in width <= 0, check configuration.")

    new_height = int(orig_h * (new_width / orig_w))

    pil_img = pil_img.resize((new_width, new_height), resample=PILImage.LANCZOS)

    base_cover_array = np.array(pil_img)
    cover_h, cover_w, _ = base_cover_array.shape

    # --- Circular mask ---
    # mask 1.0 inside the circle, 0.0 outside
    Y, X = np.ogrid[:cover_h, :cover_w]
    cx = cover_w / 2.0
    cy = cover_h / 2.0
    radius = min(cover_w, cover_h) / 2.0

    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    mask_array = (dist <= radius).astype("float32")  # (H, W), values 0..1

    def make_mask_frame(t):
        return mask_array

    mask_clip = VideoClip(make_mask_frame, ismask=True, duration=base_clip.duration)

    # --- Cover clip with manual rotation ---
    # one full rotation for the duration of the video
    rotation_speed_deg_per_sec = 360.0 / base_clip.duration

    def make_rotated_frame(t):
        angle = (rotation_speed_deg_per_sec * t) % 360.0
        img = PILImage.fromarray(base_cover_array)
        # minus to rotate "like a vinyl" (to the right)
        rotated = img.rotate(-angle, resample=PILImage.BICUBIC, expand=False)
        return np.array(rotated)

    cover_clip = VideoClip(make_rotated_frame, duration=base_clip.duration)
    cover_clip.size = (cover_w, cover_h)
    cover_clip = cover_clip.set_position(("center", "center"))
    cover_clip.mask = mask_clip  # circular mask

    # --- COMPOSITE VIDEO ---
    final_clip = CompositeVideoClip([base_clip, cover_clip])
    final_clip = final_clip.set_audio(base_clip.audio)

    final_clip.write_videofile(
        output_path,
        codec="libx264",
        audio_codec="aac",
        fps=base_clip.fps,
    )

    base_clip.close()
    cover_clip.close()
    mask_clip.close()
    final_clip.close()

    return output_path


def main():
    if len(sys.argv) < 2:
        print("Usage: python build_vinyl_movie.py <folder_with_analysis_json>")
        sys.exit(1)

    folder_path = sys.argv[1]
    if not os.path.isdir(folder_path):
        print(f"Error: {folder_path} is not a directory")
        sys.exit(1)

    # Load analysis.json
    data, json_path = load_analysis_json(folder_path)

    total_target_sec = data.get("total_target_sec", None)

    tracks = data.get("tracks", [])
    if not tracks:
        print("No tracks in analysis.json")
        sys.exit(1)

    # For now, operate on the first track
    track = tracks[0]

    vinyl_art_path = track.get("vinyl_art")
    if not vinyl_art_path:
        print("Missing 'vinyl_art' field in the first track")
        sys.exit(1)

    # Path to vinyl.mp4 (in the same folder as the script)
    script_dir = get_script_dir()
    template_video_path = os.path.join(script_dir, "vinyl.mp4")

    # Output path (in the folder provided as parameter) — as full path
    output_path = os.path.abspath(os.path.join(folder_path, OUTPUT_FILENAME))

    print(f"Creating video based on {template_video_path}")
    print(f"Cover: {vinyl_art_path}")
    print(f"Result: {output_path}")

    created_path = create_vinyl_video(
        template_video_path=template_video_path,
        cover_path=vinyl_art_path,
        total_target_sec=total_target_sec,
        output_path=output_path,
    )

    # Save full path in analysis.json as 'vinyl_movie'
    track["vinyl_movie"] = created_path

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Updated {json_path} with field 'vinyl_movie' = {created_path}")


if __name__ == "__main__":
    main()
