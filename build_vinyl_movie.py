#!/usr/bin/env python3
import sys
import os
import json

import numpy as np
from PIL import Image as PILImage
from moviepy.editor import VideoFileClip, CompositeVideoClip, VideoClip

# ------------------- KONFIG -------------------

ANALYSIS_JSON_FILENAME = "analysis.json"
OUTPUT_FILENAME = "_vinyl_movie.mp4"

# Procent szerokości wideo, jaki ma zajmować okładka (0.0 - 1.0)
COVER_WIDTH_FRACTION = 0.5


# ------------------- FUNKCJE POMOCNICZE -------------------


def load_analysis_json(folder_path: str):
    json_path = os.path.join(folder_path, ANALYSIS_JSON_FILENAME)
    if not os.path.isfile(json_path):
        raise FileNotFoundError(f"Nie znaleziono pliku {ANALYSIS_JSON_FILENAME} w {folder_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data, json_path


def get_script_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def create_vinyl_video(template_video_path: str, cover_path: str, total_target_sec: float, output_path: str) -> str:
    if not os.path.isfile(template_video_path):
        raise FileNotFoundError(f"Nie znaleziono pliku wideo: {template_video_path}")

    if not os.path.isfile(cover_path):
        raise FileNotFoundError(f"Nie znaleziono pliku z okładką: {cover_path}")

    # --- BAZOWE WIDEO ---
    base_clip = VideoFileClip(template_video_path)

    # Przytnij do total_target_sec (jeśli krótsze, to bierz całość)
    duration = min(total_target_sec, base_clip.duration) if total_target_sec else base_clip.duration
    base_clip = base_clip.subclip(0, duration)

    # --- Wczytanie i skalowanie okładki w PIL ---
    pil_img = PILImage.open(cover_path).convert("RGB")
    orig_w, orig_h = pil_img.size

    # docelowa szerokość okładki
    new_width = int(base_clip.w * COVER_WIDTH_FRACTION)
    if new_width <= 0:
        raise ValueError("COVER_WIDTH_FRACTION dał szerokość <= 0, sprawdź konfigurację.")

    new_height = int(orig_h * (new_width / orig_w))

    pil_img = pil_img.resize((new_width, new_height), resample=PILImage.LANCZOS)

    base_cover_array = np.array(pil_img)
    cover_h, cover_w, _ = base_cover_array.shape

    # --- Maska kołowa ---
    # maska 1.0 w kole, 0.0 poza nim
    Y, X = np.ogrid[:cover_h, :cover_w]
    cx = cover_w / 2.0
    cy = cover_h / 2.0
    radius = min(cover_w, cover_h) / 2.0

    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    mask_array = (dist <= radius).astype("float32")  # (H, W), wartości 0..1

    def make_mask_frame(t):
        return mask_array

    mask_clip = VideoClip(make_mask_frame, ismask=True, duration=base_clip.duration)

    # --- Klip okładki z ręczną rotacją ---
    # jeden pełny obrót na czas trwania filmu
    rotation_speed_deg_per_sec = 360.0 / base_clip.duration

    def make_rotated_frame(t):
        angle = (rotation_speed_deg_per_sec * t) % 360.0
        img = PILImage.fromarray(base_cover_array)
        # minus żeby kręciło się "jak winyl" (w prawo)
        rotated = img.rotate(-angle, resample=PILImage.BICUBIC, expand=False)
        return np.array(rotated)

    cover_clip = VideoClip(make_rotated_frame, duration=base_clip.duration)
    cover_clip.size = (cover_w, cover_h)
    cover_clip = cover_clip.set_position(("center", "center"))
    cover_clip.mask = mask_clip  # okrągła maska

    # --- ZŁOŻENIE WIDEO ---
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
        print("Użycie: python build_vinyl_movie.py <folder_z_analysis_json>")
        sys.exit(1)

    folder_path = sys.argv[1]
    if not os.path.isdir(folder_path):
        print(f"Błąd: {folder_path} nie jest katalogiem")
        sys.exit(1)

    # Wczytaj analysis.json
    data, json_path = load_analysis_json(folder_path)

    total_target_sec = data.get("total_target_sec", None)

    tracks = data.get("tracks", [])
    if not tracks:
        print("Brak tracków w analysis.json")
        sys.exit(1)

    # Na razie operujemy na pierwszym tracku
    track = tracks[0]

    vinyl_art_path = track.get("vinyl_art")
    if not vinyl_art_path:
        print("Brak pola 'vinyl_art' w pierwszym tracku")
        sys.exit(1)

    # Ścieżka do vinyl.mp4 (w tym samym folderze co skrypt)
    script_dir = get_script_dir()
    template_video_path = os.path.join(script_dir, "vinyl.mp4")

    # Ścieżka wyjściowa (w folderze podanym jako parametr) — jako pełna ścieżka
    output_path = os.path.abspath(os.path.join(folder_path, OUTPUT_FILENAME))

    print(f"Tworzę wideo na podstawie {template_video_path}")
    print(f"Okładka: {vinyl_art_path}")
    print(f"Wynik: {output_path}")

    created_path = create_vinyl_video(
        template_video_path=template_video_path,
        cover_path=vinyl_art_path,
        total_target_sec=total_target_sec,
        output_path=output_path,
    )

    # Zapisz pełną ścieżkę w analysis.json jako 'vinyl_movie'
    track["vinyl_movie"] = created_path

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Zaktualizowano {json_path} polem 'vinyl_movie' = {created_path}")


if __name__ == "__main__":
    main()
