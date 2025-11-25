#!/usr/bin/env python3
# build_vinyl_art.py

import sys
import os
import math
import json
import numpy as np
from PIL import Image, ImageFilter


# ============================================================
#  GENERATOR VINYLOWEGO KOŁA
# ============================================================

def image_inside_smart_circle(
    input_path: str,
    output_path: str,
    margin_factor: float = 1.0,
    blur_factor: float = 0.06,
    blend_factor: float = 0.35,
    wobble_factor: float = 0.03,
):
    """
    Generuje okrągły "vinyl art":
      - obrazek w środku (nie przycięty),
      - koło o promieniu ~ a/sqrt(2),
      - krawędź wypełniona ekstrudowaną, lekko pofalowaną teksturą krawędzi,
      - dalsza część gładko przechodzi w rozmyte tło.
    """

    img = Image.open(input_path).convert("RGBA")
    w, h = img.size
    short_side = min(w, h)

    # promień = a / sqrt(2)
    base_radius = (short_side / math.sqrt(2.0)) * margin_factor
    radius = max(base_radius, short_side / 2.0 + 1.0)

    diameter = int(math.ceil(radius * 2.0))
    diameter = max(diameter, max(w, h) + 2)

    radius = diameter / 2.0
    cx = cy = radius

    # rozmyte tło
    bg = img.resize((diameter, diameter), Image.LANCZOS)
    bg = bg.filter(ImageFilter.GaussianBlur(radius=diameter * blur_factor))
    bg_arr = np.array(bg, dtype=np.float32)

    # siatka współrzędnych
    yy, xx = np.indices((diameter, diameter))
    dist_center_sq = (xx - cx) ** 2 + (yy - cy) ** 2
    circle_mask = dist_center_sq <= radius**2

    # pozycja oryginalnego obrazu
    left = (diameter - w) // 2
    top = (diameter - h) // 2
    right = left + w
    bottom = top + h

    rect_mask = (xx >= left) & (xx < right) & (yy >= top) & (yy < bottom)
    img_arr = np.array(img, dtype=np.float32)

    # wynik = rozmyte tło w kole
    result = np.zeros_like(bg_arr, dtype=np.float32)
    result[circle_mask] = bg_arr[circle_mask]

    # obszar pomiędzy rect a kołem
    border_mask = circle_mask & (~rect_mask)

    if np.any(border_mask):

        # najbliższy punkt na prostokącie
        nearest_x = np.clip(xx, left, right - 1)
        nearest_y = np.clip(yy, top, bottom - 1)

        dist_rect = np.sqrt((xx - nearest_x)**2 + (yy - nearest_y)**2)

        max_blend_dist = radius * blend_factor
        max_blend_dist = max(max_blend_dist, 1.0)

        t_scalar = np.clip(dist_rect / max_blend_dist, 0.0, 1.0)
        t = t_scalar[..., np.newaxis]

        # falowanie
        wobble_max = short_side * wobble_factor
        rng = np.random.default_rng()
        noise_x = rng.standard_normal((diameter, diameter))
        noise_y = rng.standard_normal((diameter, diameter))

        amp = (np.clip(dist_rect / max_blend_dist, 0.0, 1.0)) * wobble_max

        sample_x = nearest_x + noise_x * amp
        sample_y = nearest_y + noise_y * amp

        sample_x = np.clip(sample_x, left, right - 1)
        sample_y = np.clip(sample_y, top, bottom - 1)

        local_x = (sample_x - left).astype(int)
        local_y = (sample_y - top).astype(int)

        edge_colors = img_arr[local_y, local_x]

        blended = edge_colors * (1 - t) + bg_arr * t
        result[border_mask] = blended[border_mask]

    # środek – oryginalna okładka
    result[top:bottom, left:right] = img_arr

    # poza kołem alpha = 0
    outside = ~circle_mask
    result[outside, 3] = 0

    out_img = Image.fromarray(np.clip(result, 0, 255).astype(np.uint8), mode="RGBA")
    out_img.save(output_path, "PNG")


# ============================================================
#  GŁÓWNA LOGIKA JSON
# ============================================================

def process_folder(folder_path: str):

    json_path = os.path.join(folder_path, "analysis.json")

    if not os.path.isfile(json_path):
        print(f"ERROR: analysis.json not found in folder: {folder_path}")
        sys.exit(1)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    tracks = data.get("tracks", [])
    if not isinstance(tracks, list):
        print("ERROR: analysis.json has no 'tracks' list.")
        sys.exit(1)

    for i, track in enumerate(tracks):
        cover_path = track.get("cover_path")

        if not cover_path:
            print(f"[WARN] Track {i}: missing cover_path")
            continue

        if not os.path.isfile(cover_path):
            print(f"[WARN] Track {i}: cover file does not exist: {cover_path}")
            continue

        base, _ext = os.path.splitext(cover_path)
        out_path = base + "_vinyl.png"

        print(f"[INFO] Generating vinyl art: {cover_path} → {out_path}")

        image_inside_smart_circle(
            input_path=cover_path,
            output_path=out_path,
            margin_factor=1.0,   # kwadrat wpisany w koło
            blur_factor=0.06,
            blend_factor=0.35,
            wobble_factor=0.03,
        )

        track["vinyl_art"] = out_path

    # zapis JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"[OK] Updated: {json_path}")


# ============================================================
#  ENTRY POINT
# ============================================================

def main():
    if len(sys.argv) < 2:
        print("Usage: python build_vinyl_art.py <folder_with_json>")
        sys.exit(1)

    folder = sys.argv[1]
    process_folder(folder)


if __name__ == "__main__":
    main()
