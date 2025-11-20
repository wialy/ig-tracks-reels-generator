# build_audio.py
import sys
import os
import json

from pydub import AudioSegment

from media_utils import TOTAL_TARGET_SEC


def main():
    if len(sys.argv) < 2:
        print("Usage: python build_audio.py <folder_path>")
        sys.exit(1)

    folder = sys.argv[1]

    analysis_path = os.path.join(folder, "analysis.json")
    if not os.path.isfile(analysis_path):
        print(f"Error: {analysis_path} not found. Run analyze_tracks.py first.")
        sys.exit(1)

    with open(analysis_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    tracks = data.get("tracks", [])
    if not tracks:
        print("No tracks in analysis.json.")
        sys.exit(0)

    print(f"Building combined audio from {len(tracks)} track(s).")

    segments = []
    for t in tracks:
        path = t["file_path"]
        basename = os.path.basename(path)
        start_sec = float(t["segment_start_sec"])
        dur_sec = float(t["segment_duration_sec"])
        start_ms = int(start_sec * 1000)
        dur_ms = int(dur_sec * 1000)

        print(f"  {basename}: start={start_sec:.2f}s, dur={dur_sec:.2f}s")

        audio_full = AudioSegment.from_file(path)
        end_ms = start_ms + dur_ms
        if end_ms > len(audio_full):
            end_ms = len(audio_full)
            start_ms = max(0, end_ms - dur_ms)

        clip = audio_full[start_ms:end_ms]

        # pad / trim
        if len(clip) < dur_ms:
            pad = AudioSegment.silent(duration=dur_ms - len(clip))
            clip += pad
        elif len(clip) > dur_ms:
            clip = clip[:dur_ms]

        # fades (0.25s or <= 1/4 duration)
        fade_ms = min(500, dur_ms // 4)
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

    # enforce TOTAL_TARGET_SEC
    target_ms = int(TOTAL_TARGET_SEC * 1000)
    if len(combined) < target_ms:
        pad = AudioSegment.silent(duration=target_ms - len(combined))
        combined += pad
    elif len(combined) > target_ms:
        combined = combined[:target_ms]

    out_path = os.path.join(folder, "combined_best15.mp3")
    combined.export(out_path, format="mp3")
    print(f"\nCombined audio saved to: {out_path}")


if __name__ == "__main__":
    main()
