# IG Chart Reels Generator

This project generates Instagram-style chart reels from a folder of audio tracks. It analyzes tracks, extracts metadata and cover art, builds a combined audio file, and creates a vertical video suitable for social media sharing.

## Features
- **Audio Analysis:** Finds the best segment of each track for highlight.
- **Metadata Extraction:** Reads artist, title, album, year, and cover art from audio files.
- **Combined Audio:** Merges selected segments into a single audio file.
- **Video Generation:** Creates a vertical video with covers, captions, and background video.
- **Cleanup:** Removes temporary files after processing.

## Requirements
- Python 3.8+
- [pydub](https://github.com/jiaaro/pydub)
- [moviepy](https://zulko.github.io/moviepy/)
- [Pillow](https://python-pillow.org/)
- [librosa](https://librosa.org/)
- [mutagen](https://mutagen.readthedocs.io/en/latest/)

Install dependencies:
```bash
pip install pydub moviepy Pillow librosa mutagen
```

## Usage
1. Place your audio files (mp3, wav, flac, m4a, ogg) in a folder.
2. Add a background video (mp4) to the same folder.
3. Run the pipeline:

```bash
python run.py <folder_path>
```

This will:
- Analyze tracks and extract metadata/cover art
- Build a combined audio file
- Generate a vertical video with covers and captions
- Clean up temporary files

### Individual Steps
You can run each step separately:

- Analyze tracks:
  ```bash
  python analyze_tracks.py <folder_path>
  ```
- Build audio:
  ```bash
  python build_audio.py <folder_path>
  ```
- Build video:
  ```bash
  python build_video.py <folder_path>
  ```
- Cleanup:
  ```bash
  python cleanup.py <folder_path>
  ```

## Output
- `_audio_1080x1920`:Final vertical video for sharing

## Customization
- Edit font sizes, layout, and video size in `build_video.py`
- Change target duration in `media_utils.py` (`TOTAL_TARGET_SEC`)

## License
MIT

## Author
Alex Ilchenko
