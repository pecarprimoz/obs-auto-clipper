# OBS Auto Clipper

`auto_clip.py` scans an OBS recording, finds loud mic callouts on a chosen audio track, and cuts highlight clips around those markers.

The current workflow is optimized for a personal "clip it" style marker:

- detect strong peaks on your isolated mic track
- cut context before the marker
- keep a small postroll after the marker
- optionally compress the output for Discord free uploads

## Requirements

- Python 3.10+
- `ffmpeg` in `PATH`
- `ffprobe` in `PATH`
- OBS recordings with a separate mic track

## Recommended OBS Setup

- `Track 1`: full mix
- `Track 2`: mic only
- `Track 3`: game only
- `Track 4`: Discord only

The script analyzes `Track 2` by default.

## Usage

Basic run:

```powershell
python .\auto_clip.py .\my_session.mkv
```

Custom lookback:

```powershell
python .\auto_clip.py .\my_session.mkv --lookback 45
```

Fixed total clip duration:

```powershell
python .\auto_clip.py .\my_session.mkv --clip-length 50
```

Discord-friendly output:

```powershell
python .\auto_clip.py .\my_session.mkv --clip-length 50 --discord-compress
```

Analyze a different OBS audio track:

```powershell
python .\auto_clip.py .\my_session.mkv --track 3
```

## Flags

- `--lookback`: seconds to keep before the detected marker
- `--clip-length`: total clip duration, including postroll
- `--postroll`: seconds to keep after the detected marker, default `5`
- `--discord-compress`: re-encode each clip to stay under `9.8 MB`
- `--track`: OBS audio track number to analyze, default `2`

`--lookback` and `--clip-length` are mutually exclusive.

## Output

For an input like `my_session.mkv`, the script creates:

- `my_session_auto/`
- `my_session_auto/auto_review.txt`
- `my_session_auto/auto_annotations.txt`
- `my_session_auto/*.mp4`

`auto_review.txt` is the main human-readable report. It includes:

- detected marker timestamp
- extracted clip start and end
- shout segment bounds
- loudness and score
- output file name and size

`auto_annotations.txt` is a simple text version of the clip ranges if you want to reuse or edit them later.

## Example: Original vs Produced Clips

The repo includes a small sample recording in `examples/` so you can see the shape of the workflow:

- original recording: `examples/clip_it_test.mkv`
- duration: `103.217s`
- size: `83.86 MB`

Current produced clips from that sample:

- `examples/clip_it_test_auto/01_marker_00-00-20_800.mp4`
  - duration: `17.0s`
  - size: `29.68 MB`
- `examples/clip_it_test_auto/02_marker_00-01-34_850.mp4`
  - duration: `17.0s`
  - size: `27.10 MB`

If you add `--discord-compress`, the script re-encodes the produced clips to stay under `9.8 MB` for sharing on Discord free uploads.

## Notes

- This is a `v1` loudness-based detector, not speech recognition.
- Long sessions are fine; the script processes audio linearly and does not load the full video into memory.
- Detection quality depends mostly on having a clean isolated mic track.
- Media files are ignored by git by default in this repo layout. If you want to publish sample recordings, use Git LFS or adjust `.gitignore`.

## Example Layout

```text
obs-auto-clipper/
  auto_clip.py
  README.md
  .gitignore
  examples/
```
