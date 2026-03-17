#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import sys
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np


TARGET_DISCORD_BYTES = 9_800_000
ANALYSIS_SAMPLE_RATE = 16_000
DEFAULT_MIC_TRACK = 2
DEFAULT_LOOKBACK_SECONDS = 30.0
DEFAULT_POSTROLL_SECONDS = 5.0
WINDOW_SECONDS = 0.25
HOP_SECONDS = 0.05
SMOOTHING_WINDOWS = 5
MERGE_GAP_SECONDS = 8.0
PERCENTILE_ATTEMPTS = (
    (99.5, 6.0),
    (99.2, 5.5),
    (98.8, 5.0),
)


@dataclass
class Marker:
    index: int
    marker_seconds: float
    clip_start_seconds: float
    clip_end_seconds: float
    segment_start_seconds: float
    segment_end_seconds: float
    rms_db: float
    peak_db: float
    score: float
    output_path: Path
    limited_for_discord: bool = False
    output_size_bytes: int | None = None


def format_timestamp(seconds: float, include_millis: bool = False) -> str:
    total_millis = int(round(max(0.0, seconds) * 1000))
    hours, remainder = divmod(total_millis, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    if include_millis:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def format_filesize(num_bytes: int | None) -> str:
    if num_bytes is None:
        return "unknown"
    return f"{num_bytes / 1_000_000:.2f} MB"


def timestamp_slug(seconds: float) -> str:
    return format_timestamp(seconds, include_millis=True).replace(":", "-").replace(".", "_")


def require_tool(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(f"{name} was not found in PATH.")


def run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, capture_output=True, text=True, check=True)


def probe_audio_tracks(input_path: Path) -> list[str]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a",
        "-show_entries",
        "stream=index:stream_tags=title",
        "-of",
        "csv=p=0",
        str(input_path),
    ]
    result = run_command(command)
    tracks: list[str] = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(",")]
        title = parts[1] if len(parts) > 1 and parts[1] else f"Track{len(tracks) + 1}"
        tracks.append(title)
    return tracks


def get_duration_seconds(input_path: Path) -> float:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(input_path),
    ]
    result = run_command(command)
    return float(result.stdout.strip())


def extract_analysis_track(input_path: Path, audio_stream_index: int, wav_path: Path) -> None:
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(input_path),
        "-map",
        f"0:a:{audio_stream_index}",
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(ANALYSIS_SAMPLE_RATE),
        "-c:a",
        "pcm_s16le",
        str(wav_path),
    ]
    run_command(command)


def compute_window_features(wav_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    with wave.open(str(wav_path), "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        if channels != 1 or sample_width != 2:
            raise RuntimeError("expected mono 16-bit PCM analysis audio")

        window_samples = max(1, int(round(WINDOW_SECONDS * sample_rate)))
        hop_samples = max(1, int(round(HOP_SECONDS * sample_rate)))
        chunk_samples = sample_rate * 10

        buffer = np.empty(0, dtype=np.float32)
        window_times: list[float] = []
        rms_db_values: list[float] = []
        peak_db_values: list[float] = []
        start_sample = 0

        while True:
            raw = wav_file.readframes(chunk_samples)
            if not raw:
                break

            chunk = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            buffer = np.concatenate((buffer, chunk)) if buffer.size else chunk

            offset = 0
            while buffer.size - offset >= window_samples:
                segment = buffer[offset : offset + window_samples]
                rms = float(np.sqrt(np.mean(segment * segment) + 1e-12))
                peak = float(np.max(np.abs(segment)))
                window_times.append(start_sample / sample_rate)
                rms_db_values.append(20.0 * math.log10(rms + 1e-12))
                peak_db_values.append(20.0 * math.log10(peak + 1e-12))
                start_sample += hop_samples
                offset += hop_samples

            if offset:
                buffer = buffer[offset:]

        duration_seconds = wav_file.getnframes() / sample_rate

    return (
        np.asarray(window_times, dtype=np.float64),
        np.asarray(rms_db_values, dtype=np.float64),
        np.asarray(peak_db_values, dtype=np.float64),
        duration_seconds,
    )


def robust_zscore(values: np.ndarray) -> tuple[np.ndarray, float, float]:
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    scale = max(1.4826 * mad, 0.75)
    return (values - median) / scale, median, scale


def find_local_maxima(values: np.ndarray) -> np.ndarray:
    if values.size < 3:
        return np.arange(values.size)
    interior = np.where((values[1:-1] >= values[:-2]) & (values[1:-1] >= values[2:]))[0] + 1
    maxima = interior.tolist()
    if values[0] > values[1]:
        maxima.insert(0, 0)
    if values[-1] > values[-2]:
        maxima.append(values.size - 1)
    return np.asarray(maxima, dtype=np.int64)


def suppress_nearby_candidates(
    candidate_indices: list[int],
    times: np.ndarray,
    scores: np.ndarray,
    min_gap_seconds: float,
) -> list[int]:
    kept: list[int] = []
    for index in candidate_indices:
        if not kept:
            kept.append(index)
            continue

        previous = kept[-1]
        if times[index] - times[previous] < min_gap_seconds:
            if scores[index] > scores[previous]:
                kept[-1] = index
            continue

        kept.append(index)
    return kept


def detect_markers(
    times: np.ndarray,
    rms_db: np.ndarray,
    peak_db: np.ndarray,
    lookback_seconds: float,
    postroll_seconds: float,
    duration_seconds: float,
) -> tuple[list[Marker], dict[str, float]]:
    if times.size == 0:
        return [], {}

    smoothing_kernel = np.ones(SMOOTHING_WINDOWS, dtype=np.float64) / SMOOTHING_WINDOWS
    smoothed_rms = np.convolve(rms_db, smoothing_kernel, mode="same")
    score_rms, baseline_db, _ = robust_zscore(smoothed_rms)
    score_peak, _, _ = robust_zscore(peak_db)
    combined_score = score_rms + (0.35 * score_peak)
    local_maxima = find_local_maxima(smoothed_rms)

    candidate_indices: list[int] = []
    chosen_percentile = 0.0
    chosen_min_score = 0.0
    chosen_rms_threshold = 0.0
    peak_gate = max(float(np.quantile(peak_db, 0.95)), float(np.median(peak_db) + 3.0))

    for percentile, min_score in PERCENTILE_ATTEMPTS:
        rms_threshold = max(
            float(np.quantile(smoothed_rms, percentile / 100.0)),
            baseline_db + 12.0,
        )
        current = [
            index
            for index in local_maxima
            if smoothed_rms[index] >= rms_threshold
            and combined_score[index] >= min_score
            and peak_db[index] >= peak_gate
        ]
        current.sort(key=lambda idx: times[idx])
        current = suppress_nearby_candidates(current, times, combined_score, MERGE_GAP_SECONDS)
        if current:
            candidate_indices = current
            chosen_percentile = percentile
            chosen_min_score = min_score
            chosen_rms_threshold = rms_threshold
            break

    if not candidate_indices:
        return [], {
            "baseline_db": baseline_db,
            "peak_gate_db": peak_gate,
        }

    markers: list[Marker] = []
    segment_floor_margin = 5.0

    for clip_index, index in enumerate(candidate_indices, start=1):
        segment_floor = max(baseline_db + 8.0, smoothed_rms[index] - segment_floor_margin)
        left = index
        while left > 0 and smoothed_rms[left - 1] >= segment_floor:
            left -= 1
        right = index
        while right + 1 < smoothed_rms.size and smoothed_rms[right + 1] >= segment_floor:
            right += 1

        marker_seconds = float(times[index])
        clip_end_seconds = min(duration_seconds, marker_seconds)
        clip_start_seconds = max(0.0, clip_end_seconds - lookback_seconds)
        segment_end_seconds = min(duration_seconds, float(times[right] + WINDOW_SECONDS))
        markers.append(
            Marker(
                index=clip_index,
                marker_seconds=marker_seconds,
                clip_start_seconds=clip_start_seconds,
                clip_end_seconds=min(duration_seconds, clip_end_seconds + postroll_seconds),
                segment_start_seconds=float(times[left]),
                segment_end_seconds=segment_end_seconds,
                rms_db=float(rms_db[index]),
                peak_db=float(peak_db[index]),
                score=float(combined_score[index]),
                output_path=Path(),
            )
        )

    return markers, {
        "baseline_db": baseline_db,
        "rms_threshold_db": chosen_rms_threshold,
        "peak_gate_db": peak_gate,
        "percentile": chosen_percentile,
        "min_score": chosen_min_score,
    }


def cut_clip(input_path: Path, marker: Marker, output_path: Path) -> None:
    duration_seconds = max(0.01, marker.clip_end_seconds - marker.clip_start_seconds)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{marker.clip_start_seconds:.3f}",
        "-i",
        str(input_path),
        "-t",
        f"{duration_seconds:.3f}",
        "-map",
        "0:v:0",
        "-map",
        "0:a:0?",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "18",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    run_command(command)


def limit_clip_to_discord(input_path: Path, target_bytes: int = TARGET_DISCORD_BYTES) -> bool:
    if input_path.stat().st_size <= target_bytes:
        return False

    duration_seconds = get_duration_seconds(input_path)
    if duration_seconds <= 0:
        raise RuntimeError(f"Unable to read duration for {input_path}")

    temp_output = input_path.with_name(f"{input_path.stem}_discord{input_path.suffix}")
    audio_kbps = 96
    best_size = None
    best_output: Path | None = None

    try:
        for safety_factor in (0.97, 0.94, 0.91):
            effective_bytes = int(target_bytes * safety_factor)
            total_kbps = max(220, math.floor((effective_bytes * 8 / 1000) / duration_seconds))
            video_kbps = max(120, total_kbps - audio_kbps)
            buffer_kbps = video_kbps * 2
            pass_log = input_path.with_name(f"{input_path.stem}_discord_pass")

            pass_one = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(input_path),
                "-c:v",
                "libx264",
                "-b:v",
                f"{video_kbps}k",
                "-maxrate",
                f"{video_kbps}k",
                "-bufsize",
                f"{buffer_kbps}k",
                "-pass",
                "1",
                "-passlogfile",
                str(pass_log),
                "-an",
                "-f",
                "null",
                "NUL",
            ]
            pass_two = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(input_path),
                "-c:v",
                "libx264",
                "-b:v",
                f"{video_kbps}k",
                "-maxrate",
                f"{video_kbps}k",
                "-bufsize",
                f"{buffer_kbps}k",
                "-pass",
                "2",
                "-passlogfile",
                str(pass_log),
                "-c:a",
                "aac",
                "-b:a",
                f"{audio_kbps}k",
                "-movflags",
                "+faststart",
                str(temp_output),
            ]

            try:
                run_command(pass_one)
                run_command(pass_two)
            finally:
                for suffix in ("-0.log", "-0.log.mbtree"):
                    log_path = pass_log.with_name(pass_log.name + suffix)
                    if log_path.exists():
                        log_path.unlink()

            current_size = temp_output.stat().st_size
            if best_size is None or current_size < best_size:
                best_size = current_size
                best_output = temp_output

            if current_size <= target_bytes:
                temp_output.replace(input_path)
                return True

        if best_output is not None and best_output.exists():
            best_output.replace(input_path)
            return True
        return False
    finally:
        if temp_output.exists():
            temp_output.unlink()


def write_reports(
    report_path: Path,
    annotation_path: Path,
    input_path: Path,
    markers: list[Marker],
    track_label: str,
    detection_info: dict[str, float],
    limit_for_discord: bool,
    lookback_seconds: float,
    postroll_seconds: float,
) -> None:
    report_lines = [
        f"Input: {input_path.name}",
        f"Mic track: {track_label}",
        f"Lookback: {lookback_seconds:.0f}s",
        f"Postroll: {postroll_seconds:.0f}s",
        f"Discord size limit enabled: {'yes' if limit_for_discord else 'no'}",
    ]

    if detection_info:
        baseline = detection_info.get("baseline_db")
        rms_threshold = detection_info.get("rms_threshold_db")
        peak_gate = detection_info.get("peak_gate_db")
        percentile = detection_info.get("percentile")
        min_score = detection_info.get("min_score")
        if baseline is not None:
            report_lines.append(f"Baseline RMS: {baseline:.2f} dB")
        if rms_threshold is not None:
            report_lines.append(f"RMS threshold: {rms_threshold:.2f} dB")
        if peak_gate is not None:
            report_lines.append(f"Peak gate: {peak_gate:.2f} dB")
        if percentile is not None and min_score is not None:
            report_lines.append(f"Detector: top {100.0 - percentile:.1f}% windows, min score {min_score:.1f}")

    report_lines.append("")

    annotation_lines: list[str] = []
    for marker in markers:
        report_lines.append(
            (
                f"{marker.index:02d}. marker {format_timestamp(marker.marker_seconds, include_millis=True)}"
                f" | clip {format_timestamp(marker.clip_start_seconds, include_millis=True)}"
                f" -> {format_timestamp(marker.clip_end_seconds, include_millis=True)}"
                f" | shout {format_timestamp(marker.segment_start_seconds, include_millis=True)}"
                f" -> {format_timestamp(marker.segment_end_seconds, include_millis=True)}"
                f" | rms {marker.rms_db:.2f} dB"
                f" | peak {marker.peak_db:.2f} dB"
                f" | score {marker.score:.2f}"
                f" | file {marker.output_path.name}"
                f" | size {format_filesize(marker.output_size_bytes)}"
                f" | discord {'yes' if marker.limited_for_discord else 'no'}"
            )
        )
        annotation_lines.append(
            f"{format_timestamp(marker.clip_start_seconds)} -> auto_marker_{marker.index:02d} <- {format_timestamp(marker.clip_end_seconds)}"
        )

    if not markers:
        report_lines.append("No hype markers were detected.")

    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    annotation_path.write_text("\n".join(annotation_lines) + ("\n" if annotation_lines else ""), encoding="utf-8")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Detect loud mic markers in an OBS recording and cut shareable clips around each marker."
    )
    parser.add_argument("input_path", help="Video file to analyze. Relative paths are resolved from the current directory.")
    parser.add_argument(
        "--track",
        type=int,
        default=DEFAULT_MIC_TRACK,
        help="OBS audio track number to analyze. Default: 2",
    )
    lookback_group = parser.add_mutually_exclusive_group()
    lookback_group.add_argument(
        "--lookback",
        type=float,
        default=None,
        help="Seconds to keep before each detected marker. Default: 30",
    )
    lookback_group.add_argument(
        "--clip-length",
        type=float,
        default=None,
        help="Total clip duration in seconds, including postroll. The pre-marker window is derived from this value.",
    )
    parser.add_argument(
        "--postroll",
        type=float,
        default=DEFAULT_POSTROLL_SECONDS,
        help="Seconds to keep after each detected marker. Default: 5",
    )
    parser.add_argument(
        "--discord-compress",
        action="store_true",
        help="Re-encode each extracted clip to stay under 9.8 MB for Discord free uploads.",
    )
    return parser


def resolve_runtime_settings(args: argparse.Namespace) -> tuple[bool, float]:
    if args.postroll < 0:
        raise RuntimeError("postroll seconds cannot be negative")

    if args.clip_length is not None:
        if args.clip_length <= args.postroll:
            raise RuntimeError("clip length must be greater than postroll so there is still some pre-marker context")
        lookback_seconds = args.clip_length - args.postroll
    elif args.lookback is not None:
        lookback_seconds = args.lookback
    else:
        lookback_seconds = DEFAULT_LOOKBACK_SECONDS

    if lookback_seconds <= 0:
        raise RuntimeError("lookback seconds must be greater than 0")

    return bool(args.discord_compress), float(lookback_seconds)


def main() -> int:
    parser = build_argument_parser()
    args = parser.parse_args()

    require_tool("ffmpeg")
    require_tool("ffprobe")

    input_path = Path(args.input_path).expanduser().resolve()
    if not input_path.exists():
        raise RuntimeError(f"Input file not found: {input_path}")
    if args.track < 1:
        raise RuntimeError("track numbers start at 1")

    limit_to_discord, lookback_seconds = resolve_runtime_settings(args)

    audio_tracks = probe_audio_tracks(input_path)
    if not audio_tracks:
        raise RuntimeError(f"No audio tracks found in {input_path}")
    if args.track > len(audio_tracks):
        available = ", ".join(f"{index + 1}:{label}" for index, label in enumerate(audio_tracks))
        raise RuntimeError(f"Track {args.track} is not available. Found audio tracks: {available}")

    output_dir = input_path.with_name(f"{input_path.stem}_auto")
    output_dir.mkdir(parents=True, exist_ok=True)

    analysis_wav = output_dir / f"{input_path.stem}_track{args.track}_analysis.wav"
    report_path = output_dir / "auto_review.txt"
    annotation_path = output_dir / "auto_annotations.txt"

    try:
        extract_analysis_track(input_path, args.track - 1, analysis_wav)
        times, rms_db, peak_db, duration_seconds = compute_window_features(analysis_wav)
        markers, detection_info = detect_markers(
            times=times,
            rms_db=rms_db,
            peak_db=peak_db,
            lookback_seconds=lookback_seconds,
            postroll_seconds=float(args.postroll),
            duration_seconds=duration_seconds,
        )

        for marker in markers:
            clip_name = f"{marker.index:02d}_marker_{timestamp_slug(marker.marker_seconds)}.mp4"
            clip_path = output_dir / clip_name
            cut_clip(input_path, marker, clip_path)
            marker.output_path = clip_path

            if limit_to_discord:
                marker.limited_for_discord = limit_clip_to_discord(clip_path)

            marker.output_size_bytes = clip_path.stat().st_size

        write_reports(
            report_path=report_path,
            annotation_path=annotation_path,
            input_path=input_path,
            markers=markers,
            track_label=audio_tracks[args.track - 1],
            detection_info=detection_info,
            limit_for_discord=limit_to_discord,
            lookback_seconds=lookback_seconds,
            postroll_seconds=float(args.postroll),
        )
    finally:
        if analysis_wav.exists():
            analysis_wav.unlink()

    print(f"Review file: {report_path}")
    print(f"Annotation file: {annotation_path}")
    if markers:
        print(f"Created {len(markers)} clip(s) in {output_dir}")
    else:
        print("No hype markers were detected.")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as error:
        stderr = (error.stderr or "").strip()
        stdout = (error.stdout or "").strip()
        details = stderr or stdout or "command failed"
        print(details, file=sys.stderr)
        raise SystemExit(1)
    except Exception as error:  # noqa: BLE001
        print(str(error), file=sys.stderr)
        raise SystemExit(1)
