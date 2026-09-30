from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_TRACK_DIR = ROOT / "outputs" / "analytics" / "tracks"
DEFAULT_FRAME_METRICS_DIR = ROOT / "outputs" / "analytics" / "frame_metrics"
DEFAULT_EVENT_DIR = ROOT / "outputs" / "analytics" / "events"


def safe_video_id(video_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", str(video_id)).strip("_")


def resolve_project_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else ROOT / path


def min_max_normalize(values: pd.Series) -> pd.Series:
    values = values.astype(float).fillna(0.0)
    span = values.max() - values.min()
    if span <= 0:
        return pd.Series(np.zeros(len(values)), index=values.index)
    return (values - values.min()) / span


def per_frame_player_motion(tracks: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for _, group in tracks.sort_values("frame_index").groupby("track_id"):
        previous = None
        for row in group.itertuples(index=False):
            current = np.array([float(row.center_x), float(row.center_y)])
            if previous is not None:
                displacement = float(np.linalg.norm(current - previous))
                rows.append(
                    {
                        "frame_index": int(row.frame_index),
                        "player_motion_pixels": displacement,
                    }
                )
            previous = current

    if not rows:
        return pd.DataFrame(
            columns=["frame_index", "mean_player_motion_pixels"]
        )

    return (
        pd.DataFrame(rows)
        .groupby("frame_index", as_index=False)["player_motion_pixels"]
        .mean()
        .rename(columns={"player_motion_pixels": "mean_player_motion_pixels"})
    )


def build_activity_series(
    tracks: pd.DataFrame,
    frame_metrics: pd.DataFrame,
    smooth_window: int,
) -> pd.DataFrame:
    activity = frame_metrics.copy()
    motion = per_frame_player_motion(tracks)
    activity = activity.merge(motion, on="frame_index", how="left")
    activity["mean_player_motion_pixels"] = activity[
        "mean_player_motion_pixels"
    ].fillna(0.0)

    components = {
        "motion_component": min_max_normalize(
            activity["mean_player_motion_pixels"]
        ),
        "players_component": min_max_normalize(
            activity["visible_player_count"]
        ),
        "spacing_component": min_max_normalize(
            activity["average_pairwise_player_distance_pixels"]
        ),
        "spread_component": min_max_normalize(
            activity["x_position_spread"] + activity["y_position_spread"]
        ),
    }

    for name, values in components.items():
        activity[name] = values

    activity["raw_activity_score"] = (
        0.50 * activity["motion_component"]
        + 0.20 * activity["players_component"]
        + 0.15 * activity["spacing_component"]
        + 0.15 * activity["spread_component"]
    )
    activity["activity_score"] = (
        activity["raw_activity_score"]
        .rolling(window=max(1, smooth_window), center=True, min_periods=1)
        .mean()
    )
    return activity.sort_values("frame_index").reset_index(drop=True)


def select_activity_peaks(
    activity: pd.DataFrame,
    top_k: int,
    min_separation_frames: int,
    percentile_threshold: float,
) -> pd.DataFrame:
    if activity.empty:
        return activity

    threshold = float(
        np.percentile(activity["activity_score"], percentile_threshold)
    )
    candidates = []
    scores = activity["activity_score"].to_numpy(dtype=float)
    frame_indices = activity["frame_index"].to_numpy(dtype=int)

    for idx, score in enumerate(scores):
        previous_score = scores[idx - 1] if idx > 0 else -np.inf
        next_score = scores[idx + 1] if idx < len(scores) - 1 else -np.inf
        if score >= threshold and score >= previous_score and score >= next_score:
            candidates.append(
                {
                    "row_index": idx,
                    "frame_index": int(frame_indices[idx]),
                    "activity_score": float(score),
                }
            )

    selected = []
    for candidate in sorted(
        candidates,
        key=lambda row: row["activity_score"],
        reverse=True,
    ):
        if all(
            abs(candidate["frame_index"] - chosen["frame_index"])
            >= min_separation_frames
            for chosen in selected
        ):
            selected.append(candidate)
        if len(selected) >= top_k:
            break

    selected_indices = [row["row_index"] for row in selected]
    return activity.iloc[selected_indices].sort_values("frame_index")


def detect_events(
    tracks: pd.DataFrame,
    frame_metrics: pd.DataFrame,
    context_frames: int,
    smooth_window: int,
    top_k: int,
    min_separation_frames: int,
    percentile_threshold: float,
) -> pd.DataFrame:
    if tracks.empty or frame_metrics.empty:
        return pd.DataFrame(
            columns=[
                "video_id",
                "peak_frame",
                "peak_timestamp",
                "start_frame",
                "end_frame",
                "activity_score",
            ]
        )

    activity = build_activity_series(
        tracks=tracks,
        frame_metrics=frame_metrics,
        smooth_window=smooth_window,
    )
    peaks = select_activity_peaks(
        activity=activity,
        top_k=top_k,
        min_separation_frames=min_separation_frames,
        percentile_threshold=percentile_threshold,
    )

    min_frame = int(frame_metrics["frame_index"].min())
    max_frame = int(frame_metrics["frame_index"].max())
    video_id = str(frame_metrics["video_id"].iloc[0])

    rows = []
    for peak in peaks.itertuples(index=False):
        peak_frame = int(peak.frame_index)
        rows.append(
            {
                "video_id": video_id,
                "peak_frame": peak_frame,
                "peak_timestamp": float(peak.timestamp_seconds),
                "start_frame": max(min_frame, peak_frame - context_frames),
                "end_frame": min(max_frame, peak_frame + context_frames),
                "activity_score": float(peak.activity_score),
            }
        )

    return pd.DataFrame(rows)


def process_video(
    track_path: Path,
    frame_metrics_dir: Path,
    event_dir: Path,
    overwrite: bool,
    context_frames: int,
    smooth_window: int,
    top_k: int,
    min_separation_frames: int,
    percentile_threshold: float,
) -> Path | None:
    tracks = pd.read_csv(track_path)
    if tracks.empty:
        safe_id = track_path.stem
        frame_metrics_path = frame_metrics_dir / f"{safe_id}.csv"
        video_id = safe_id
    else:
        video_id = str(tracks["video_id"].iloc[0])
        safe_id = safe_video_id(video_id)
        frame_metrics_path = frame_metrics_dir / f"{safe_id}.csv"

    if not frame_metrics_path.exists():
        print(f"Missing frame metrics, skipping events: {frame_metrics_path}")
        return None

    event_dir.mkdir(parents=True, exist_ok=True)
    output_path = event_dir / f"{safe_id}.csv"
    if output_path.exists() and not overwrite:
        print(f"Skipping existing events: {output_path}")
        return output_path

    frame_metrics = pd.read_csv(frame_metrics_path)
    events = detect_events(
        tracks=tracks,
        frame_metrics=frame_metrics,
        context_frames=context_frames,
        smooth_window=smooth_window,
        top_k=top_k,
        min_separation_frames=min_separation_frames,
        percentile_threshold=percentile_threshold,
    )
    if events.empty:
        events = pd.DataFrame(
            columns=[
                "video_id",
                "peak_frame",
                "peak_timestamp",
                "start_frame",
                "end_frame",
                "activity_score",
            ]
        )
    elif "video_id" not in events.columns:
        events.insert(0, "video_id", video_id)

    events.to_csv(output_path, index=False)
    print(f"Saved activity events: {output_path}")
    return output_path


def run_event_detection(
    track_dir: Path,
    frame_metrics_dir: Path,
    event_dir: Path,
    overwrite: bool,
    context_frames: int,
    smooth_window: int,
    top_k: int,
    min_separation_frames: int,
    percentile_threshold: float,
) -> None:
    track_files = sorted(track_dir.glob("*.csv"))
    if not track_files:
        print(f"No tracking CSVs found in {track_dir}")
        return

    for track_path in track_files:
        process_video(
            track_path=track_path,
            frame_metrics_dir=frame_metrics_dir,
            event_dir=event_dir,
            overwrite=overwrite,
            context_frames=context_frames,
            smooth_window=smooth_window,
            top_k=top_k,
            min_separation_frames=min_separation_frames,
            percentile_threshold=percentile_threshold,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect high-activity windows from pixel-based player movement."
    )
    parser.add_argument("--track-dir", type=Path, default=DEFAULT_TRACK_DIR)
    parser.add_argument(
        "--frame-metrics-dir",
        type=Path,
        default=DEFAULT_FRAME_METRICS_DIR,
    )
    parser.add_argument("--event-dir", type=Path, default=DEFAULT_EVENT_DIR)
    parser.add_argument("--context-frames", type=int, default=45)
    parser.add_argument("--smooth-window", type=int, default=9)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--min-separation-frames", type=int, default=60)
    parser.add_argument("--percentile-threshold", type=float, default=75.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_event_detection(
        track_dir=resolve_project_path(args.track_dir),
        frame_metrics_dir=resolve_project_path(args.frame_metrics_dir),
        event_dir=resolve_project_path(args.event_dir),
        overwrite=args.overwrite,
        context_frames=args.context_frames,
        smooth_window=args.smooth_window,
        top_k=args.top_k,
        min_separation_frames=args.min_separation_frames,
        percentile_threshold=args.percentile_threshold,
    )


if __name__ == "__main__":
    main()
