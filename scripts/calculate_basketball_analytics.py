from __future__ import annotations

import argparse
import re
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_TRACK_DIR = ROOT / "outputs" / "analytics" / "tracks"
DEFAULT_PLAYER_METRICS_DIR = ROOT / "outputs" / "analytics" / "player_metrics"
DEFAULT_FRAME_METRICS_DIR = ROOT / "outputs" / "analytics" / "frame_metrics"


def safe_video_id(video_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", str(video_id)).strip("_")


def resolve_project_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else ROOT / path


def calculate_player_metrics(tracks: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for track_id, group in tracks.sort_values("frame_index").groupby("track_id"):
        centers = group[["center_x", "center_y"]].to_numpy(dtype=float)
        if len(centers) > 1:
            deltas = np.linalg.norm(np.diff(centers, axis=0), axis=1)
        else:
            deltas = np.array([], dtype=float)

        frames_seen = int(len(group))
        total_displacement = float(deltas.sum()) if len(deltas) else 0.0
        average_displacement = (
            total_displacement / max(frames_seen - 1, 1)
            if frames_seen > 1
            else 0.0
        )
        max_displacement = float(deltas.max()) if len(deltas) else 0.0

        rows.append(
            {
                "track_id": int(track_id),
                "frames_seen": frames_seen,
                "first_frame": int(group["frame_index"].min()),
                "last_frame": int(group["frame_index"].max()),
                "total_pixel_displacement": total_displacement,
                "average_pixel_displacement_per_frame": average_displacement,
                "max_pixel_displacement": max_displacement,
            }
        )

    columns = [
        "track_id",
        "frames_seen",
        "first_frame",
        "last_frame",
        "total_pixel_displacement",
        "average_pixel_displacement_per_frame",
        "max_pixel_displacement",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns).sort_values("track_id").reset_index(drop=True)


def average_pairwise_distance(centers: np.ndarray) -> float:
    if len(centers) < 2:
        return 0.0
    distances = [
        float(np.linalg.norm(first - second))
        for first, second in combinations(centers, 2)
    ]
    return float(np.mean(distances)) if distances else 0.0


def calculate_frame_metrics(tracks: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for frame_index, group in tracks.sort_values("frame_index").groupby("frame_index"):
        centers = group[["center_x", "center_y"]].to_numpy(dtype=float)
        visible_count = int(len(group))
        centroid = centers.mean(axis=0) if visible_count else np.array([0.0, 0.0])

        rows.append(
            {
                "video_id": str(group["video_id"].iloc[0]),
                "frame_index": int(frame_index),
                "timestamp_seconds": float(group["timestamp_seconds"].iloc[0]),
                "visible_player_count": visible_count,
                "average_pairwise_player_distance_pixels": average_pairwise_distance(centers),
                "x_position_spread": float(group["center_x"].max() - group["center_x"].min())
                if visible_count
                else 0.0,
                "y_position_spread": float(group["center_y"].max() - group["center_y"].min())
                if visible_count
                else 0.0,
                "centroid_x": float(centroid[0]),
                "centroid_y": float(centroid[1]),
            }
        )

    columns = [
        "video_id",
        "frame_index",
        "timestamp_seconds",
        "visible_player_count",
        "average_pairwise_player_distance_pixels",
        "x_position_spread",
        "y_position_spread",
        "centroid_x",
        "centroid_y",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns).sort_values("frame_index").reset_index(drop=True)


def process_track_file(
    track_path: Path,
    player_metrics_dir: Path,
    frame_metrics_dir: Path,
    overwrite: bool,
) -> tuple[Path, Path]:
    tracks = pd.read_csv(track_path)
    if tracks.empty:
        video_id = track_path.stem
    else:
        video_id = str(tracks["video_id"].iloc[0])

    safe_id = safe_video_id(video_id)
    player_output = player_metrics_dir / f"{safe_id}.csv"
    frame_output = frame_metrics_dir / f"{safe_id}.csv"

    if player_output.exists() and frame_output.exists() and not overwrite:
        print(f"Skipping existing metrics: {safe_id}")
        return player_output, frame_output

    player_metrics_dir.mkdir(parents=True, exist_ok=True)
    frame_metrics_dir.mkdir(parents=True, exist_ok=True)

    player_metrics = calculate_player_metrics(tracks)
    frame_metrics = calculate_frame_metrics(tracks) if not tracks.empty else pd.DataFrame(
        columns=[
            "video_id",
            "frame_index",
            "timestamp_seconds",
            "visible_player_count",
            "average_pairwise_player_distance_pixels",
            "x_position_spread",
            "y_position_spread",
            "centroid_x",
            "centroid_y",
        ]
    )

    player_metrics.to_csv(player_output, index=False)
    frame_metrics.to_csv(frame_output, index=False)

    print(f"Saved player metrics: {player_output}")
    print(f"Saved frame metrics: {frame_output}")
    return player_output, frame_output


def run_metrics(
    track_dir: Path,
    player_metrics_dir: Path,
    frame_metrics_dir: Path,
    overwrite: bool,
) -> None:
    track_files = sorted(track_dir.glob("*.csv"))
    if not track_files:
        print(f"No tracking CSVs found in {track_dir}")
        return

    for track_path in track_files:
        process_track_file(
            track_path=track_path,
            player_metrics_dir=player_metrics_dir,
            frame_metrics_dir=frame_metrics_dir,
            overwrite=overwrite,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate pixel-based basketball tracking analytics."
    )
    parser.add_argument("--track-dir", type=Path, default=DEFAULT_TRACK_DIR)
    parser.add_argument(
        "--player-metrics-dir",
        type=Path,
        default=DEFAULT_PLAYER_METRICS_DIR,
    )
    parser.add_argument(
        "--frame-metrics-dir",
        type=Path,
        default=DEFAULT_FRAME_METRICS_DIR,
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_metrics(
        track_dir=resolve_project_path(args.track_dir),
        player_metrics_dir=resolve_project_path(args.player_metrics_dir),
        frame_metrics_dir=resolve_project_path(args.frame_metrics_dir),
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
