from __future__ import annotations

import argparse
import re
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_SUBSET_PATH = ROOT / "data" / "subset.csv"
DEFAULT_TRACK_DIR = ROOT / "outputs" / "analytics" / "tracks"
DEFAULT_VISUALIZATION_DIR = ROOT / "outputs" / "analytics" / "visualizations"
DEFAULT_HEATMAP_DIR = ROOT / "outputs" / "analytics" / "heatmaps"


def safe_video_id(video_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", str(video_id)).strip("_")


def resolve_project_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else ROOT / path


def load_unique_videos(subset_path: Path) -> pd.DataFrame:
    subset = pd.read_csv(subset_path)
    return subset[["video_id", "video_path"]].drop_duplicates("video_id")


def track_color(track_id: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(track_id)
    color = rng.integers(40, 235, size=3)
    return int(color[0]), int(color[1]), int(color[2])


def filter_visualized_tracks(
    tracks: pd.DataFrame,
    max_tracks: int,
    min_track_frames: int,
) -> pd.DataFrame:
    """
    Keep the most stable tracks for presentation.

    Raw CSVs retain every YOLO person track. The annotated video is a
    visual aid, so it defaults to the longest tracks to avoid drawing
    refs, bench personnel, spectators, and short noisy tracks.
    """
    if tracks.empty or max_tracks <= 0:
        return tracks.iloc[0:0].copy()

    summary = (
        tracks.groupby("track_id")
        .agg(
            frames_seen=("frame_index", "nunique"),
            mean_confidence=("confidence", "mean"),
        )
        .reset_index()
    )
    summary = summary[summary["frames_seen"] >= min_track_frames]
    if summary.empty:
        summary = (
            tracks.groupby("track_id")
            .agg(
                frames_seen=("frame_index", "nunique"),
                mean_confidence=("confidence", "mean"),
            )
            .reset_index()
        )

    keep_ids = (
        summary.sort_values(
            ["frames_seen", "mean_confidence"],
            ascending=[False, False],
        )
        .head(max_tracks)["track_id"]
        .tolist()
    )

    return tracks[tracks["track_id"].isin(keep_ids)].copy()


def create_annotated_video(
    video_id: str,
    video_path: Path,
    tracks: pd.DataFrame,
    output_path: Path,
    trail_length: int,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0:
        fps = 30.0

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    tracks_by_frame = {
        int(frame_index): group.copy()
        for frame_index, group in tracks.groupby("frame_index")
    }
    trails: dict[int, deque[tuple[int, int]]] = defaultdict(
        lambda: deque(maxlen=trail_length)
    )

    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            frame_tracks = tracks_by_frame.get(frame_index)
            if frame_tracks is not None:
                for row in frame_tracks.itertuples(index=False):
                    track_id = int(row.track_id)
                    x1 = int(round(row.x1))
                    y1 = int(round(row.y1))
                    x2 = int(round(row.x2))
                    y2 = int(round(row.y2))
                    center = (
                        int(round(row.center_x)),
                        int(round(row.center_y)),
                    )
                    trails[track_id].append(center)
                    color = track_color(track_id)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        frame,
                        f"ID {track_id}",
                        (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2,
                        cv2.LINE_AA,
                    )

                    trail = list(trails[track_id])
                    for first, second in zip(trail, trail[1:]):
                        cv2.line(frame, first, second, color, 2, cv2.LINE_AA)

            cv2.putText(
                frame,
                f"{video_id} | frame {frame_index}",
                (18, 34),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            writer.write(frame)
            frame_index += 1
    finally:
        capture.release()
        writer.release()

    return output_path


def create_occupancy_heatmap(
    video_path: Path,
    tracks: pd.DataFrame,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    ok, frame = capture.read()
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()

    if not ok:
        frame = np.zeros((height, width, 3), dtype=np.uint8)

    occupancy = np.zeros((height, width), dtype=np.float32)
    for row in tracks.itertuples(index=False):
        x = int(round(row.center_x))
        y = int(round(row.center_y))
        if 0 <= x < width and 0 <= y < height:
            occupancy[y, x] += 1.0

    occupancy = cv2.GaussianBlur(occupancy, (0, 0), sigmaX=25, sigmaY=25)
    heatmap = cv2.normalize(occupancy, None, 0, 255, cv2.NORM_MINMAX)
    heatmap = heatmap.astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    blended = cv2.addWeighted(frame, 0.55, heatmap, 0.45, 0)

    if not cv2.imwrite(str(output_path), blended):
        raise ValueError(f"Could not save heatmap: {output_path}")

    return output_path


def process_video(
    video_id: str,
    video_path: Path,
    track_dir: Path,
    visualization_dir: Path,
    heatmap_dir: Path,
    overwrite: bool,
    trail_length: int,
    max_visualized_tracks: int,
    min_visualized_track_frames: int,
) -> tuple[Path, Path] | None:
    safe_id = safe_video_id(video_id)
    track_path = track_dir / f"{safe_id}.csv"
    if not track_path.exists():
        print(f"Missing tracks, skipping visualization: {track_path}")
        return None

    tracks = pd.read_csv(track_path)
    visual_tracks = filter_visualized_tracks(
        tracks=tracks,
        max_tracks=max_visualized_tracks,
        min_track_frames=min_visualized_track_frames,
    )
    annotated_path = visualization_dir / f"{safe_id}_tracked.mp4"
    heatmap_path = heatmap_dir / f"{safe_id}_occupancy_heatmap.jpg"

    if (
        annotated_path.exists()
        and heatmap_path.exists()
        and not overwrite
    ):
        print(f"Skipping existing visualizations: {safe_id}")
        return annotated_path, heatmap_path

    create_annotated_video(
        video_id=video_id,
        video_path=video_path,
        tracks=visual_tracks,
        output_path=annotated_path,
        trail_length=trail_length,
    )
    create_occupancy_heatmap(
        video_path=video_path,
        tracks=visual_tracks,
        output_path=heatmap_path,
    )

    print(f"Saved annotated video: {annotated_path}")
    print(f"Saved occupancy heatmap: {heatmap_path}")
    return annotated_path, heatmap_path


def run_visualizations(
    subset_path: Path,
    track_dir: Path,
    visualization_dir: Path,
    heatmap_dir: Path,
    overwrite: bool,
    trail_length: int,
    max_visualized_tracks: int,
    min_visualized_track_frames: int,
    limit: int | None,
    video_id: str | None,
) -> None:
    videos = load_unique_videos(subset_path)
    if video_id:
        videos = videos[videos["video_id"] == video_id]
        if videos.empty:
            raise ValueError(f"video_id not found in subset: {video_id}")
    if limit is not None:
        videos = videos.head(limit)

    for row in videos.itertuples(index=False):
        process_video(
            video_id=str(row.video_id),
            video_path=resolve_project_path(row.video_path),
            track_dir=track_dir,
            visualization_dir=visualization_dir,
            heatmap_dir=heatmap_dir,
            overwrite=overwrite,
            trail_length=trail_length,
            max_visualized_tracks=max_visualized_tracks,
            min_visualized_track_frames=min_visualized_track_frames,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create tracked videos and player occupancy heatmaps."
    )
    parser.add_argument("--subset", type=Path, default=DEFAULT_SUBSET_PATH)
    parser.add_argument("--track-dir", type=Path, default=DEFAULT_TRACK_DIR)
    parser.add_argument(
        "--visualization-dir",
        type=Path,
        default=DEFAULT_VISUALIZATION_DIR,
    )
    parser.add_argument("--heatmap-dir", type=Path, default=DEFAULT_HEATMAP_DIR)
    parser.add_argument("--trail-length", type=int, default=30)
    parser.add_argument("--max-visualized-tracks", type=int, default=10)
    parser.add_argument("--min-visualized-track-frames", type=int, default=15)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--video-id", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_visualizations(
        subset_path=resolve_project_path(args.subset),
        track_dir=resolve_project_path(args.track_dir),
        visualization_dir=resolve_project_path(args.visualization_dir),
        heatmap_dir=resolve_project_path(args.heatmap_dir),
        overwrite=args.overwrite,
        trail_length=args.trail_length,
        max_visualized_tracks=args.max_visualized_tracks,
        min_visualized_track_frames=args.min_visualized_track_frames,
        limit=args.limit,
        video_id=args.video_id,
    )


if __name__ == "__main__":
    main()
