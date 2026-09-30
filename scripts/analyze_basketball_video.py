from __future__ import annotations

import argparse
import re
from pathlib import Path

import cv2
import pandas as pd
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_SUBSET_PATH = ROOT / "data" / "subset.csv"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "analytics" / "tracks"
YOLO_MODEL_PATH = ROOT / "scripts" / "yolo11n.pt"

PERSON_CLASS_ID = 0
YOLO_CONFIDENCE = 0.20
YOLO_IMAGE_SIZE = 640


def safe_video_id(video_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", str(video_id)).strip("_")


def resolve_project_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else ROOT / path


def load_unique_videos(subset_path: Path) -> pd.DataFrame:
    subset = pd.read_csv(subset_path)
    required = {"video_id", "video_path"}
    missing = required - set(subset.columns)
    if missing:
        raise ValueError(f"Missing required subset columns: {sorted(missing)}")
    return subset[["video_id", "video_path"]].drop_duplicates("video_id")


def track_video(
    video_id: str,
    video_path: Path,
    yolo_model: YOLO,
    output_dir: Path,
    overwrite: bool = False,
    confidence: float = YOLO_CONFIDENCE,
    image_size: int = YOLO_IMAGE_SIZE,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{safe_video_id(video_id)}.csv"
    if output_path.exists() and not overwrite:
        print(f"Skipping existing tracks: {output_path}")
        return output_path

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        fps = 30.0

    rows: list[dict[str, float | int | str]] = []
    frame_index = 0

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            results = yolo_model.track(
                source=frame,
                persist=frame_index > 0,
                tracker="botsort.yaml",
                classes=[PERSON_CLASS_ID],
                conf=confidence,
                imgsz=image_size,
                verbose=False,
            )

            if results:
                boxes = results[0].boxes
                if boxes is not None and boxes.id is not None:
                    xyxy = boxes.xyxy.cpu().numpy()
                    confs = boxes.conf.cpu().numpy()
                    track_ids = boxes.id.cpu().numpy().astype(int)

                    for box, conf, track_id in zip(xyxy, confs, track_ids):
                        x1, y1, x2, y2 = [float(value) for value in box]
                        rows.append(
                            {
                                "video_id": video_id,
                                "frame_index": frame_index,
                                "timestamp_seconds": frame_index / fps,
                                "track_id": int(track_id),
                                "confidence": float(conf),
                                "x1": x1,
                                "y1": y1,
                                "x2": x2,
                                "y2": y2,
                                "center_x": (x1 + x2) / 2.0,
                                "center_y": (y1 + y2) / 2.0,
                            }
                        )

            frame_index += 1
    finally:
        capture.release()

    pd.DataFrame(
        rows,
        columns=[
            "video_id",
            "frame_index",
            "timestamp_seconds",
            "track_id",
            "confidence",
            "x1",
            "y1",
            "x2",
            "y2",
            "center_x",
            "center_y",
        ],
    ).to_csv(output_path, index=False)

    print(f"Saved {len(rows)} detections: {output_path}")
    return output_path


def run_tracking(
    subset_path: Path,
    output_dir: Path,
    overwrite: bool,
    limit: int | None,
    video_id: str | None,
) -> None:
    if not YOLO_MODEL_PATH.exists():
        raise FileNotFoundError(f"YOLO model not found: {YOLO_MODEL_PATH}")

    videos = load_unique_videos(subset_path)
    if video_id:
        videos = videos[videos["video_id"] == video_id]
        if videos.empty:
            raise ValueError(f"video_id not found in subset: {video_id}")
    if limit is not None:
        videos = videos.head(limit)

    print(f"Loading YOLO model: {YOLO_MODEL_PATH}")
    yolo_model = YOLO(str(YOLO_MODEL_PATH))

    for row in videos.itertuples(index=False):
        video_path = resolve_project_path(row.video_path)
        print(f"Tracking {row.video_id}")
        track_video(
            video_id=str(row.video_id),
            video_path=video_path,
            yolo_model=yolo_model,
            output_dir=output_dir,
            overwrite=overwrite,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Track Sports-QA basketball players with YOLO11 + BoT-SORT."
    )
    parser.add_argument("--subset", type=Path, default=DEFAULT_SUBSET_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--video-id", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_tracking(
        subset_path=resolve_project_path(args.subset),
        output_dir=resolve_project_path(args.output_dir),
        overwrite=args.overwrite,
        limit=args.limit,
        video_id=args.video_id,
    )


if __name__ == "__main__":
    main()
