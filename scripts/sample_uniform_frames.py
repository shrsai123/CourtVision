import csv
from pathlib import Path

import cv2
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SUBSET_PATH = ROOT / "data" / "subset.csv"
FRAME_DIR = ROOT / "data" / "uniform_frames"
MANIFEST_PATH = ROOT / "data" / "uniform_frame_manifest.csv"
NUM_FRAMES = 8


def resolve_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return ROOT / path


def sample_indices(frame_count: int) -> list[int]:
    if frame_count <= 0:
        raise ValueError("Video has no readable frames.")
    if frame_count == 1:
        return [0] * NUM_FRAMES
    return [
        round(i * (frame_count - 1) / (NUM_FRAMES - 1))
        for i in range(NUM_FRAMES)
    ]


def write_frame(video_path: Path, frame_index: int, target_path: Path) -> None:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok:
            raise ValueError(
                f"Could not read frame {frame_index} from {video_path}"
            )
        target_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(target_path), frame):
            raise ValueError(f"Could not write frame: {target_path}")
    finally:
        capture.release()


def main() -> None:
    subset = pd.read_csv(SUBSET_PATH)
    required_columns = {"video_id", "video_path"}
    missing_columns = required_columns - set(subset.columns)
    if missing_columns:
        raise ValueError(f"subset.csv is missing: {missing_columns}")

    rows = []
    unique_videos = (
        subset[["video_id", "video_path"]]
        .drop_duplicates()
        .sort_values("video_id")
    )

    for video in unique_videos.itertuples(index=False):
        video_path = resolve_path(video.video_path)
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        capture.release()

        for frame_order, frame_index in enumerate(sample_indices(frame_count)):
            safe_stem = Path(video.video_id).name
            frame_path = (
                FRAME_DIR
                / safe_stem
                / f"frame_{frame_order:02d}.jpg"
            )
            if not frame_path.exists():
                write_frame(video_path, frame_index, frame_path)
            rows.append(
                {
                    "video_id": video.video_id,
                    "frame_order": frame_order,
                    "frame_index": frame_index,
                    "frame_path": str(frame_path.relative_to(ROOT)),
                }
            )

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST_PATH.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "video_id",
                "frame_order",
                "frame_index",
                "frame_path",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} frames to {FRAME_DIR}")
    print(f"Wrote manifest to {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
