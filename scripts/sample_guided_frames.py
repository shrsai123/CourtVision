import argparse
import csv
import shutil
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_SUBSET_PATH = ROOT / "data" / "subset.csv"
DEFAULT_FRAME_DIR = ROOT / "data" / "guided_frames"
DEFAULT_MANIFEST_PATH = (
    ROOT / "data" / "guided_frame_manifest.csv"
)

DEFAULT_NUM_FRAMES = 8
DEFAULT_CANDIDATE_STRIDE = 5
DEFAULT_MIN_FRAME_GAP = 15

# Motion is calculated on smaller grayscale images for speed.
MOTION_WIDTH = 320
MOTION_HEIGHT = 180

YOLO_MODEL_PATH = ROOT / "scripts" / "yolo11n.pt"

YOLO_CONFIDENCE = 0.20
YOLO_IMAGE_SIZE = 640

PERSON_CLASS_ID = 0
SPORTS_BALL_CLASS_ID = 32

MOTION_WEIGHT = 0.57
PERSON_WEIGHT = 0.29
BALL_WEIGHT = 0.14


def resolve_path(path_value: str) -> Path:
    """Convert a relative CSV path into an absolute path."""
    path = Path(path_value)

    if path.is_absolute():
        return path

    return ROOT / path


def calculate_motion_score(
    previous_gray: np.ndarray,
    current_gray: np.ndarray,
) -> float:
    """
    Calculate mean absolute pixel difference between two frames.

    Higher scores indicate greater visual change or motion.
    """
    difference = cv2.absdiff(
        previous_gray,
        current_gray,
    )

    return float(difference.mean())

def calculate_yolo_scores(
    frame: np.ndarray,
    yolo_model,
    confidence_threshold: float,
    image_size: int,
) -> dict:
    """
    Detect people and sports balls in one candidate frame.

    Returns raw scores that will later be normalized
    across all candidates from the same video.
    """
    results = yolo_model.predict(
        source=frame,
        conf=confidence_threshold,
        imgsz=image_size,
        classes=[
            PERSON_CLASS_ID,
            SPORTS_BALL_CLASS_ID,
        ],
        verbose=False,
    )

    result = results[0]
    boxes = result.boxes

    if boxes is None or len(boxes) == 0:
        return {
            "person_count": 0,
            "person_confidence_sum": 0.0,
            "person_area_ratio": 0.0,
            "person_score": 0.0,
            "ball_count": 0,
            "ball_score": 0.0,
        }

    class_ids = (
        boxes.cls
        .detach()
        .cpu()
        .numpy()
        .astype(int)
    )

    confidences = (
        boxes.conf
        .detach()
        .cpu()
        .numpy()
    )

    coordinates = (
        boxes.xyxy
        .detach()
        .cpu()
        .numpy()
    )

    frame_height, frame_width = frame.shape[:2]
    frame_area = float(frame_height * frame_width)

    person_confidences = []
    person_total_area = 0.0
    ball_confidences = []

    for class_id, confidence, box in zip(
        class_ids,
        confidences,
        coordinates,
    ):
        x1, y1, x2, y2 = box

        if class_id == PERSON_CLASS_ID:
            person_confidences.append(float(confidence))

            box_width = max(0.0, float(x2 - x1))
            box_height = max(0.0, float(y2 - y1))

            person_total_area += box_width * box_height

        elif class_id == SPORTS_BALL_CLASS_ID:
            ball_confidences.append(float(confidence))

    person_count = len(person_confidences)
    ball_count = len(ball_confidences)

    person_confidence_sum = sum(person_confidences)

    person_area_ratio = min(
        person_total_area / frame_area,
        1.0,
    )

    # Combines detection confidence with how much of the
    # image is occupied by players.
    person_score = (
        person_confidence_sum
        + person_area_ratio
    )

    # Normally only one basketball should be important.
    ball_score = (
        max(ball_confidences)
        if ball_confidences
        else 0.0
    )

    return {
        "person_count": person_count,
        "person_confidence_sum": float(
            person_confidence_sum
        ),
        "person_area_ratio": float(
            person_area_ratio
        ),
        "person_score": float(person_score),
        "ball_count": ball_count,
        "ball_score": float(ball_score),
    }

def prepare_motion_frame(
    frame: np.ndarray,
) -> np.ndarray:
    """
    Convert a video frame to a smaller grayscale image.

    Resizing makes motion scoring faster and reduces noise caused
    by different original video resolutions.
    """
    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY,
    )

    gray = cv2.resize(
        gray,
        (MOTION_WIDTH, MOTION_HEIGHT),
        interpolation=cv2.INTER_AREA,
    )

    # A small blur reduces compression noise.
    gray = cv2.GaussianBlur(
        gray,
        (5, 5),
        0,
    )

    return gray


def collect_candidate_frames(
    video_path: Path,
    candidate_stride: int,
    yolo_model,
    yolo_confidence: float,
    yolo_image_size: int,
) :
    """
    Read a video sequentially and keep every Nth frame.

    Returns:
        candidates:
            Candidate frames and their motion scores.
        fps:
            Frames per second.
        total_frames:
            Number of readable frames in the video.
    """
    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        raise ValueError(
            f"Could not open video: {video_path}"
        )

    fps = float(
        capture.get(cv2.CAP_PROP_FPS)
    )

    if fps <= 0:
        print(
            f"Warning: Invalid FPS for {video_path.name}. "
            "Using 30 FPS."
        )
        fps = 30.0

    candidates = []

    previous_candidate_gray = None
    last_frame = None
    last_frame_index = -1

    frame_index = 0

    while True:
        success, frame = capture.read()

        if not success:
            break

        last_frame = frame.copy()
        last_frame_index = frame_index

        if frame_index % candidate_stride == 0:
            current_gray = prepare_motion_frame(frame)

            if previous_candidate_gray is None:
                motion_score = 0.0
            else:
                motion_score = calculate_motion_score(
                    previous_gray=previous_candidate_gray,
                    current_gray=current_gray,
                )
            yolo_scores = calculate_yolo_scores(
        frame=frame,
        yolo_model=yolo_model,
        confidence_threshold=yolo_confidence,
        image_size=yolo_image_size,
    )


            candidates.append(
                {
                    "frame_index": frame_index,
                    "motion_score": motion_score,
                     "person_count": yolo_scores[
                "person_count"
            ],
            "person_confidence_sum": yolo_scores[
                "person_confidence_sum"
            ],
            "person_area_ratio": yolo_scores[
                "person_area_ratio"
            ],
            "person_score": yolo_scores[
                "person_score"
            ],
            "ball_count": yolo_scores[
                "ball_count"
            ],
            "ball_score": yolo_scores[
                "ball_score"
            ],
                    "frame": frame.copy(),
                }
            )

            previous_candidate_gray = current_gray

        frame_index += 1

    capture.release()

    total_frames = frame_index

    if total_frames == 0:
        raise ValueError(
            f"No readable frames found in {video_path}"
        )

    # Include the last frame when it was not already selected
    # as a candidate.
    if (
        last_frame is not None
        and candidates
        and candidates[-1]["frame_index"] != last_frame_index
    ):
        current_gray = prepare_motion_frame(last_frame)

        motion_score = calculate_motion_score(
            previous_gray=previous_candidate_gray,
            current_gray=current_gray,
        )

        yolo_scores = calculate_yolo_scores(
    frame=last_frame,
    yolo_model=yolo_model,
    confidence_threshold=yolo_confidence,
    image_size=yolo_image_size,
)
        candidates.append({
            "frame_index": last_frame_index,
            "motion_score": motion_score,
             "person_count": yolo_scores[
                "person_count"
            ],
            "person_confidence_sum": yolo_scores[
                "person_confidence_sum"
            ],
            "person_area_ratio": yolo_scores[
                "person_area_ratio"
            ],
            "person_score": yolo_scores[
                "person_score"
            ],
            "ball_count": yolo_scores[
                "ball_count"
            ],
            "ball_score": yolo_scores[
                "ball_score"
            ],
            "frame": last_frame,
        })

    if not candidates:
        raise ValueError(
            f"No candidate frames found in {video_path}"
        )

    return candidates, fps, total_frames

def normalize_candidate_feature(
    candidates: list[dict],
    source_key: str,
    normalized_key: str,
) -> None:
    """
    Normalize one candidate feature between 0 and 1
    within a single video.
    """
    values = np.array(
        [
            candidate[source_key]
            for candidate in candidates
        ],
        dtype=np.float32,
    )

    minimum = float(values.min())
    maximum = float(values.max())

    denominator = maximum - minimum

    for candidate in candidates:
        if denominator <= 1e-8:
            normalized_value = 0.0
        else:
            normalized_value = (
                candidate[source_key] - minimum
            ) / denominator

        candidate[normalized_key] = float(
            normalized_value
        )

def calculate_combined_scores(
    candidates: list[dict],
) -> None:

    normalize_candidate_feature(
        candidates,
        "motion_score",
        "normalized_motion_score",
    )

    normalize_candidate_feature(
        candidates,
        "person_score",
        "normalized_person_score",
    )

    normalize_candidate_feature(
        candidates,
        "ball_score",
        "normalized_ball_score",
    )

    for candidate in candidates:

        candidate["combined_score"] = (
            MOTION_WEIGHT
            * candidate[
                "normalized_motion_score"
            ]

            + PERSON_WEIGHT
            * candidate[
                "normalized_person_score"
            ]

            + BALL_WEIGHT
            * candidate[
                "normalized_ball_score"
            ]
        )

def select_diverse_frames(
    candidates: list[dict],
    num_frames: int,
    minimum_gap: int,
) -> list[dict]:
    """
    Select high-motion frames while avoiding neighboring frames.

    First pass:
        Enforce the minimum temporal gap.

    Second pass:
        Fill remaining positions with the next highest-scoring
        frames if the gap constraint was too restrictive.
    """
    if len(candidates) < num_frames:
        raise ValueError(
            f"Only {len(candidates)} candidate frames are "
            f"available, but {num_frames} are required."
        )

    ranked_candidates = sorted(
        candidates,
        key=lambda candidate: candidate[
             "combined_score"
        ],
        reverse=True,
    )

    selected = []
    selected_indices = set()

    # First pass: enforce temporal separation.
    for candidate in ranked_candidates:
        frame_index = candidate["frame_index"]

        sufficiently_distant = all(
            abs(
                frame_index
                - selected_candidate["frame_index"]
            ) >= minimum_gap
            for selected_candidate in selected
        )

        if sufficiently_distant:
            selected.append(candidate)
            selected_indices.add(frame_index)

        if len(selected) == num_frames:
            break

    # Second pass: fill remaining slots if the minimum gap
    # prevented us from reaching the required frame count.
    if len(selected) < num_frames:
        for candidate in ranked_candidates:
            frame_index = candidate["frame_index"]

            if frame_index in selected_indices:
                continue

            selected.append(candidate)
            selected_indices.add(frame_index)

            if len(selected) == num_frames:
                break

    if len(selected) != num_frames:
        raise ValueError(
            f"Selected only {len(selected)} frames; "
            f"expected {num_frames}."
        )

    # Qwen must receive the selected frames chronologically.
    return sorted(
        selected,
        key=lambda candidate: candidate["frame_index"],
    )


def create_montage(
    selected_frames: list[dict],
    fps: float,
    output_path: Path,
) -> None:
    """Create a 4-by-2 visualization of the selected frames."""
    tile_width = 320
    image_height = 180
    label_height = 45
    columns = 4

    tiles = []

    for order, candidate in enumerate(
        selected_frames,
        start=1,
    ):
        frame = candidate["frame"]
        frame_index = candidate["frame_index"]
        motion = candidate["normalized_motion_score"]
        person = candidate["normalized_person_score"]
        ball = candidate["normalized_ball_score"]
        combined = candidate["combined_score"]

        resized = cv2.resize(
            frame,
            (tile_width, image_height),
            interpolation=cv2.INTER_AREA,
        )

        tile = np.zeros(
            (
                image_height + label_height,
                tile_width,
                3,
            ),
            dtype=np.uint8,
        )

        tile[:image_height] = resized

        timestamp = frame_index / fps

        label = (
    f"{order}: idx={frame_index} "
    f"M={motion:.2f} "
    f"P={person:.2f} "
    f"B={ball:.2f} "
    f"S={combined:.2f}"
)

        cv2.putText(
            tile,
            label,
            (6, image_height + 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        tiles.append(tile)

    rows = []

    for start in range(
        0,
        len(tiles),
        columns,
    ):
        row_tiles = tiles[start:start + columns]

        while len(row_tiles) < columns:
            row_tiles.append(
                np.zeros_like(tiles[0])
            )

        rows.append(np.hstack(row_tiles))

    montage = np.vstack(rows)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not cv2.imwrite(
        str(output_path),
        montage,
    ):
        raise ValueError(
            f"Could not save montage: {output_path}"
        )


def process_video(
    video_id: str,
    video_path: Path,
    output_root: Path,
    num_frames: int,
    candidate_stride: int,
    minimum_gap: int,
    overwrite: bool,
    yolo_model,
    yolo_confidence: float,
    yolo_image_size: int,
):
    """
    Select, save, and describe guided frames for one video.
    """
    candidates, fps, total_frames = collect_candidate_frames(
        video_path=video_path,
        candidate_stride=candidate_stride,
        yolo_model=yolo_model,
        yolo_confidence=yolo_confidence,
        yolo_image_size=yolo_image_size,
    )

    calculate_combined_scores(candidates)
    person_candidates = sum(
        candidate["person_count"] > 0
        for candidate in candidates
    )
    ball_candidates = sum(
        candidate["ball_count"] > 0
        for candidate in candidates
    )
    print(
        f"YOLO detections: "
        f"people={person_candidates}/{len(candidates)}, "
        f"ball={ball_candidates}/{len(candidates)}"
    )

    selected_frames = select_diverse_frames(
        candidates=candidates,
        num_frames=num_frames,
        minimum_gap=minimum_gap,
    )

    safe_video_id = video_id.replace("/", "__")
    video_output_dir = output_root / safe_video_id

    if overwrite and video_output_dir.exists():
        shutil.rmtree(video_output_dir)

    video_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest_rows = []

    for frame_order, candidate in enumerate(
        selected_frames,
        start=1,
    ):
        frame_index = candidate["frame_index"]
        timestamp_seconds = frame_index / fps

        frame_filename = (
            f"frame_{frame_order:02d}_"
            f"idx_{frame_index:06d}.jpg"
        )

        frame_path = (
            video_output_dir / frame_filename
        )

        if not cv2.imwrite(
            str(frame_path),
            candidate["frame"],
        ):
            raise ValueError(
                f"Could not save frame: {frame_path}"
            )

        manifest_rows.append(
            {
                "video_id": video_id,
                "frame_order": frame_order,
                "frame_index": frame_index,
                "timestamp_seconds": round(
                    timestamp_seconds,
                    4,
                ),
                "frame_path": str(
                    frame_path.relative_to(ROOT)
                ),
                "motion_score": round(
                    candidate["motion_score"],
                    6,
                ),
                "normalized_motion_score": round(
                    candidate[
                        "normalized_motion_score"
                    ],
                    6,
                ),
                "person_count": candidate["person_count"],
                "person_confidence_sum": round(
                    candidate["person_confidence_sum"],
                    6,
                ),
                "person_area_ratio": round(
                    candidate["person_area_ratio"],
                    6,
                ),
                "person_score": round(
                    candidate["person_score"],
                    6,
                ),
                "normalized_person_score": round(
                    candidate["normalized_person_score"],
                    6,
                ),
                "ball_count": candidate["ball_count"],
                "ball_score": round(
                    candidate["ball_score"],
                    6,
                ),
                "normalized_ball_score": round(
                    candidate["normalized_ball_score"],
                    6,
                ),
                "combined_score": round(
                    candidate["combined_score"],
                    6,
                ),
                "motion_weight": MOTION_WEIGHT,
                "person_weight": PERSON_WEIGHT,
                "ball_weight": BALL_WEIGHT,
                "selection_method": "guided_motion_yolo",
                "candidate_stride": candidate_stride,
                "minimum_frame_gap": minimum_gap,
                "video_total_frames": total_frames,
                "video_fps": round(fps, 4),
            }
        )

    montage_path = (
        video_output_dir / "guided_motion_yolo_montage.jpg"
    )

    create_montage(
        selected_frames=selected_frames,
        fps=fps,
        output_path=montage_path,
    )

    selected_indices = [
        candidate["frame_index"]
        for candidate in selected_frames
    ]

    print(
        f"Processed {video_id}: "
        f"{len(candidates)} candidates â†’ "
        f"{len(selected_frames)} selected"
    )
    print("Selected indices:", selected_indices)

    return manifest_rows


def load_unique_videos(
    subset_path: Path,
) -> list[dict]:
    """Load each unique video once from subset.csv."""
    subset = pd.read_csv(subset_path)

    required_columns = {
        "video_id",
        "video_path",
    }

    missing_columns = (
        required_columns - set(subset.columns)
    )

    if missing_columns:
        raise ValueError(
            f"subset.csv is missing columns: "
            f"{sorted(missing_columns)}"
        )

    unique_videos = (
        subset[
            [
                "video_id",
                "video_path",
            ]
        ]
        .drop_duplicates(subset=["video_id"])
        .sort_values("video_id")
        .reset_index(drop=True)
    )

    print("QA pairs:", len(subset))
    print("Unique videos:", len(unique_videos))

    return unique_videos.to_dict(
        orient="records"
    )


def write_manifest(
    manifest_path: Path,
    rows: list[dict],
) -> None:
    """Save selected-frame metadata as CSV."""
    if not rows:
        raise ValueError(
            "No guided-frame records were generated."
        )

    manifest_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with manifest_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(rows)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select motion-guided frames from "
            "Sports-QA basketball videos."
        )
    )

    parser.add_argument(
        "--subset",
        type=Path,
        default=DEFAULT_SUBSET_PATH,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_FRAME_DIR,
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
    )

    parser.add_argument(
        "--num-frames",
        type=int,
        default=DEFAULT_NUM_FRAMES,
    )

    parser.add_argument(
        "--candidate-stride",
        type=int,
        default=DEFAULT_CANDIDATE_STRIDE,
    )

    parser.add_argument(
        "--minimum-gap",
        type=int,
        default=DEFAULT_MIN_FRAME_GAP,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N unique videos.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


def make_absolute(path: Path) -> Path:
    if path.is_absolute():
        return path

    return ROOT / path


def main() -> None:
    args = parse_arguments()

    if args.num_frames <= 0:
        raise ValueError(
            "--num-frames must be greater than zero."
        )

    if args.candidate_stride <= 0:
        raise ValueError(
            "--candidate-stride must be greater than zero."
        )

    if args.minimum_gap < 0:
        raise ValueError(
            "--minimum-gap cannot be negative."
        )

    subset_path = make_absolute(args.subset)
    output_dir = make_absolute(args.output_dir)
    manifest_path = make_absolute(args.manifest)

    if not subset_path.exists():
        raise FileNotFoundError(
            f"Subset file not found: {subset_path}"
        )
    if not YOLO_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"YOLO model not found: {YOLO_MODEL_PATH}"
        )

    print(f"Loading YOLO model: {YOLO_MODEL_PATH}")
    
    yolo_model = YOLO(str(YOLO_MODEL_PATH))
    
    print("YOLO model loaded.")

    videos = load_unique_videos(subset_path)

    if args.limit is not None:
        videos = videos[:args.limit]

    all_manifest_rows = []
    failures = []

    for position, video in enumerate(
        videos,
        start=1,
    ):
        video_id = video["video_id"]
        video_path = resolve_path(
            video["video_path"]
        )

        print(
            f"\n[{position}/{len(videos)}] "
            f"{video_id}"
        )

        if not video_path.exists():
            error = (
                f"Video not found: {video_path}"
            )
            print(error)
            failures.append(
                {
                    "video_id": video_id,
                    "error": error,
                }
            )
            continue

        try:
            rows = process_video(
                video_id=video_id,
                video_path=video_path,
                output_root=output_dir,
                num_frames=args.num_frames,
                candidate_stride=args.candidate_stride,
                minimum_gap=args.minimum_gap,
                overwrite=args.overwrite,
                yolo_model=yolo_model,
                yolo_confidence=YOLO_CONFIDENCE,
                yolo_image_size=YOLO_IMAGE_SIZE,
            )

            all_manifest_rows.extend(rows)

        except Exception as error:
            print(
                f"Failed {video_id}: {error}"
            )

            failures.append(
                {
                    "video_id": video_id,
                    "error": str(error),
                }
            )

    write_manifest(
        manifest_path=manifest_path,
        rows=all_manifest_rows,
    )

    print("\nMotion + YOLO guided sampling completed.")
    print("Manifest:", manifest_path)
    print("Frames:", output_dir)
    print(
        "Videos processed:",
        len(all_manifest_rows) // args.num_frames,
    )
    print("Failures:", len(failures))

    if failures:
        print("\nFailed videos:")

        for failure in failures:
            print(
                failure["video_id"],
                "->",
                failure["error"],
            )




if __name__ == "__main__":
    main()
