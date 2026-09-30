import argparse
import csv
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from transformers import CLIPModel, CLIPProcessor
from ultralytics import YOLO

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import sample_guided_frames as base


ROOT = base.ROOT

DEFAULT_SUBSET_PATH = base.DEFAULT_SUBSET_PATH
DEFAULT_FRAME_DIR = ROOT / "data" / "guided_clip_frames"
DEFAULT_MANIFEST_PATH = ROOT / "data" / "guided_clip_frame_manifest.csv"

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
CLIP_BATCH_SIZE = 32

MOTION_WEIGHT = 0.40
PERSON_WEIGHT = 0.20
BALL_WEIGHT = 0.10
CLIP_WEIGHT = 0.30


def add_clip_image_embeddings(
    candidates: list[dict],
    clip_model,
    clip_processor,
    device: str,
    batch_size: int,
) -> None:
    """Compute normalized CLIP image embeddings for candidate frames."""
    all_embeddings = []

    for start in range(0, len(candidates), batch_size):
        batch_candidates = candidates[start:start + batch_size]
        rgb_images = [
            cv2.cvtColor(candidate["frame"], cv2.COLOR_BGR2RGB)
            for candidate in batch_candidates
        ]

        inputs = clip_processor(
            images=rgb_images,
            return_tensors="pt",
        )
        pixel_values = inputs["pixel_values"].to(device)

        with torch.inference_mode():
            image_features = clip_model.get_image_features(
                pixel_values=pixel_values
            )

        image_features = image_features / image_features.norm(
            dim=-1,
            keepdim=True,
        )
        all_embeddings.append(image_features.cpu())

    embeddings = torch.cat(all_embeddings, dim=0)

    for candidate, embedding in zip(candidates, embeddings):
        candidate["clip_image_embedding"] = embedding


def calculate_clip_scores(
    candidates: list[dict],
    question: str,
    clip_model,
    clip_processor,
    device: str,
) -> None:
    """Score each candidate frame against the QA question with CLIP."""
    clip_text = (
        "a basketball video frame relevant to the question: "
        + question
    )

    text_inputs = clip_processor(
        text=[clip_text],
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    text_inputs = {
        key: value.to(device)
        for key, value in text_inputs.items()
    }

    with torch.inference_mode():
        text_features = clip_model.get_text_features(**text_inputs)

    text_features = text_features / text_features.norm(
        dim=-1,
        keepdim=True,
    )

    image_features = torch.stack(
        [
            candidate["clip_image_embedding"]
            for candidate in candidates
        ]
    ).to(device)

    similarities = (
        image_features
        @ text_features[0]
    ).detach().cpu().tolist()

    for candidate, similarity in zip(candidates, similarities):
        candidate["clip_score"] = float(similarity)


def calculate_combined_scores(candidates: list[dict]) -> None:
    base.normalize_candidate_feature(
        candidates,
        "motion_score",
        "normalized_motion_score",
    )
    base.normalize_candidate_feature(
        candidates,
        "person_score",
        "normalized_person_score",
    )
    base.normalize_candidate_feature(
        candidates,
        "ball_score",
        "normalized_ball_score",
    )
    base.normalize_candidate_feature(
        candidates,
        "clip_score",
        "normalized_clip_score",
    )

    for candidate in candidates:
        candidate["combined_score"] = (
            MOTION_WEIGHT * candidate["normalized_motion_score"]
            + PERSON_WEIGHT * candidate["normalized_person_score"]
            + BALL_WEIGHT * candidate["normalized_ball_score"]
            + CLIP_WEIGHT * candidate["normalized_clip_score"]
        )


def create_montage(
    selected_frames: list[dict],
    fps: float,
    output_path: Path,
) -> None:
    """Create a 4-by-2 visualization of selected CLIP-guided frames."""
    tile_width = 320
    image_height = 180
    label_height = 45
    columns = 4

    tiles = []

    for order, candidate in enumerate(selected_frames, start=1):
        frame = candidate["frame"]
        frame_index = candidate["frame_index"]
        motion = candidate["normalized_motion_score"]
        person = candidate["normalized_person_score"]
        ball = candidate["normalized_ball_score"]
        clip = candidate["normalized_clip_score"]
        combined = candidate["combined_score"]

        resized = cv2.resize(
            frame,
            (tile_width, image_height),
            interpolation=cv2.INTER_AREA,
        )
        tile = np.zeros(
            (image_height + label_height, tile_width, 3),
            dtype=np.uint8,
        )
        tile[:image_height] = resized

        label = (
            f"{order}: idx={frame_index} "
            f"M={motion:.2f} "
            f"P={person:.2f} "
            f"B={ball:.2f} "
            f"C={clip:.2f} "
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
    for start in range(0, len(tiles), columns):
        row_tiles = tiles[start:start + columns]

        while len(row_tiles) < columns:
            row_tiles.append(np.zeros_like(tiles[0]))

        rows.append(np.hstack(row_tiles))

    montage = np.vstack(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not cv2.imwrite(str(output_path), montage):
        raise ValueError(f"Could not save montage: {output_path}")


def load_qa_rows(subset_path: Path) -> list[dict]:
    subset = pd.read_csv(subset_path)

    required_columns = {
        "qa_id",
        "video_id",
        "video_path",
        "question",
    }
    missing_columns = required_columns - set(subset.columns)

    if missing_columns:
        raise ValueError(
            f"subset.csv is missing columns: {sorted(missing_columns)}"
        )

    qa_rows = subset[
        [
            "qa_id",
            "video_id",
            "video_path",
            "question",
        ]
    ].copy()
    qa_rows["qa_id"] = qa_rows["qa_id"].astype(str)
    qa_rows["question"] = qa_rows["question"].fillna("").astype(str)

    print("QA pairs:", len(qa_rows))
    print("Unique videos:", qa_rows["video_id"].nunique())

    return qa_rows.to_dict(orient="records")


def process_qa(
    qa_id: str,
    video_id: str,
    question: str,
    candidates: list[dict],
    fps: float,
    total_frames: int,
    output_root: Path,
    num_frames: int,
    minimum_gap: int,
    overwrite: bool,
    clip_model,
    clip_processor,
    clip_device: str,
    clip_model_name: str,
    candidate_stride: int,
) -> list[dict]:
    qa_candidates = [
        dict(candidate)
        for candidate in candidates
    ]

    calculate_clip_scores(
        candidates=qa_candidates,
        question=question,
        clip_model=clip_model,
        clip_processor=clip_processor,
        device=clip_device,
    )
    calculate_combined_scores(qa_candidates)

    selected_frames = base.select_diverse_frames(
        candidates=qa_candidates,
        num_frames=num_frames,
        minimum_gap=minimum_gap,
    )

    safe_video_id = video_id.replace("/", "__")
    safe_qa_id = str(qa_id).replace("/", "_")
    qa_output_dir = output_root / safe_video_id / f"qa_{safe_qa_id}"

    if overwrite and qa_output_dir.exists():
        shutil.rmtree(qa_output_dir)

    qa_output_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []

    for frame_order, candidate in enumerate(selected_frames, start=1):
        frame_index = candidate["frame_index"]
        timestamp_seconds = frame_index / fps
        frame_filename = (
            f"frame_{frame_order:02d}_"
            f"idx_{frame_index:06d}.jpg"
        )
        frame_path = qa_output_dir / frame_filename

        if not cv2.imwrite(str(frame_path), candidate["frame"]):
            raise ValueError(f"Could not save frame: {frame_path}")

        manifest_rows.append(
            {
                "qa_id": qa_id,
                "video_id": video_id,
                "question": question,
                "frame_order": frame_order,
                "frame_index": frame_index,
                "timestamp_seconds": round(timestamp_seconds, 4),
                "frame_path": str(frame_path.relative_to(ROOT)),
                "motion_score": round(candidate["motion_score"], 6),
                "normalized_motion_score": round(
                    candidate["normalized_motion_score"],
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
                "person_score": round(candidate["person_score"], 6),
                "normalized_person_score": round(
                    candidate["normalized_person_score"],
                    6,
                ),
                "ball_count": candidate["ball_count"],
                "ball_score": round(candidate["ball_score"], 6),
                "normalized_ball_score": round(
                    candidate["normalized_ball_score"],
                    6,
                ),
                "clip_score": round(candidate["clip_score"], 6),
                "normalized_clip_score": round(
                    candidate["normalized_clip_score"],
                    6,
                ),
                "combined_score": round(candidate["combined_score"], 6),
                "motion_weight": MOTION_WEIGHT,
                "person_weight": PERSON_WEIGHT,
                "ball_weight": BALL_WEIGHT,
                "clip_weight": CLIP_WEIGHT,
                "clip_model": clip_model_name,
                "selection_method": "guided_motion_yolo_clip",
                "candidate_stride": candidate_stride,
                "minimum_frame_gap": minimum_gap,
                "video_total_frames": total_frames,
                "video_fps": round(fps, 4),
            }
        )

    create_montage(
        selected_frames=selected_frames,
        fps=fps,
        output_path=qa_output_dir / "guided_motion_yolo_clip_montage.jpg",
    )

    selected_indices = [
        candidate["frame_index"]
        for candidate in selected_frames
    ]
    print(
        f"Processed {qa_id} ({video_id}): "
        f"{len(qa_candidates)} candidates -> "
        f"{len(selected_frames)} selected"
    )
    print("Selected indices:", selected_indices)

    return manifest_rows


def write_manifest(manifest_path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("No guided-frame records were generated.")

    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    with manifest_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select question-aware Motion+YOLO+CLIP frames from "
            "Sports-QA basketball videos."
        )
    )
    parser.add_argument("--subset", type=Path, default=DEFAULT_SUBSET_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_FRAME_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument(
        "--num-frames",
        type=int,
        default=base.DEFAULT_NUM_FRAMES,
    )
    parser.add_argument(
        "--candidate-stride",
        type=int,
        default=base.DEFAULT_CANDIDATE_STRIDE,
    )
    parser.add_argument(
        "--minimum-gap",
        type=int,
        default=base.DEFAULT_MIN_FRAME_GAP,
    )
    parser.add_argument(
        "--clip-model-name",
        default=CLIP_MODEL_NAME,
        help="Hugging Face CLIP model ID or local model path.",
    )
    parser.add_argument(
        "--clip-batch-size",
        type=int,
        default=CLIP_BATCH_SIZE,
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N QA pairs.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    if args.num_frames <= 0:
        raise ValueError("--num-frames must be greater than zero.")
    if args.candidate_stride <= 0:
        raise ValueError("--candidate-stride must be greater than zero.")
    if args.minimum_gap < 0:
        raise ValueError("--minimum-gap cannot be negative.")
    if args.clip_batch_size <= 0:
        raise ValueError("--clip-batch-size must be greater than zero.")

    subset_path = base.make_absolute(args.subset)
    output_dir = base.make_absolute(args.output_dir)
    manifest_path = base.make_absolute(args.manifest)

    if not subset_path.exists():
        raise FileNotFoundError(f"Subset file not found: {subset_path}")
    if not base.YOLO_MODEL_PATH.exists():
        raise FileNotFoundError(f"YOLO model not found: {base.YOLO_MODEL_PATH}")

    print(f"Loading YOLO model: {base.YOLO_MODEL_PATH}")
    yolo_model = YOLO(str(base.YOLO_MODEL_PATH))
    print("YOLO model loaded.")

    clip_device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading CLIP model: {args.clip_model_name} on {clip_device}")
    clip_processor = CLIPProcessor.from_pretrained(args.clip_model_name)
    clip_model = (
        CLIPModel.from_pretrained(
            args.clip_model_name,
            use_safetensors=True,
        )
        .to(clip_device)
    )
    clip_model.eval()
    print("CLIP model loaded.")

    qa_rows = load_qa_rows(subset_path)

    if args.limit is not None:
        qa_rows = qa_rows[:args.limit]

    all_manifest_rows = []
    failures = []
    video_cache = {}

    for position, qa_row in enumerate(qa_rows, start=1):
        qa_id = str(qa_row["qa_id"])
        video_id = qa_row["video_id"]
        video_path = base.resolve_path(qa_row["video_path"])
        question = qa_row["question"]

        print(
            f"\n[{position}/{len(qa_rows)}] "
            f"qa_id={qa_id} video={video_id}"
        )

        if not video_path.exists():
            error = f"Video not found: {video_path}"
            print(error)
            failures.append(
                {
                    "qa_id": qa_id,
                    "video_id": video_id,
                    "error": error,
                }
            )
            continue

        try:
            if video_id not in video_cache:
                print(f"Collecting candidates for {video_id}")
                candidates, fps, total_frames = base.collect_candidate_frames(
                    video_path=video_path,
                    candidate_stride=args.candidate_stride,
                    yolo_model=yolo_model,
                    yolo_confidence=base.YOLO_CONFIDENCE,
                    yolo_image_size=base.YOLO_IMAGE_SIZE,
                )
                add_clip_image_embeddings(
                    candidates=candidates,
                    clip_model=clip_model,
                    clip_processor=clip_processor,
                    device=clip_device,
                    batch_size=args.clip_batch_size,
                )
                video_cache[video_id] = {
                    "candidates": candidates,
                    "fps": fps,
                    "total_frames": total_frames,
                }

            cached = video_cache[video_id]
            rows = process_qa(
                qa_id=qa_id,
                video_id=video_id,
                question=question,
                candidates=cached["candidates"],
                fps=cached["fps"],
                total_frames=cached["total_frames"],
                output_root=output_dir,
                num_frames=args.num_frames,
                minimum_gap=args.minimum_gap,
                overwrite=args.overwrite,
                clip_model=clip_model,
                clip_processor=clip_processor,
                clip_device=clip_device,
                clip_model_name=args.clip_model_name,
                candidate_stride=args.candidate_stride,
            )
            all_manifest_rows.extend(rows)

        except Exception as error:
            print(f"Failed {qa_id} ({video_id}): {error}")
            failures.append(
                {
                    "qa_id": qa_id,
                    "video_id": video_id,
                    "error": str(error),
                }
            )

    write_manifest(manifest_path=manifest_path, rows=all_manifest_rows)

    print("\nMotion+YOLO+CLIP sampling completed.")
    print("Manifest:", manifest_path)
    print("Frames:", output_dir)
    print("QA pairs processed:", len(all_manifest_rows) // args.num_frames)
    print("Unique videos cached:", len(video_cache))
    print("Failures:", len(failures))

    if failures:
        print("\nFailed QA pairs:")
        for failure in failures:
            print(
                failure["video_id"],
                failure["qa_id"],
                "->",
                failure["error"],
            )


if __name__ == "__main__":
    main()
