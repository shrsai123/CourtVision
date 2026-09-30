import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True",
)

import pandas as pd
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_guided_frames as base

ROOT = base.ROOT
SUBSET_PATH = base.SUBSET_PATH
MANIFEST_PATH = ROOT / "data" / "guided_clip_frame_manifest.csv"
OUTPUT_DIR = base.OUTPUT_DIR
OUTPUT_PATH = OUTPUT_DIR / "guided_motion_yolo_clip_predictions.jsonl"

DEFAULT_MODEL_NAME = base.DEFAULT_MODEL_NAME
DEFAULT_MAX_NEW_TOKENS = base.DEFAULT_MAX_NEW_TOKENS
DEFAULT_IMAGE_WIDTH = base.DEFAULT_IMAGE_WIDTH
DEFAULT_IMAGE_HEIGHT = base.DEFAULT_IMAGE_HEIGHT


def load_guided_frames(
    manifest: pd.DataFrame,
    num_frames: int,
) -> dict[str, list[Path]]:
    if "qa_id" not in manifest.columns:
        raise ValueError(
            "CLIP-guided manifest is missing qa_id. "
            "Regenerate it with sample_guided_frames_clip.py."
        )

    manifest["qa_id"] = manifest["qa_id"].astype(str)
    frame_paths_by_qa = {}

    for qa_id, qa_rows in manifest.groupby("qa_id"):
        qa_rows = qa_rows.sort_values("frame_order")
        qa_rows = base.select_frame_rows(
            video_rows=qa_rows,
            num_frames=num_frames,
        )

        frame_paths = [
            base.resolve_path(path)
            for path in qa_rows["frame_path"].tolist()
        ]

        if len(frame_paths) != num_frames:
            raise ValueError(
                f"{qa_id} has {len(frame_paths)} frames; "
                f"expected exactly {num_frames}."
            )

        missing = [
            path for path in frame_paths if not path.exists()
        ]

        if missing:
            raise FileNotFoundError(
                f"Missing frames for {qa_id}: {missing}"
            )

        frame_paths_by_qa[str(qa_id)] = frame_paths

    return frame_paths_by_qa


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Motion+YOLO+CLIP guided Sports-QA pipeline."
        )
    )
    parser.add_argument(
        "--model-name",
        default=os.environ.get(
            "COURTVISION_MODEL_NAME",
            DEFAULT_MODEL_NAME,
        ),
        help="Hugging Face model ID or local model path.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=int(
            os.environ.get(
                "COURTVISION_MAX_NEW_TOKENS",
                DEFAULT_MAX_NEW_TOKENS,
            )
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of unfinished examples to run.",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=int(
            os.environ.get(
                "COURTVISION_NUM_FRAMES",
                "8",
            )
        ),
        help="Number of sampled frames per QA pair to feed to the model.",
    )
    parser.add_argument(
        "--image-width",
        type=int,
        default=int(
            os.environ.get(
                "COURTVISION_IMAGE_WIDTH",
                str(DEFAULT_IMAGE_WIDTH),
            )
        ),
        help="Width to resize each frame before inference.",
    )
    parser.add_argument(
        "--image-height",
        type=int,
        default=int(
            os.environ.get(
                "COURTVISION_IMAGE_HEIGHT",
                str(DEFAULT_IMAGE_HEIGHT),
            )
        ),
        help="Height to resize each frame before inference.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    subset = pd.read_csv(SUBSET_PATH)
    manifest = pd.read_csv(MANIFEST_PATH)

    required_subset_columns = {
        "sample_id",
        "qa_id",
        "video_id",
        "question",
        "answer",
        "question_type",
    }
    missing_columns = required_subset_columns - set(subset.columns)

    if missing_columns:
        raise ValueError(f"subset.csv is missing: {missing_columns}")

    frames_by_qa = load_guided_frames(
        manifest=manifest,
        num_frames=args.num_frames,
    )

    processor = AutoProcessor.from_pretrained(args.model_name)
    model = (
        Qwen3VLForConditionalGeneration
        .from_pretrained(
            args.model_name,
            dtype="auto",
            device_map="auto",
        )
    )
    base.configure_deterministic_generation(model)
    model.eval()

    completed_ids = set()

    if OUTPUT_PATH.exists():
        with OUTPUT_PATH.open("r", encoding="utf-8") as file:
            for line in file:
                record = json.loads(line)
                completed_ids.add(str(record["qa_id"]))

        print(
            f"Resuming after {len(completed_ids)} completed examples."
        )

    run_count = 0

    with OUTPUT_PATH.open("a", encoding="utf-8") as output_file:
        for position, row in enumerate(
            subset.itertuples(index=False),
            start=1,
        ):
            qa_id = str(row.qa_id)

            if qa_id in completed_ids:
                continue

            if args.limit is not None and run_count >= args.limit:
                break

            if qa_id not in frames_by_qa:
                print(f"Skipping {qa_id}: no CLIP-guided frames")
                continue

            frame_paths = frames_by_qa[qa_id]

            try:
                prediction, inference_time = base.generate_answer(
                    model=model,
                    processor=processor,
                    frame_paths=frame_paths,
                    question=row.question,
                    max_new_tokens=args.max_new_tokens,
                    image_width=args.image_width,
                    image_height=args.image_height,
                )

                result = {
                    "sample_id": row.sample_id,
                    "qa_id": row.qa_id,
                    "split": row.split,
                    "video_id": row.video_id,
                    "method": "guided_motion_yolo_clip",
                    "num_frames": len(frame_paths),
                    "image_width": args.image_width,
                    "image_height": args.image_height,
                    "frame_paths": [
                        str(path.relative_to(ROOT))
                        for path in frame_paths
                    ],
                    "question": row.question,
                    "ground_truth": row.answer,
                    "question_type": row.question_type,
                    "prediction": prediction,
                    "inference_time_seconds": round(inference_time, 4),
                    "model": args.model_name,
                }

                output_file.write(
                    json.dumps(result, ensure_ascii=False) + "\n"
                )
                output_file.flush()
                completed_ids.add(qa_id)
                run_count += 1

                print(
                    f"[{position}/{len(subset)}] "
                    f"{row.qa_id}\n"
                    f"Question: {row.question}\n"
                    f"Ground truth: {row.answer}\n"
                    f"Prediction: {prediction}\n"
                )

            except Exception as error:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                print(f"Failed {row.qa_id}: {error}")
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
