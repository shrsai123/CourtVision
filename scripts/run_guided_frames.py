import argparse
import json
import os
from os import path
import time
from pathlib import Path
import torch


os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True",
)

import pandas as pd
import torch
from qwen_vl_utils import process_vision_info
from transformers import (
    Qwen3VLForConditionalGeneration, AutoProcessor
)


ROOT = Path(__file__).resolve().parents[1]
SUBSET_PATH = ROOT / "data" / "subset.csv"

MANIFEST_PATH= ROOT / "data" / "guided_frame_manifest.csv"

OUTPUT_DIR = ROOT / "outputs" / "predictions"
OUTPUT_PATH = (
    OUTPUT_DIR
    / "guided_motion_yolo_predictions.jsonl"
)

DEFAULT_MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"
DEFAULT_MAX_NEW_TOKENS = 16
DEFAULT_IMAGE_WIDTH = 448
DEFAULT_IMAGE_HEIGHT = 252




def configure_deterministic_generation(model) -> None:
    model.generation_config.do_sample = False
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None


def resolve_path(path_value: str) -> Path:
    path = Path(path_value)

    if path.is_absolute():
        return path

    return ROOT / path


def select_frame_rows(
    video_rows: pd.DataFrame,
    num_frames: int,
) -> pd.DataFrame:
    if len(video_rows) < num_frames:
        raise ValueError(
            f"Only {len(video_rows)} frames available; "
            f"requested {num_frames}."
        )

    if len(video_rows) == num_frames:
        return video_rows

    if num_frames == 1:
        return video_rows.iloc[[len(video_rows) // 2]]

    positions = [
        round(i * (len(video_rows) - 1) / (num_frames - 1))
        for i in range(num_frames)
    ]
    return video_rows.iloc[positions]


def load_guided_frames(
    manifest: pd.DataFrame,
    num_frames: int,
) -> dict[str, list[Path]]:
    frame_paths_by_video = {}
    grouped = manifest.groupby("video_id")

    for video_id, video_rows in grouped:
        video_rows = video_rows.sort_values("frame_order")
        video_rows = select_frame_rows(
            video_rows=video_rows,
            num_frames=num_frames,
        )

        frame_paths = [
            resolve_path(path)
            for path in video_rows["frame_path"].tolist()
        ]

        if len(frame_paths) != num_frames:
            raise ValueError(
                f"{video_id} has {len(frame_paths)} frames; "
                f"expected exactly {num_frames}."
            )

        missing = [
            path for path in frame_paths if not path.exists()
        ]

        if missing:
            raise FileNotFoundError(
                f"Missing frames for {video_id}: {missing}"
            )

        frame_paths_by_video[video_id] = frame_paths

    return frame_paths_by_video

def build_messages(
    question: str,
    frame_paths: list[Path],
    image_width: int,
    image_height: int,
) -> list[dict]:
    content= []

    for frame_path in frame_paths:
        content.append({
            "type": "image",
            "image": str(frame_path.resolve()),
            "resized_width": image_width,
            "resized_height": image_height,
        })
    prompt = (
        "The following images are frames sampled from a "
        "basketball video in chronological order.\n\n"
        f"Question: {question}\n\n"
        "Answer the question using only the visible "
        "information in the frames. Give only a short "
        "answer without explanation."
    )

    content.append(
        {
            "type": "text",
            "text": prompt,
        }
    )

    return [
        {
            "role": "user",
            "content": content,
        }
    ]
def generate_answer(
    model,
    processor,
    frame_paths: list[Path],
    question: str,
    max_new_tokens: int,
    image_width: int,
    image_height: int,
) -> tuple[str, float]:
    messages = build_messages(
        frame_paths=frame_paths,
        question=question,
        image_width=image_width,
        image_height=image_height,
    )

    prompt_text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    image_inputs, video_inputs = process_vision_info(
        messages
    )

    inputs = processor(
        text=[prompt_text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    inputs = inputs.to(model.device)

    start_time = time.perf_counter()

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    elapsed_seconds = time.perf_counter() - start_time

    generated_only = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(
            inputs.input_ids,
            generated_ids,
        )
    ]

    answer = processor.batch_decode(
        generated_only,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()

    return answer, elapsed_seconds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the motion + YOLO guided Sports-QA pipeline."
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
        help="Number of sampled frames per video to feed to the model.",
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


def main():
    args = parse_args()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

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

    missing_columns = (
        required_subset_columns - set(subset.columns)
    )

    if missing_columns:
        raise ValueError(
            f"subset.csv is missing: {missing_columns}"
        )

    frames_by_video = load_guided_frames(
        manifest=manifest,
        num_frames=args.num_frames,
    )

    processor = AutoProcessor.from_pretrained(
        args.model_name
    )

    model = (
        Qwen3VLForConditionalGeneration
        .from_pretrained(
            args.model_name,
            dtype="auto",
            device_map="auto",
        )
    )

    configure_deterministic_generation(model)
    model.eval()

    completed_ids = set()

    if OUTPUT_PATH.exists():
        with OUTPUT_PATH.open(
            "r",
            encoding="utf-8",
        ) as file:
            for line in file:
                record = json.loads(line)
                completed_ids.add(str(record["qa_id"]))

        print(
            f"Resuming after {len(completed_ids)} "
            "completed examples."
        )

    run_count = 0

    with OUTPUT_PATH.open(
        "a",
        encoding="utf-8",
    ) as output_file:
        for position, row in enumerate(
            subset.itertuples(index=False),
            start=1,
        ):
            qa_id = str(row.qa_id)

            if qa_id in completed_ids:
                continue

            if (
                args.limit is not None
                and run_count >= args.limit
            ):
                break

            if row.video_id not in frames_by_video:
                print(
                    f"Skipping {qa_id}: no frames "
                    f"for {row.video_id}"
                )
                continue

            frame_paths = frames_by_video[row.video_id]

            try:
                prediction, inference_time = (
                    generate_answer(
                        model=model,
                        processor=processor,
                        frame_paths=frame_paths,
                        question=row.question,
                        max_new_tokens=args.max_new_tokens,
                        image_width=args.image_width,
                        image_height=args.image_height,
                    )
                )

                result = {
                    "sample_id": row.sample_id,
                    "qa_id": row.qa_id,
                    "split": row.split,
                    "video_id": row.video_id,
                    "method": "guided_motion_yolo",
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
                    "inference_time_seconds": round(
                        inference_time,
                        4,
                    ),
                    "model": args.model_name,
                }

                output_file.write(
                    json.dumps(
                        result,
                        ensure_ascii=False,
                    )
                    + "\n"
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

                print(
                    f"Failed {row.qa_id}: {error}"
                )
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
