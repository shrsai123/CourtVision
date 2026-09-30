from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]

SUBSET_PATH = ROOT / "data" / "subset.csv"

UNIFORM_MANIFEST = ROOT / "data" / "uniform_frame_manifest.csv"
MOTION_YOLO_MANIFEST = ROOT / "data" / "guided_frame_manifest.csv"
CLIP_MANIFEST = ROOT / "data" / "guided_clip_frame_manifest.csv"

OUTPUT_DIR = ROOT / "outputs" / "figures"

# Larger thumbnails for readability
THUMB_WIDTH = 300
THUMB_HEIGHT = 175

FRAME_LABEL_HEIGHT = 24
METHOD_LABEL_HEIGHT = 28
ROW_GAP = 8
METHOD_GAP = 28

BACKGROUND = "white"
TEXT_COLOR = "black"


def resolve_path(path_value):
    path = Path(str(path_value))

    if path.is_absolute():
        return path

    return ROOT / path


def load_frames(df, path_col="frame_path"):
    frames = []

    # Keep selected frames in chronological order
    if "frame_order" in df.columns:
        df = df.sort_values("frame_order")
    elif "frame_index" in df.columns:
        df = df.sort_values("frame_index")

    for _, row in df.iterrows():
        frame_path = resolve_path(row[path_col])

        if not frame_path.exists():
            print(f"Warning: frame not found: {frame_path}")
            continue

        image = Image.open(frame_path).convert("RGB")

        image = image.resize(
            (THUMB_WIDTH, THUMB_HEIGHT),
            Image.Resampling.LANCZOS,
        )

        frame_index = int(row["frame_index"])

        frames.append((image, frame_index))

    return frames


def draw_centered_text(draw, text, x, y, width, font):
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]

    text_x = x + (width - text_width) // 2

    draw.text(
        (text_x, y),
        text,
        fill=TEXT_COLOR,
        font=font,
    )


def draw_method(canvas, y, method_name, frames, font):
    draw = ImageDraw.Draw(canvas)

    # Method heading
    draw.text(
        (5, y),
        method_name,
        fill=TEXT_COLOR,
        font=font,
    )

    y += METHOD_LABEL_HEIGHT

    for i, (frame, frame_index) in enumerate(frames[:8]):

        # 4 frames per row
        row = i // 4
        col = i % 4

        x = col * THUMB_WIDTH

        frame_y = y + row * (
            THUMB_HEIGHT
            + FRAME_LABEL_HEIGHT
            + ROW_GAP
        )

        canvas.paste(
            frame,
            (x, frame_y),
        )

        frame_label = f"f={frame_index}"

        draw_centered_text(
            draw=draw,
            text=frame_label,
            x=x,
            y=frame_y + THUMB_HEIGHT + 3,
            width=THUMB_WIDTH,
            font=font,
        )

    # Return y position after both rows
    method_height = (
        METHOD_LABEL_HEIGHT
        + 2 * (
            THUMB_HEIGHT
            + FRAME_LABEL_HEIGHT
        )
        + ROW_GAP
    )

    return y - METHOD_LABEL_HEIGHT + method_height


def create_montage(qa_id):
    subset = pd.read_csv(SUBSET_PATH)

    uniform = pd.read_csv(UNIFORM_MANIFEST)
    motion_yolo = pd.read_csv(MOTION_YOLO_MANIFEST)
    clip = pd.read_csv(CLIP_MANIFEST)

    # Find the selected QA example
    qa_rows = subset[
        subset["qa_id"].astype(str) == str(qa_id)
    ]

    if qa_rows.empty:
        raise ValueError(
            f"qa_id not found in subset.csv: {qa_id}"
        )

    qa = qa_rows.iloc[0]

    video_id = str(qa["video_id"])
    question = str(qa["question"])

    print(f"QA ID: {qa_id}")
    print(f"Video: {video_id}")
    print(f"Question: {question}")

    # Uniform is video-specific
    uniform_rows = uniform[
        uniform["video_id"].astype(str) == video_id
    ]

    # Motion + YOLO is also video-specific
    motion_rows = motion_yolo[
        motion_yolo["video_id"].astype(str) == video_id
    ]

    # CLIP selection is QA/question-specific
    clip_rows = clip[
        clip["qa_id"].astype(str) == str(qa_id)
    ]

    uniform_frames = load_frames(uniform_rows)
    motion_frames = load_frames(motion_rows)
    clip_frames = load_frames(clip_rows)

    print(f"Uniform frames: {len(uniform_frames)}")
    print(f"Motion + YOLO frames: {len(motion_frames)}")
    print(f"Motion + YOLO + CLIP frames: {len(clip_frames)}")

    if len(uniform_frames) != 8:
        print(
            f"Warning: Uniform has {len(uniform_frames)} frames, expected 8."
        )

    if len(motion_frames) != 8:
        print(
            f"Warning: Motion+YOLO has {len(motion_frames)} frames, expected 8."
        )

    if len(clip_frames) != 8:
        print(
            f"Warning: CLIP has {len(clip_frames)} frames, expected 8."
        )

    font = ImageFont.load_default()

    # 4 frames horizontally
    width = 4 * THUMB_WIDTH

    single_method_height = (
        METHOD_LABEL_HEIGHT
        + 2 * (
            THUMB_HEIGHT
            + FRAME_LABEL_HEIGHT
        )
        + ROW_GAP
    )

    height = (
        3 * single_method_height
        + 2 * METHOD_GAP
        + 10
    )

    montage = Image.new(
        "RGB",
        (width, height),
        BACKGROUND,
    )

    y = 5

    y = draw_method(
        montage,
        y,
        "(a) Uniform Sampling",
        uniform_frames,
        font,
    )

    y += METHOD_GAP

    y = draw_method(
        montage,
        y,
        "(b) Motion + YOLO",
        motion_frames,
        font,
    )

    y += METHOD_GAP

    draw_method(
        montage,
        y,
        "(c) Motion + YOLO + CLIP",
        clip_frames,
        font,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    safe_qa_id = (
        str(qa_id)
        .replace("/", "_")
        .replace("\\", "_")
    )

    output_path = (
        OUTPUT_DIR
        / f"sampling_comparison_{safe_qa_id}.png"
    )

    montage.save(
        output_path,
        quality=95,
    )

    print("\nSaved montage:")
    print(output_path)

    return output_path


if __name__ == "__main__":
    # Change this if you want another QA example
    QA_ID = "18454"

    create_montage(QA_ID)