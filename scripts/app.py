from __future__ import annotations

import asyncio
import hashlib
import json
import re
import shutil
import string
import subprocess
import sys
from collections import Counter
from pathlib import Path

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import gradio as gr
import pandas as pd
import cv2


ROOT = Path(__file__).resolve().parents[1]

SUBSET_PATH = ROOT / "data" / "subset.csv"

UNIFORM_PREDICTIONS = (
    ROOT / "outputs" / "predictions" / "uniform_predictions.jsonl"
)
MOTION_YOLO_PREDICTIONS = (
    ROOT / "outputs" / "predictions" / "guided_motion_yolo_predictions.jsonl"
)
CLIP_PREDICTIONS = (
    ROOT / "outputs" / "predictions" / "guided_motion_yolo_clip_predictions.jsonl"
)

UNIFORM_MANIFEST = ROOT / "data" / "uniform_frame_manifest.csv"
MOTION_YOLO_MANIFEST = ROOT / "data" / "guided_frame_manifest.csv"
CLIP_MANIFEST = ROOT / "data" / "guided_clip_frame_manifest.csv"

ANALYTICS_ROOT = ROOT / "outputs" / "analytics"
TRACK_DIR = ANALYTICS_ROOT / "tracks"
PLAYER_METRICS_DIR = ANALYTICS_ROOT / "player_metrics"
FRAME_METRICS_DIR = ANALYTICS_ROOT / "frame_metrics"
VISUALIZATION_DIR = ANALYTICS_ROOT / "visualizations"
HEATMAP_DIR = ANALYTICS_ROOT / "heatmaps"

METHODS = {
    "Uniform": {
        "predictions": UNIFORM_PREDICTIONS,
        "manifest": UNIFORM_MANIFEST,
        "manifest_key": "video_id",
    },
    "Motion + YOLO": {
        "predictions": MOTION_YOLO_PREDICTIONS,
        "manifest": MOTION_YOLO_MANIFEST,
        "manifest_key": "video_id",
    },
    "Motion + YOLO + CLIP": {
        "predictions": CLIP_PREDICTIONS,
        "manifest": CLIP_MANIFEST,
        "manifest_key": "qa_id",
    },
}


def normalize_answer(text: str) -> str:
    text = str(text).lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def token_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()

    if not pred_tokens and not gt_tokens:
        return 1.0
    if not pred_tokens or not gt_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gt_tokens)
    overlap = sum(common.values())

    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(gt_tokens)

    return 2 * precision * recall / (precision + recall)


def exact_match(prediction: str, ground_truth: str) -> bool:
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def resolve_project_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else ROOT / path


def safe_video_id(video_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", str(video_id)).strip("_")


def browser_video_path(source_path: Path) -> str | None:
    """
    Return a browser-friendly MP4 path.

    Sports-QA clips are commonly AVI. The first time an AVI example is
    opened, it is converted to a cached MP4 inside demo/.video_cache.
    """
    if not source_path.exists():
        return None

    cache_dir = Path(__file__).resolve().parent / ".video_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    if (
        source_path.parent == cache_dir
        and source_path.suffix.lower() == ".mp4"
        and "_browser_h264" in source_path.stem
    ):
        return str(source_path)

    source_key = hashlib.sha1(str(source_path).encode("utf-8")).hexdigest()[:10]
    target = cache_dir / f"{source_path.stem}_{source_key}_browser_h264.mp4"

    if (
        target.exists()
        and target.stat().st_size > 0
        and target.stat().st_mtime >= source_path.stat().st_mtime
    ):
        return str(target)

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        command = [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(source_path),
            "-map",
            "0:v:0",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-preset",
            "veryfast",
            "-movflags",
            "+faststart",
            "-an",
            str(target),
        ]
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0 and target.exists() and target.stat().st_size > 0:
            return str(target)

    capture = cv2.VideoCapture(str(source_path))
    if not capture.isOpened():
        return None

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0:
        fps = 30.0

    writer = cv2.VideoWriter(
        str(target),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            writer.write(frame)
    finally:
        capture.release()
        writer.release()

    if not target.exists() or target.stat().st_size == 0:
        return None

    return str(target)


def load_jsonl(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing prediction file: {path}")

    records = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))

    df = pd.DataFrame(records)
    if df.empty:
        raise ValueError(f"No prediction records found in {path}")

    df["qa_id"] = df["qa_id"].astype(str)
    return df.drop_duplicates(subset=["qa_id"], keep="last").copy()


def load_manifest(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing manifest file: {path}")

    df = pd.read_csv(path)
    if "qa_id" in df.columns:
        df["qa_id"] = df["qa_id"].astype(str)
    return df


def load_all_data():
    if not SUBSET_PATH.exists():
        raise FileNotFoundError(f"Missing subset file: {SUBSET_PATH}")

    subset = pd.read_csv(SUBSET_PATH)
    subset["qa_id"] = subset["qa_id"].astype(str)

    predictions = {
        method: load_jsonl(config["predictions"])
        for method, config in METHODS.items()
    }
    manifests = {
        method: load_manifest(config["manifest"])
        for method, config in METHODS.items()
    }

    valid_ids = set(subset["qa_id"])
    for method in predictions:
        predictions[method] = predictions[method][
            predictions[method]["qa_id"].isin(valid_ids)
        ].copy()

    return subset, predictions, manifests


SUBSET, PREDICTIONS, MANIFESTS = load_all_data()


def example_label(row) -> str:
    question = str(row.question).replace("\n", " ").strip()
    if len(question) > 90:
        question = question[:87] + "..."
    return f"{row.sample_id} | {row.question_type} | {question}"


EXAMPLE_MAP = {
    example_label(row): str(row.qa_id)
    for row in SUBSET.itertuples(index=False)
}
EXAMPLE_CHOICES = list(EXAMPLE_MAP)


def get_subset_row(qa_id: str) -> pd.Series:
    match = SUBSET[SUBSET["qa_id"] == str(qa_id)]
    if match.empty:
        raise KeyError(f"qa_id not found in subset: {qa_id}")
    return match.iloc[0]


def get_prediction_row(method: str, qa_id: str) -> pd.Series:
    df = PREDICTIONS[method]
    match = df[df["qa_id"] == str(qa_id)]
    if match.empty:
        raise KeyError(f"{method} missing qa_id={qa_id}")
    return match.iloc[0]


def get_frames(method: str, qa_id: str, video_id: str):
    manifest = MANIFESTS[method]
    key = METHODS[method]["manifest_key"]

    if key == "qa_id":
        rows = manifest[
            manifest["qa_id"].astype(str) == str(qa_id)
        ].copy()
    else:
        rows = manifest[
            manifest["video_id"] == video_id
        ].copy()

    if rows.empty:
        return [], pd.DataFrame()

    rows = rows.sort_values("frame_order")

    paths = []
    for value in rows["frame_path"].tolist():
        path = resolve_project_path(value)
        if path.exists():
            paths.append(str(path))

    return paths, rows


def prediction_badge(prediction: str, ground_truth: str) -> str:
    em = exact_match(prediction, ground_truth)
    f1 = token_f1(prediction, ground_truth)

    if em:
        return f"✅ Exact Match · Token F1: {f1:.3f}"
    if f1 > 0:
        return f"🟡 Partial lexical overlap · Token F1: {f1:.3f}"
    return f"❌ No lexical match · Token F1: {f1:.3f}"


def score_table_for_frames(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame(
            columns=["frame_order", "frame_index", "timestamp_seconds"]
        )

    preferred = [
        "frame_order",
        "frame_index",
        "timestamp_seconds",
        "normalized_motion_score",
        "normalized_person_score",
        "normalized_ball_score",
        "normalized_clip_score",
        "combined_score",
    ]

    columns = [c for c in preferred if c in rows.columns]
    return rows[columns].reset_index(drop=True)


def build_metrics_table() -> pd.DataFrame:
    result = []

    for method, df in PREDICTIONS.items():
        em = df.apply(
            lambda row: exact_match(
                row["prediction"],
                row["ground_truth"],
            ),
            axis=1,
        )
        f1 = df.apply(
            lambda row: token_f1(
                row["prediction"],
                row["ground_truth"],
            ),
            axis=1,
        )

        result.append(
            {
                "Method": method,
                "Examples": len(df),
                "Exact Match": f"{em.mean() * 100:.2f}%",
                "Mean Token F1": round(float(f1.mean()), 4),
            }
        )

    return pd.DataFrame(result)


def build_type_table() -> pd.DataFrame:
    result = []

    for method, df in PREDICTIONS.items():
        merged = SUBSET[
            ["qa_id", "question_type"]
        ].merge(
            df[["qa_id", "prediction", "ground_truth"]],
            on="qa_id",
            how="inner",
        )

        merged["exact_match"] = merged.apply(
            lambda row: exact_match(
                row["prediction"],
                row["ground_truth"],
            ),
            axis=1,
        )
        merged["token_f1"] = merged.apply(
            lambda row: token_f1(
                row["prediction"],
                row["ground_truth"],
            ),
            axis=1,
        )

        for question_type, group in merged.groupby("question_type"):
            result.append(
                {
                    "Method": method,
                    "Question Type": question_type,
                    "N": len(group),
                    "Exact Match": f"{group['exact_match'].mean() * 100:.2f}%",
                    "Mean Token F1": round(
                        float(group["token_f1"].mean()),
                        4,
                    ),
                }
            )

    return pd.DataFrame(result)


METRICS_TABLE = build_metrics_table()
TYPE_TABLE = build_type_table()

VIDEO_ROWS = SUBSET[["video_id", "video_path"]].drop_duplicates("video_id")


def analytics_video_label(row) -> str:
    return str(row.video_id)


ANALYTICS_VIDEO_MAP = {
    analytics_video_label(row): {
        "video_id": str(row.video_id),
        "video_path": str(row.video_path),
    }
    for row in VIDEO_ROWS.itertuples(index=False)
}
ANALYTICS_VIDEO_CHOICES = list(ANALYTICS_VIDEO_MAP)


def default_analytics_video_choice() -> str:
    for choice in ANALYTICS_VIDEO_CHOICES:
        video_id = ANALYTICS_VIDEO_MAP[choice]["video_id"]
        if (TRACK_DIR / f"{safe_video_id(video_id)}.csv").exists():
            return choice
    return ANALYTICS_VIDEO_CHOICES[0]


def read_analytics_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def analytics_paths(video_id: str) -> dict[str, Path]:
    safe_id = safe_video_id(video_id)
    return {
        "tracks": TRACK_DIR / f"{safe_id}.csv",
        "player_metrics": PLAYER_METRICS_DIR / f"{safe_id}.csv",
        "frame_metrics": FRAME_METRICS_DIR / f"{safe_id}.csv",
        "annotated_video": VISUALIZATION_DIR / f"{safe_id}_tracked.mp4",
        "heatmap": HEATMAP_DIR / f"{safe_id}_occupancy_heatmap.jpg",
    }


def analytics_summary(
    tracks: pd.DataFrame,
    player_metrics: pd.DataFrame,
    frame_metrics: pd.DataFrame,
) -> str:
    track_count = (
        int(tracks["track_id"].nunique())
        if not tracks.empty and "track_id" in tracks.columns
        else 0
    )
    detection_count = int(len(tracks))
    tracked_frame_count = (
        int(tracks["frame_index"].nunique())
        if not tracks.empty and "frame_index" in tracks.columns
        else 0
    )
    mean_visible_count = (
        detection_count / tracked_frame_count
        if tracked_frame_count
        else 0.0
    )
    longest_track = (
        int(player_metrics["frames_seen"].max())
        if not player_metrics.empty and "frames_seen" in player_metrics.columns
        else 0
    )
    mean_spacing = (
        float(frame_metrics["average_pairwise_player_distance_pixels"].mean())
        if (
            not frame_metrics.empty
            and "average_pairwise_player_distance_pixels" in frame_metrics.columns
        )
        else 0.0
    )
    return (
        "### Analytics Summary\n"
        f"**YOLO person track IDs:** {track_count}  \n"
        f"**Tracked person-box detections:** {detection_count}  \n"
        f"**Mean visible person boxes per frame:** {mean_visible_count:.2f}  \n"
        f"**Longest player track:** {longest_track} frames  \n"
        f"**Mean pairwise player spacing:** {mean_spacing:.2f} pixels\n\n"
        "_Counts are YOLO class-0 person boxes tracked across frames; "
        "they may include referees, bench personnel, or spectators._"
    )


def show_analytics(video_choice: str):
    video_info = ANALYTICS_VIDEO_MAP[video_choice]
    video_id = video_info["video_id"]
    original_video = browser_video_path(
        resolve_project_path(video_info["video_path"])
    )
    paths = analytics_paths(video_id)

    tracks = read_analytics_csv(paths["tracks"])
    player_metrics = read_analytics_csv(paths["player_metrics"])
    frame_metrics = read_analytics_csv(paths["frame_metrics"])

    annotated_video = (
        browser_video_path(paths["annotated_video"])
        if paths["annotated_video"].exists()
        else None
    )
    heatmap = str(paths["heatmap"]) if paths["heatmap"].exists() else None

    missing = [
        name
        for name, path in paths.items()
        if not path.exists()
    ]
    status = (
        "### Precomputed Analytics\n"
        "All analytics artifacts are available."
        if not missing
        else (
            "### Precomputed Analytics\n"
            "Missing artifacts: "
            + ", ".join(missing)
            + "\n\nRun the analytics scripts before presenting this tab."
        )
    )

    return [
        original_video,
        annotated_video,
        heatmap,
        analytics_summary(
            tracks=tracks,
            player_metrics=player_metrics,
            frame_metrics=frame_metrics,
        ),
        status,
        player_metrics,
        frame_metrics,
    ]


def show_example(choice: str):
    qa_id = EXAMPLE_MAP[choice]
    subset_row = get_subset_row(qa_id)

    video_path = resolve_project_path(subset_row["video_path"])
    video_value = browser_video_path(video_path)

    question_md = (
        f"### Question\n"
        f"{subset_row['question']}\n\n"
        f"**Question type:** {subset_row['question_type']}  \n"
        f"**Ground truth:** `{subset_row['answer']}`  \n"
        f"**QA ID:** `{qa_id}`"
    )

    outputs = [video_value, question_md]

    for method in METHODS:
        prediction_row = get_prediction_row(method, qa_id)
        prediction = str(prediction_row["prediction"])
        ground_truth = str(prediction_row["ground_truth"])

        frame_paths, frame_rows = get_frames(
            method=method,
            qa_id=qa_id,
            video_id=subset_row["video_id"],
        )

        result_md = (
            f"### {method}\n"
            f"**Prediction:** `{prediction}`  \n"
            f"{prediction_badge(prediction, ground_truth)}"
        )

        outputs.extend(
            [
                frame_paths,
                result_md,
                score_table_for_frames(frame_rows),
            ]
        )

    return outputs


CUSTOM_CSS = """
.gradio-container {
    max-width: 1500px !important;
}
.method-card {
    border: 1px solid var(--border-color-primary);
    border-radius: 12px;
    padding: 6px;
}
"""

with gr.Blocks(
    title="CourtVision",
    css=CUSTOM_CSS,
    theme=gr.themes.Soft(),
) as demo:

    gr.Markdown(
        """
# 🏀 CourtVision

### CV-guided keyframe selection for basketball VideoQA

Compare **Uniform**, **Motion + YOLO**, and
**Question-Aware Motion + YOLO + CLIP** on the same Sports-QA examples.

This demo reads your **precomputed frames and Qwen3-VL predictions**.
It does not rerun the VLM when you switch examples.
"""
    )

    with gr.Tabs():

        with gr.Tab("Example Explorer"):
            selector = gr.Dropdown(
                choices=EXAMPLE_CHOICES,
                value=EXAMPLE_CHOICES[0],
                label="Select Sports-QA example",
                interactive=True,
            )

            with gr.Row():
                video = gr.Video(
                    label="Sports-QA basketball clip",
                    interactive=False,
                    format="mp4",
                    include_audio=False,
                )
                question_info = gr.Markdown()

            gr.Markdown("## Selected frames and predictions")

            method_components = []

            for method in METHODS:
                with gr.Group(elem_classes=["method-card"]):
                    gr.Markdown(f"## {method}")

                    gallery = gr.Gallery(
                        label=f"{method} — selected frames",
                        columns=4,
                        rows=2,
                        height=430,
                        object_fit="contain",
                        preview=True,
                    )

                    result = gr.Markdown()

                    score_df = gr.Dataframe(
                        label="Frame-selection scores",
                        interactive=False,
                        wrap=True,
                    )

                    method_components.extend(
                        [gallery, result, score_df]
                    )

            outputs = [
                video,
                question_info,
                *method_components,
            ]

            selector.change(
                fn=show_example,
                inputs=selector,
                outputs=outputs,
            )
            demo.load(
                fn=show_example,
                inputs=selector,
                outputs=outputs,
            )

        with gr.Tab("Evaluation"):
            gr.Markdown(
                """
## Overall quantitative results

All methods are evaluated on the same fixed Sports-QA subset.
"""
            )

            gr.Dataframe(
                value=METRICS_TABLE,
                interactive=False,
                wrap=True,
                label="Overall results",
            )

            gr.Markdown("## Results by question type")

            gr.Dataframe(
                value=TYPE_TABLE,
                interactive=False,
                wrap=True,
                label="Question-type breakdown",
            )

        with gr.Tab("Analytics"):
            analytics_selector = gr.Dropdown(
                choices=ANALYTICS_VIDEO_CHOICES,
                value=default_analytics_video_choice(),
                label="Select Sports-QA video",
                interactive=True,
            )

            with gr.Row():
                analytics_original_video = gr.Video(
                    label="Original video",
                    interactive=False,
                    format="mp4",
                    include_audio=False,
                )
                analytics_tracked_video = gr.Video(
                    label="Annotated tracked video",
                    interactive=False,
                    format="mp4",
                    include_audio=False,
                )

            with gr.Row():
                analytics_heatmap = gr.Image(
                    label="Player occupancy heatmap",
                    interactive=False,
                    type="filepath",
                )
                with gr.Column():
                    analytics_summary_md = gr.Markdown()
                    analytics_status_md = gr.Markdown()

            analytics_player_metrics = gr.Dataframe(
                label="Player metrics",
                interactive=False,
                wrap=True,
            )
            analytics_frame_metrics = gr.Dataframe(
                label="Frame metrics",
                interactive=False,
                wrap=True,
            )

            analytics_outputs = [
                analytics_original_video,
                analytics_tracked_video,
                analytics_heatmap,
                analytics_summary_md,
                analytics_status_md,
                analytics_player_metrics,
                analytics_frame_metrics,
            ]

            analytics_selector.change(
                fn=show_analytics,
                inputs=analytics_selector,
                outputs=analytics_outputs,
            )
            demo.load(
                fn=show_analytics,
                inputs=analytics_selector,
                outputs=analytics_outputs,
            )


if __name__ == "__main__":
    demo.launch(allowed_paths=[str(ROOT)])
