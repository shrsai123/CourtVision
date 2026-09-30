import argparse
import json
import math
import re
import string
from collections import Counter
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SUBSET_PATH = ROOT / "data" / "subset.csv"
DEFAULT_BASELINE_PATH = ROOT / "outputs" / "predictions" / "uniform_predictions.jsonl"
DEFAULT_MOTION_YOLO_PATH = ROOT / "outputs" / "predictions" / "guided_motion_yolo_predictions.jsonl"
DEFAULT_MOTION_YOLO_CLIP_PATH = (
    ROOT / "outputs" / "predictions" / "guided_motion_yolo_clip_predictions.jsonl"
)
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "evaluation"
DEFAULT_OUTPUT_PREFIX = "baseline_vs_motion_yolo_vs_motion_yolo_clip"
DEFAULT_UNIFORM_MANIFEST_PATH = ROOT / "data" / "uniform_frame_manifest.csv"
DEFAULT_MOTION_YOLO_MANIFEST_PATH = ROOT / "data" / "guided_frame_manifest.csv"
DEFAULT_MOTION_YOLO_CLIP_MANIFEST_PATH = (
    ROOT / "data" / "guided_clip_frame_manifest.csv"
)

METHODS = [
    {
        "key": "baseline",
        "label": "Baseline",
        "path_attr": "baseline",
        "manifest_attr": "baseline_manifest",
    },
    {
        "key": "motion_yolo",
        "label": "Motion + YOLO",
        "path_attr": "motion_yolo",
        "manifest_attr": "motion_yolo_manifest",
    },
    {
        "key": "motion_yolo_clip",
        "label": "Motion + YOLO + CLIP",
        "path_attr": "motion_yolo_clip",
        "manifest_attr": "motion_yolo_clip_manifest",
    },
]


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


def load_jsonl(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Prediction file not found: {path}")

    records = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {path}"
                ) from error

    if not records:
        raise ValueError(f"No prediction records found in {path}")

    df = pd.DataFrame(records)
    required = {
        "qa_id",
        "video_id",
        "question",
        "ground_truth",
        "question_type",
        "prediction",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")

    df["qa_id"] = df["qa_id"].astype(str)

    if df["qa_id"].duplicated().any():
        duplicate_ids = sorted(
            df.loc[df["qa_id"].duplicated(keep=False), "qa_id"].unique()
        )
        raise ValueError(f"Duplicate qa_id values in {path.name}: {duplicate_ids}")

    return df


def score_predictions(df: pd.DataFrame, method_key: str) -> pd.DataFrame:
    scored = df.copy()
    scored[f"{method_key}_normalized_prediction"] = scored["prediction"].map(
        normalize_answer
    )
    scored[f"{method_key}_normalized_ground_truth"] = scored["ground_truth"].map(
        normalize_answer
    )
    scored[f"{method_key}_exact_match"] = (
        scored[f"{method_key}_normalized_prediction"]
        == scored[f"{method_key}_normalized_ground_truth"]
    )
    scored[f"{method_key}_token_f1"] = scored.apply(
        lambda row: token_f1(row["prediction"], row["ground_truth"]),
        axis=1,
    )
    return scored


def exact_mcnemar_p_value(method_a_wins: int, method_b_wins: int) -> float:
    discordant = method_a_wins + method_b_wins
    if discordant == 0:
        return 1.0

    smaller = min(method_a_wins, method_b_wins)
    tail_probability = sum(
        math.comb(discordant, k) for k in range(smaller + 1)
    ) / (2 ** discordant)
    return min(1.0, 2.0 * tail_probability)


def method_columns(df: pd.DataFrame, method_key: str, keep_metadata: bool) -> pd.DataFrame:
    columns = [
        "qa_id",
        "video_id",
        "question",
        "ground_truth",
        "question_type",
        "prediction",
        f"{method_key}_normalized_prediction",
        f"{method_key}_exact_match",
        f"{method_key}_token_f1",
    ]

    selected = df[columns].rename(
        columns={
            "prediction": f"{method_key}_prediction",
        }
    )

    if keep_metadata:
        return selected

    return selected.rename(
        columns={
            "video_id": f"{method_key}_video_id",
            "question": f"{method_key}_question",
            "ground_truth": f"{method_key}_ground_truth",
            "question_type": f"{method_key}_question_type",
        }
    )


def filter_to_subset(dataframes: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    subset = pd.read_csv(SUBSET_PATH)
    subset["qa_id"] = subset["qa_id"].astype(str)
    valid_ids = set(subset["qa_id"])

    filtered = {}
    for key, df in dataframes.items():
        df = df.copy()
        df["qa_id"] = df["qa_id"].astype(str)
        filtered[key] = df[df["qa_id"].isin(valid_ids)].copy()

    return filtered


def validate_same_examples(comparison: pd.DataFrame, method_keys: list[str]) -> None:
    for method_key in method_keys[1:]:
        mismatched = (
            (comparison["video_id"] != comparison[f"{method_key}_video_id"])
            | (comparison["question"] != comparison[f"{method_key}_question"])
            | (
                comparison["ground_truth"]
                != comparison[f"{method_key}_ground_truth"]
            )
            | (
                comparison["question_type"]
                != comparison[f"{method_key}_question_type"]
            )
        )

        if mismatched.any():
            bad_ids = comparison.loc[mismatched, "qa_id"].head(10).tolist()
            raise ValueError(
                "Prediction files are not based on the same "
                f"video/question/answer pairs. First mismatched qa_id values: "
                f"{bad_ids}"
            )


def create_comparison(dataframes: dict[str, pd.DataFrame]) -> pd.DataFrame:
    method_keys = [method["key"] for method in METHODS]
    filtered = filter_to_subset(dataframes)

    print(
        "After filtering to subset.csv: "
        + ", ".join(
            f"{method['label']}={len(filtered[method['key']])}"
            for method in METHODS
        )
    )

    scored = {
        method["key"]: score_predictions(
            filtered[method["key"]],
            method["key"],
        )
        for method in METHODS
    }

    comparison = method_columns(
        scored["baseline"],
        "baseline",
        keep_metadata=True,
    )

    for method in METHODS[1:]:
        method_key = method["key"]
        comparison = comparison.merge(
            method_columns(
                scored[method_key],
                method_key,
                keep_metadata=False,
            ),
            on="qa_id",
            how="inner",
            validate="one_to_one",
        )

    expected_count = len(filtered["baseline"])
    for method in METHODS:
        actual_count = len(filtered[method["key"]])
        if actual_count != expected_count:
            raise ValueError(
                "Prediction files do not contain the same number of "
                f"filtered examples. Baseline={expected_count}, "
                f"{method['label']}={actual_count}"
            )

    if len(comparison) != expected_count:
        raise ValueError(
            "Prediction files do not contain exactly the same QA IDs. "
            f"Expected={expected_count}, Matched={len(comparison)}"
        )

    validate_same_examples(comparison, method_keys)

    for method in METHODS[1:]:
        method_key = method["key"]
        comparison[f"f1_change_{method_key}_minus_baseline"] = (
            comparison[f"{method_key}_token_f1"]
            - comparison["baseline_token_f1"]
        )

    comparison["best_exact_match_methods"] = comparison.apply(
        lambda row: ";".join(
            method["key"]
            for method in METHODS
            if bool(row[f"{method['key']}_exact_match"])
        ),
        axis=1,
    )

    comparison["best_token_f1_method"] = comparison.apply(
        lambda row: max(
            METHODS,
            key=lambda method: row[f"{method['key']}_token_f1"],
        )["key"],
        axis=1,
    )

    drop_columns = []
    for method in METHODS[1:]:
        method_key = method["key"]
        drop_columns.extend(
            [
                f"{method_key}_video_id",
                f"{method_key}_question",
                f"{method_key}_ground_truth",
                f"{method_key}_question_type",
            ]
        )

    return comparison.drop(columns=drop_columns)


def method_metrics(comparison: pd.DataFrame, method_key: str) -> dict:
    return {
        "exact_match_accuracy": float(
            comparison[f"{method_key}_exact_match"].mean()
        ),
        "mean_token_f1": float(comparison[f"{method_key}_token_f1"].mean()),
        "median_token_f1": float(comparison[f"{method_key}_token_f1"].median()),
    }


def pairwise_summary(
    comparison: pd.DataFrame,
    method_a: dict,
    method_b: dict,
) -> dict:
    key_a = method_a["key"]
    key_b = method_b["key"]
    a_correct = comparison[f"{key_a}_exact_match"].astype(bool)
    b_correct = comparison[f"{key_b}_exact_match"].astype(bool)

    a_only = int((a_correct & ~b_correct).sum())
    b_only = int((~a_correct & b_correct).sum())
    both_correct = int((a_correct & b_correct).sum())
    both_incorrect = int((~a_correct & ~b_correct).sum())

    return {
        "method_a": key_a,
        "method_b": key_b,
        "accuracy_change_a_minus_b_percentage_points": float(
            (a_correct.mean() - b_correct.mean()) * 100.0
        ),
        "mean_token_f1_change_a_minus_b": float(
            comparison[f"{key_a}_token_f1"].mean()
            - comparison[f"{key_b}_token_f1"].mean()
        ),
        "both_correct": both_correct,
        f"only_{key_a}_correct": a_only,
        f"only_{key_b}_correct": b_only,
        "both_incorrect": both_incorrect,
        "mcnemar_exact_test": {
            "discordant_pairs": a_only + b_only,
            f"{key_a}_wins": a_only,
            f"{key_b}_wins": b_only,
            "two_sided_p_value": exact_mcnemar_p_value(a_only, b_only),
        },
    }


def summarize(comparison: pd.DataFrame) -> dict:
    summary = {
        "num_examples": len(comparison),
        "methods": {
            method["key"]: method_metrics(comparison, method["key"])
            for method in METHODS
        },
        "pairwise": {},
    }

    pairs = [
        (METHODS[1], METHODS[0]),
        (METHODS[2], METHODS[0]),
        (METHODS[2], METHODS[1]),
    ]

    for method_a, method_b in pairs:
        pair_key = f"{method_a['key']}_minus_{method_b['key']}"
        summary["pairwise"][pair_key] = pairwise_summary(
            comparison,
            method_a,
            method_b,
        )

    return summary


def summarize_by_question_type(comparison: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for question_type, group in comparison.groupby("question_type", dropna=False):
        row = {
            "question_type": question_type,
            "n": len(group),
        }

        for method in METHODS:
            method_key = method["key"]
            row[f"{method_key}_accuracy"] = group[
                f"{method_key}_exact_match"
            ].mean()
            row[f"{method_key}_token_f1"] = group[
                f"{method_key}_token_f1"
            ].mean()

        row["motion_yolo_accuracy_change_pp"] = (
            row["motion_yolo_accuracy"] - row["baseline_accuracy"]
        ) * 100.0
        row["motion_yolo_clip_accuracy_change_pp"] = (
            row["motion_yolo_clip_accuracy"] - row["baseline_accuracy"]
        ) * 100.0
        row["clip_minus_motion_yolo_accuracy_change_pp"] = (
            row["motion_yolo_clip_accuracy"] - row["motion_yolo_accuracy"]
        ) * 100.0
        row["motion_yolo_token_f1_change"] = (
            row["motion_yolo_token_f1"] - row["baseline_token_f1"]
        )
        row["motion_yolo_clip_token_f1_change"] = (
            row["motion_yolo_clip_token_f1"] - row["baseline_token_f1"]
        )
        row["clip_minus_motion_yolo_token_f1_change"] = (
            row["motion_yolo_clip_token_f1"] - row["motion_yolo_token_f1"]
        )

        rows.append(row)

    return pd.DataFrame(rows).sort_values(
        by=["n", "question_type"],
        ascending=[False, True],
    )


def create_manual_review_template(comparison: pd.DataFrame) -> pd.DataFrame:
    mask = ~(
        comparison["baseline_exact_match"].astype(bool)
        & comparison["motion_yolo_exact_match"].astype(bool)
        & comparison["motion_yolo_clip_exact_match"].astype(bool)
    )
    review = comparison[mask].copy()
    review = review[
        [
            "qa_id",
            "video_id",
            "question_type",
            "question",
            "ground_truth",
            "baseline_prediction",
            "motion_yolo_prediction",
            "motion_yolo_clip_prediction",
            "baseline_exact_match",
            "motion_yolo_exact_match",
            "motion_yolo_clip_exact_match",
            "baseline_token_f1",
            "motion_yolo_token_f1",
            "motion_yolo_clip_token_f1",
            "best_exact_match_methods",
            "best_token_f1_method",
        ]
    ]
    review["baseline_semantic_label"] = ""
    review["motion_yolo_semantic_label"] = ""
    review["motion_yolo_clip_semantic_label"] = ""
    review["allowed_labels"] = "Correct|Partial|Incorrect|Ambiguous"
    review["reviewer_notes"] = ""
    return review


def load_manifest(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        print(f"Frame manifest not found; skipping: {path}")
        return None

    manifest = pd.read_csv(path)
    if manifest.empty:
        print(f"Frame manifest is empty; skipping: {path}")
        return None

    return manifest


def frame_group_columns(manifest: pd.DataFrame, method_key: str) -> list[str]:
    if method_key == "motion_yolo_clip" and "qa_id" in manifest.columns:
        manifest["qa_id"] = manifest["qa_id"].astype(str)
        return ["qa_id"]

    return ["video_id"]


def frame_selection_analysis(
    manifests: dict[str, pd.DataFrame | None],
    comparison: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    qa_to_video = dict(zip(comparison["qa_id"], comparison["video_id"]))

    for method in METHODS:
        method_key = method["key"]
        manifest = manifests.get(method_key)
        if manifest is None:
            continue

        required = {"frame_index", "frame_path"}
        missing = required - set(manifest.columns)
        if missing:
            print(
                f"Skipping frame analysis for {method['label']}; "
                f"manifest is missing {sorted(missing)}"
            )
            continue

        group_columns = frame_group_columns(manifest, method_key)
        grouped = manifest.groupby(group_columns, dropna=False)

        for group_key, group in grouped:
            if not isinstance(group_key, tuple):
                group_key = (group_key,)

            row = {
                "method": method_key,
                "method_label": method["label"],
                "num_selected_frames": int(len(group)),
                "first_frame_index": int(group["frame_index"].min()),
                "last_frame_index": int(group["frame_index"].max()),
                "frame_index_span": int(
                    group["frame_index"].max() - group["frame_index"].min()
                ),
                "mean_frame_index": float(group["frame_index"].mean()),
                "selected_frame_indices": " ".join(
                    str(int(value))
                    for value in sorted(group["frame_index"].tolist())
                ),
                "frame_paths": " ".join(group["frame_path"].astype(str).tolist()),
            }

            if "qa_id" in group_columns:
                qa_id = str(group_key[0])
                row["qa_id"] = qa_id
                row["video_id"] = qa_to_video.get(
                    qa_id,
                    str(group["video_id"].iloc[0])
                    if "video_id" in group.columns
                    else "",
                )
            else:
                row["qa_id"] = ""
                row["video_id"] = str(group_key[0])

            if "timestamp_seconds" in group.columns:
                row["first_timestamp_seconds"] = float(
                    group["timestamp_seconds"].min()
                )
                row["last_timestamp_seconds"] = float(
                    group["timestamp_seconds"].max()
                )
                row["temporal_span_seconds"] = float(
                    group["timestamp_seconds"].max()
                    - group["timestamp_seconds"].min()
                )

            if "video_total_frames" in group.columns:
                total_frames = float(group["video_total_frames"].iloc[0])
                if total_frames > 0:
                    row["temporal_coverage_ratio"] = float(
                        row["frame_index_span"] / total_frames
                    )

            if "clip_score" in group.columns:
                row["mean_clip_score"] = float(group["clip_score"].mean())
                row["max_clip_score"] = float(group["clip_score"].max())
                row["mean_normalized_clip_score"] = float(
                    group["normalized_clip_score"].mean()
                )

            rows.append(row)

    return pd.DataFrame(rows)


def build_sectioned_report(
    summary: dict,
    by_type: pd.DataFrame,
    frame_analysis: pd.DataFrame,
) -> dict:
    return {
        "1_quantitative_automatic_evaluation": {
            "metrics": [
                "Exact Match",
                "Token F1",
                "McNemar paired test",
            ],
            "methods": summary["methods"],
            "pairwise_mcnemar": {
                key: value["mcnemar_exact_test"]
                for key, value in summary["pairwise"].items()
            },
            "pairwise_metric_changes": {
                key: {
                    "accuracy_change_a_minus_b_percentage_points": value[
                        "accuracy_change_a_minus_b_percentage_points"
                    ],
                    "mean_token_f1_change_a_minus_b": value[
                        "mean_token_f1_change_a_minus_b"
                    ],
                }
                for key, value in summary["pairwise"].items()
            },
        },
        "2_human_semantic_evaluation": {
            "labels": [
                "Correct",
                "Partial",
                "Incorrect",
                "Ambiguous",
            ],
            "metric": "Human semantic accuracy",
            "template_file": (
                f"{DEFAULT_OUTPUT_PREFIX}_human_semantic_review.csv"
            ),
            "semantic_accuracy_formula": (
                "Count labels marked Correct as semantically correct; "
                "report Correct / reviewed examples per method. "
                "Partial can be reported separately."
            ),
        },
        "3_frame_selection_analysis": {
            "artifacts": [
                "selected-frame visualizations",
                "question relevance",
                "temporal coverage",
            ],
            "num_rows": int(len(frame_analysis)),
            "columns": list(frame_analysis.columns),
        },
        "4_question_type_analysis": {
            "question_types": by_type["question_type"].astype(str).tolist(),
            "table": by_type.to_dict(orient="records"),
        },
    }


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    columns = list(df.columns)
    rows = []
    rows.append("| " + " | ".join(columns) + " |")
    rows.append("| " + " | ".join("---" for _ in columns) + " |")

    for _, row in df.iterrows():
        values = [
            str(row[column])
            for column in columns
        ]
        rows.append("| " + " | ".join(values) + " |")

    return "\n".join(rows)


def write_markdown_report(
    path: Path,
    summary: dict,
    by_type: pd.DataFrame,
) -> None:
    lines = []
    lines.append("# CourtVision Evaluation")
    lines.append("")
    lines.append("## 1. Quantitative Automatic Evaluation")
    lines.append("")
    lines.append("| Method | Exact Match | Token F1 |")
    lines.append("|---|---:|---:|")
    for method in METHODS:
        metrics = summary["methods"][method["key"]]
        lines.append(
            f"| {method['label']} | "
            f"{metrics['exact_match_accuracy']:.2%} | "
            f"{metrics['mean_token_f1']:.4f} |"
        )
    lines.append("")
    lines.append("### McNemar Paired Tests")
    lines.append("")
    lines.append("| Comparison | Discordant Pairs | p-value |")
    lines.append("|---|---:|---:|")
    for pair_key, pair in summary["pairwise"].items():
        method_a = next(
            method for method in METHODS if method["key"] == pair["method_a"]
        )
        method_b = next(
            method for method in METHODS if method["key"] == pair["method_b"]
        )
        mcnemar = pair["mcnemar_exact_test"]
        lines.append(
            f"| {method_a['label']} vs {method_b['label']} | "
            f"{mcnemar['discordant_pairs']} | "
            f"{mcnemar['two_sided_p_value']:.6f} |"
        )
    lines.append("")
    lines.append("## 2. Human Semantic Evaluation")
    lines.append("")
    lines.append(
        "Use the human semantic review CSV to label each prediction as "
        "Correct, Partial, Incorrect, or Ambiguous."
    )
    lines.append("")
    lines.append(
        "Human semantic accuracy should be reported as Correct labels divided "
        "by reviewed examples for each method. Partial labels can be reported "
        "separately."
    )
    lines.append("")
    lines.append("## 3. Frame-Selection Analysis")
    lines.append("")
    lines.append(
        "Use the frame-selection analysis CSV together with saved montage "
        "images to discuss selected-frame visualizations, question relevance, "
        "and temporal coverage."
    )
    lines.append("")
    lines.append("## 4. Question-Type Analysis")
    lines.append("")
    lines.append(dataframe_to_markdown(by_type))
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def print_summary(summary: dict) -> None:
    print("\nFINAL EVALUATION")
    print("=" * 60)
    print(f"Examples: {summary['num_examples']}")

    print("\nMethods:")
    for method in METHODS:
        metrics = summary["methods"][method["key"]]
        print(
            f"  {method['label']}: "
            f"EM={metrics['exact_match_accuracy']:.2%}, "
            f"F1={metrics['mean_token_f1']:.4f}"
        )

    print("\nPairwise Changes:")
    for pair_key, pair in summary["pairwise"].items():
        method_a = next(
            method for method in METHODS if method["key"] == pair["method_a"]
        )
        method_b = next(
            method for method in METHODS if method["key"] == pair["method_b"]
        )
        print(
            f"  {method_a['label']} - {method_b['label']}: "
            f"EM {pair['accuracy_change_a_minus_b_percentage_points']:+.2f} pp, "
            f"F1 {pair['mean_token_f1_change_a_minus_b']:+.4f}, "
            f"McNemar p={pair['mcnemar_exact_test']['two_sided_p_value']:.6f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare baseline, Motion+YOLO, and Motion+YOLO+CLIP predictions."
        )
    )
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE_PATH)
    parser.add_argument("--uniform", type=Path, default=None)
    parser.add_argument(
        "--motion-yolo",
        type=Path,
        default=DEFAULT_MOTION_YOLO_PATH,
    )
    parser.add_argument(
        "--motion-yolo-clip",
        type=Path,
        default=DEFAULT_MOTION_YOLO_CLIP_PATH,
    )
    parser.add_argument(
        "--baseline-manifest",
        type=Path,
        default=DEFAULT_UNIFORM_MANIFEST_PATH,
    )
    parser.add_argument(
        "--motion-yolo-manifest",
        type=Path,
        default=DEFAULT_MOTION_YOLO_MANIFEST_PATH,
    )
    parser.add_argument(
        "--motion-yolo-clip-manifest",
        type=Path,
        default=DEFAULT_MOTION_YOLO_CLIP_MANIFEST_PATH,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-prefix", default=DEFAULT_OUTPUT_PREFIX)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    if path.is_absolute():
        return path
    return ROOT / path


def main() -> None:
    args = parse_args()

    baseline_arg = args.uniform if args.uniform is not None else args.baseline
    paths = {
        "baseline": resolve(baseline_arg),
        "motion_yolo": resolve(args.motion_yolo),
        "motion_yolo_clip": resolve(args.motion_yolo_clip),
    }
    manifest_paths = {
        "baseline": resolve(args.baseline_manifest),
        "motion_yolo": resolve(args.motion_yolo_manifest),
        "motion_yolo_clip": resolve(args.motion_yolo_clip_manifest),
    }

    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataframes = {
        method["key"]: load_jsonl(paths[method["key"]])
        for method in METHODS
    }

    comparison = create_comparison(dataframes)
    summary = summarize(comparison)
    by_type = summarize_by_question_type(comparison)
    manual_review = create_manual_review_template(comparison)
    manifests = {
        key: load_manifest(path)
        for key, path in manifest_paths.items()
    }
    frame_analysis = frame_selection_analysis(
        manifests=manifests,
        comparison=comparison,
    )
    sectioned_report = build_sectioned_report(
        summary=summary,
        by_type=by_type,
        frame_analysis=frame_analysis,
    )

    comparison_path = output_dir / f"{args.output_prefix}_detailed.csv"
    summary_path = output_dir / f"{args.output_prefix}_summary.json"
    by_type_path = output_dir / f"{args.output_prefix}_by_question_type.csv"
    review_path = output_dir / f"{args.output_prefix}_human_semantic_review.csv"
    frame_analysis_path = (
        output_dir / f"{args.output_prefix}_frame_selection_analysis.csv"
    )
    sectioned_report_path = output_dir / f"{args.output_prefix}_sectioned_report.json"
    markdown_report_path = output_dir / f"{args.output_prefix}_report.md"

    comparison.to_csv(comparison_path, index=False)
    by_type.to_csv(by_type_path, index=False)
    manual_review.to_csv(review_path, index=False)
    frame_analysis.to_csv(frame_analysis_path, index=False)

    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    with sectioned_report_path.open("w", encoding="utf-8") as file:
        json.dump(sectioned_report, file, indent=2)
    write_markdown_report(
        path=markdown_report_path,
        summary=summary,
        by_type=by_type,
    )

    print_summary(summary)

    print("\nBY QUESTION TYPE")
    print("=" * 60)
    print(by_type.to_string(index=False))

    print("\nSaved:")
    print(" ", comparison_path)
    print(" ", summary_path)
    print(" ", by_type_path)
    print(" ", review_path)
    print(" ", frame_analysis_path)
    print(" ", sectioned_report_path)
    print(" ", markdown_report_path)


if __name__ == "__main__":
    main()
