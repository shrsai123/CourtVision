'''import csv
import json
import zipfile
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
META_DIR = ROOT / "sportsqa_videos" / "meta-data"
ZIP_PATH = ROOT / "sportsqa_videos" / "basketball.zip"
OUT_DIR = ROOT / "data"
SAMPLE_DIR = ROOT / "sportsqa_videos" / "opened_samples" / "basketball"


def load_annotations():
    rows = []
    for split in ("train", "val", "test"):
        path = META_DIR / f"{split}.json"
        with path.open("r", encoding="utf-8") as f:
            for row in json.load(f):
                row = dict(row)
                row["split"] = split
                rows.append(row)
    return rows


def normalize_basketball_rows(rows):
    normalized = []
    for row in rows:
        if not row["video"].startswith("basketball/"):
            continue
        normalized.append(
            {
                "qa_id": row["qa_id"],
                "split": row["split"],
                "video_id": row["video"],
                "question": row["question"],
                "answer": row["answer"],
                "question_type": row["type"],
                "ans_cls": row["ans_cls"],
            }
        )
    return normalized


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

    all_rows = load_annotations()
    basketball_rows = normalize_basketball_rows(all_rows)
    qa_video_ids = {row["video_id"] for row in basketball_rows}

    with zipfile.ZipFile(ZIP_PATH) as zf:
        video_entries = [name for name in zf.namelist() if name.startswith("basketball/") and name.endswith(".avi")]
        archive_video_ids = {Path(name).with_suffix("").as_posix() for name in video_entries}

        joined_ids = sorted(qa_video_ids & archive_video_ids)
        missing_from_archive = sorted(qa_video_ids - archive_video_ids)
        archive_not_in_qa = sorted(archive_video_ids - qa_video_ids)

        sample_ids = []
        seen = set()
        for row in basketball_rows:
            video_id = row["video_id"]
            if video_id in joined_ids and video_id not in seen:
                sample_ids.append(video_id)
                seen.add(video_id)
            if len(sample_ids) == 10:
                break

        extracted = []
        for video_id in sample_ids:
            member = f"{video_id}.avi"
            target = SAMPLE_DIR / Path(member).name
            if not target.exists() or target.stat().st_size != zf.getinfo(member).file_size:
                with zf.open(member) as src, target.open("wb") as dst:
                    dst.write(src.read())
            extracted.append(str(target.relative_to(ROOT)))

    jsonl_path = OUT_DIR / "sportsqa_basketball_qa.jsonl"
    write_jsonl(jsonl_path, basketball_rows)

    with (OUT_DIR / "sportsqa_basketball_clip_samples.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["video_id", "local_path"])
        writer.writeheader()
        for video_id, local_path in zip(sample_ids, extracted):
            writer.writerow({"video_id": video_id, "local_path": local_path})

    report = {
        "source_annotations": [str((META_DIR / f"{split}.json").relative_to(ROOT)) for split in ("train", "val", "test")],
        "source_video_archive": str(ZIP_PATH.relative_to(ROOT)),
        "normalized_annotation_file": str(jsonl_path.relative_to(ROOT)),
        "format": ["video_id", "question", "answer", "question_type"],
        "original_keys": list(all_rows[0].keys()),
        "total_qa_rows": len(all_rows),
        "total_unique_videos": len({row["video"] for row in all_rows}),
        "basketball_qa_rows": len(basketball_rows),
        "basketball_unique_qa_videos": len(qa_video_ids),
        "basketball_archive_videos": len(archive_video_ids),
        "joined_basketball_videos": len(joined_ids),
        "missing_qa_videos_from_archive": missing_from_archive,
        "archive_videos_without_qa": archive_not_in_qa,
        "basketball_question_type_counts": dict(Counter(row["question_type"] for row in basketball_rows)),
        "sample_annotation": basketball_rows[0],
        "sample_clips": [{"video_id": video_id, "local_path": path} for video_id, path in zip(sample_ids, extracted)],
    }

    report_path = OUT_DIR / "sportsqa_basketball_join_report.json"
    with report_path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
    '''
import csv
import json
import random
import shutil
import zipfile
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

META_DIR = ROOT / "sportsqa_videos" / "meta-data"
ZIP_PATH = ROOT / "sportsqa_videos" / "basketball.zip"

OUT_DIR = ROOT / "data"
SAMPLE_DIR = (
    ROOT
    / "sportsqa_videos"
    / "opened_samples"
    / "basketball"
)

# Keep this at 20 while developing the pipeline.
# Change to 100–300 for the final experiment.
TARGET_QA_PAIRS = 60

# Use train while developing.
# Later, evaluate the frozen pipeline on val or test.
SELECTED_SPLITS = {"train"}

RANDOM_SEED = 42


def load_annotations():
    rows = []

    for split in ("train", "val", "test"):
        path = META_DIR / f"{split}.json"

        with path.open("r", encoding="utf-8") as file:
            annotations = json.load(file)

        for row in annotations:
            normalized_row = dict(row)
            normalized_row["split"] = split
            rows.append(normalized_row)

    return rows


def normalize_basketball_rows(rows):
    normalized = []

    for row in rows:
        video_id = row.get("video", "")

        if not video_id.startswith("basketball/"):
            continue

        normalized.append(
            {
                "qa_id": row["qa_id"],
                "split": row["split"],
                "video_id": video_id,
                "question": row["question"],
                "answer": row["answer"],
                "question_type": row["type"],
                "ans_cls": row["ans_cls"],
            }
        )

    return normalized


def write_jsonl(path, rows):
    with path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as file:
        for row in rows:
            file.write(
                json.dumps(row, ensure_ascii=False) + "\n"
            )


def select_qa_rows(
    basketball_rows,
    available_video_ids,
    target_count,
):
    eligible_rows = [
        row
        for row in basketball_rows
        if (
            row["video_id"] in available_video_ids
            and row["split"] in SELECTED_SPLITS
        )
    ]

    random_generator = random.Random(RANDOM_SEED)
    random_generator.shuffle(eligible_rows)

    selected_rows = eligible_rows[:target_count]

    if len(selected_rows) < target_count:
        raise ValueError(
            f"Requested {target_count} QA pairs, but only "
            f"{len(selected_rows)} eligible pairs were found."
        )

    return selected_rows


def extract_selected_videos(
    zip_file,
    selected_rows,
):
    unique_video_ids = sorted(
        {
            row["video_id"]
            for row in selected_rows
        }
    )

    video_paths = {}

    for video_id in unique_video_ids:
        member = f"{video_id}.avi"
        target = SAMPLE_DIR / Path(member).name

        archive_size = zip_file.getinfo(member).file_size

        needs_extraction = (
            not target.exists()
            or target.stat().st_size != archive_size
        )

        if needs_extraction:
            with zip_file.open(member) as source:
                with target.open("wb") as destination:
                    shutil.copyfileobj(
                        source,
                        destination,
                        length=1024 * 1024,
                    )

        video_paths[video_id] = str(
            target.relative_to(ROOT)
        )

    return video_paths


def write_subset_csv(
    selected_rows,
    video_paths,
):
    subset_path = OUT_DIR / "subset.csv"

    fieldnames = [
        "sample_id",
        "qa_id",
        "split",
        "video_id",
        "video_path",
        "question",
        "answer",
        "question_type",
        "ans_cls",
    ]

    with subset_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for sample_id, row in enumerate(
            selected_rows,
            start=1,
        ):
            writer.writerow(
                {
                    "sample_id": sample_id,
                    "qa_id": row["qa_id"],
                    "split": row["split"],
                    "video_id": row["video_id"],
                    "video_path": video_paths[
                        row["video_id"]
                    ],
                    "question": row["question"],
                    "answer": row["answer"],
                    "question_type": row[
                        "question_type"
                    ],
                    "ans_cls": row["ans_cls"],
                }
            )

    return subset_path


def validate_selected_data(
    selected_rows,
    video_paths,
):
    missing_files = []

    for video_id, relative_path in video_paths.items():
        full_path = ROOT / relative_path

        if (
            not full_path.exists()
            or full_path.stat().st_size == 0
        ):
            missing_files.append(
                {
                    "video_id": video_id,
                    "path": str(full_path),
                }
            )

    print("\nValidation")
    print("-" * 50)
    print("Selected QA pairs:", len(selected_rows))
    print("Unique videos:", len(video_paths))
    print("Missing or empty videos:", len(missing_files))
    print(
        "Question types:",
        dict(
            Counter(
                row["question_type"]
                for row in selected_rows
            )
        ),
    )
    print(
        "Splits:",
        dict(
            Counter(
                row["split"]
                for row in selected_rows
            )
        ),
    )

    if missing_files:
        raise FileNotFoundError(
            f"Some extracted videos are missing: "
            f"{missing_files[:5]}"
        )


def main():
    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    SAMPLE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not META_DIR.exists():
        raise FileNotFoundError(
            f"Metadata directory not found: {META_DIR}"
        )

    if not ZIP_PATH.exists():
        raise FileNotFoundError(
            f"Basketball ZIP not found: {ZIP_PATH}"
        )

    all_rows = load_annotations()
    basketball_rows = normalize_basketball_rows(all_rows)

    qa_video_ids = {
        row["video_id"]
        for row in basketball_rows
    }

    with zipfile.ZipFile(ZIP_PATH) as zip_file:
        video_entries = [
            name
            for name in zip_file.namelist()
            if (
                name.startswith("basketball/")
                and name.lower().endswith(".avi")
            )
        ]

        archive_video_ids = {
            Path(name).with_suffix("").as_posix()
            for name in video_entries
        }

        joined_ids = qa_video_ids & archive_video_ids

        missing_from_archive = sorted(
            qa_video_ids - archive_video_ids
        )

        archive_not_in_qa = sorted(
            archive_video_ids - qa_video_ids
        )

        selected_rows = select_qa_rows(
            basketball_rows=basketball_rows,
            available_video_ids=joined_ids,
            target_count=TARGET_QA_PAIRS,
        )

        video_paths = extract_selected_videos(
            zip_file=zip_file,
            selected_rows=selected_rows,
        )

    # Save every normalized basketball QA pair.
    jsonl_path = (
        OUT_DIR / "sportsqa_basketball_qa.jsonl"
    )

    write_jsonl(
        jsonl_path,
        basketball_rows,
    )

    # Save the selected experiment subset.
    subset_path = write_subset_csv(
        selected_rows=selected_rows,
        video_paths=video_paths,
    )

    validate_selected_data(
        selected_rows=selected_rows,
        video_paths=video_paths,
    )

    report = {
        "source_annotations": [
            str(
                (
                    META_DIR / f"{split}.json"
                ).relative_to(ROOT)
            )
            for split in ("train", "val", "test")
        ],
        "source_video_archive": str(
            ZIP_PATH.relative_to(ROOT)
        ),
        "normalized_annotation_file": str(
            jsonl_path.relative_to(ROOT)
        ),
        "subset_file": str(
            subset_path.relative_to(ROOT)
        ),
        "target_qa_pairs": TARGET_QA_PAIRS,
        "selected_splits": sorted(SELECTED_SPLITS),
        "random_seed": RANDOM_SEED,
        "total_qa_rows": len(all_rows),
        "total_unique_videos": len(
            {
                row["video"]
                for row in all_rows
            }
        ),
        "basketball_qa_rows": len(
            basketball_rows
        ),
        "basketball_unique_qa_videos": len(
            qa_video_ids
        ),
        "basketball_archive_videos": len(
            archive_video_ids
        ),
        "joined_basketball_videos": len(
            joined_ids
        ),
        "selected_qa_pairs": len(
            selected_rows
        ),
        "selected_unique_videos": len(
            video_paths
        ),
        "selected_question_type_counts": dict(
            Counter(
                row["question_type"]
                for row in selected_rows
            )
        ),
        "all_basketball_question_type_counts": dict(
            Counter(
                row["question_type"]
                for row in basketball_rows
            )
        ),
        "missing_qa_videos_from_archive": (
            missing_from_archive
        ),
        "archive_videos_without_qa": (
            archive_not_in_qa
        ),
        "sample_annotation": selected_rows[0],
        "sample_clips": [
            {
                "video_id": video_id,
                "local_path": local_path,
            }
            for video_id, local_path
            in video_paths.items()
        ],
    }

    report_path = (
        OUT_DIR
        / "sportsqa_basketball_join_report.json"
    )

    with report_path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as file:
        json.dump(
            report,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("\nDataset preparation completed.")
    print("Subset CSV:", subset_path)
    print("Join report:", report_path)


if __name__ == "__main__":
    main()