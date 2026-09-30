# CourtVision

## Project Team

Group members: Shreyas Sai Raman

## Project Description

CourtVision is an experimental basketball VideoQA system that studies whether computer-vision-guided keyframe selection can improve question answering on Sports-QA basketball videos. The project compares three frame selection strategies: uniform sampling, Motion + YOLO sampling, and question-aware Motion + YOLO + CLIP sampling. Each method selects eight frames from the same basketball clip, passes those frames and the Sports-QA question to Qwen3-VL, and evaluates the resulting answer against the dataset ground truth.

The Gradio demo is Sports-QA-first. It loads precomputed frame selections, Qwen3-VL predictions, and evaluation results so examples switch instantly instead of rerunning expensive model inference. A separate basketball analytics module adds player/person tracking, pixel-based movement metrics, spacing metrics, tracked-video visualizations, and occupancy heatmaps. These analytics are separate from the VideoQA sampling and evaluation pipeline.

## Demo and Dataset URLs

- Local Gradio demo: `http://127.0.0.1:7860` after running `py -3.11 scripts\app.py`
- Sports-QA videos: https://huggingface.co/datasets/HopLeeTop/Sports-QA
- Sports-QA metadata and features: https://figshare.com/articles/dataset/Sports-QA_Dataset/29992531
- Sports-QA paper: https://arxiv.org/abs/2401.01505
- Presentation video: https://www.youtube.com/watch?v=y_ZGYXBsTuI

## Project Structure

```text
CourtVision/
  data/                         Sports-QA subset, frame manifests, saved frames
  outputs/
    predictions/                Saved Qwen3-VL predictions
    evaluation/                 Evaluation reports and CSVs
    analytics/                  Tracking, metrics, heatmaps, visualizations
  scripts/
    app.py                      Main Gradio app
    prepare_sportsqa_basketball.py
    sample_uniform_frames.py
    sample_guided_frames.py
    sample_guided_frames_clip.py
    run_uniform_baseline.py
    run_guided_frames.py
    run_guided_frames_clip.py
    evaluate_methods.py
    analyze_basketball_video.py
    calculate_basketball_analytics.py
    visualize_basketball_analytics.py
    detect_activity_windows.py
```

## Setup

Create and activate a Python environment, then install dependencies:

```powershell
py -3.11 -m pip install -r requirements.txt
```

The project expects the Sports-QA basketball videos and metadata to already be present in:

```text
sportsqa_videos/
data/subset.csv
```

The existing YOLO model is stored at:

```text
scripts/yolo11n.pt
```

## Run the Gradio Demo

```powershell
py -3.11 scripts\app.py
```

Open the local URL printed by Gradio, usually:

```text
http://127.0.0.1:7860
```

The demo has three main tabs:

- Example Explorer: compare frame selections and Qwen3-VL predictions for one Sports-QA question.
- Evaluation: show exact match and token F1 results across the 60-question subset.
- Analytics: show precomputed tracking visualizations, occupancy heatmaps, and pixel-based movement/spacing tables.

## VideoQA Pipeline

The VideoQA pipeline is based on precomputed artifacts for the demo. To regenerate them, run the scripts in this general order:

```powershell
py -3.11 scripts\prepare_sportsqa_basketball.py
py -3.11 scripts\sample_uniform_frames.py
py -3.11 scripts\sample_guided_frames.py
py -3.11 scripts\sample_guided_frames_clip.py
py -3.11 scripts\run_uniform_baseline.py
py -3.11 scripts\run_guided_frames.py
py -3.11 scripts\run_guided_frames_clip.py
py -3.11 scripts\evaluate_methods.py
```

The demo does not rerun Qwen3-VL when an example is selected. It reads saved predictions from:

```text
outputs/predictions/
```

## Basketball Analytics Pipeline

The analytics pipeline is separate from the VideoQA experiment. It does not modify the Uniform, Motion + YOLO, Motion + YOLO + CLIP, Qwen inference, or evaluation scripts.

Run analytics in this order:

```powershell
py -3.11 scripts\analyze_basketball_video.py
py -3.11 scripts\calculate_basketball_analytics.py
py -3.11 scripts\visualize_basketball_analytics.py --overwrite
```

Optional event-window detection is implemented but not displayed in the Gradio app:

```powershell
py -3.11 scripts\detect_activity_windows.py
```

Analytics outputs are saved under:

```text
outputs/analytics/tracks/
outputs/analytics/player_metrics/
outputs/analytics/frame_metrics/
outputs/analytics/visualizations/
outputs/analytics/heatmaps/
outputs/analytics/events/
```

All movement and spacing values are pixel-based. The project does not perform court calibration, team assignment, possession tracking, or real-world speed estimation.

## Notes and Limitations

- Sports-QA clips in this project are visual-only; audio is disabled in the Gradio player.
- YOLO class `person` can include referees, bench personnel, or spectators.
- Analytics counts should be read as tracked person boxes, not verified on-court players.
- The upload/custom-video workflow is not the centerpiece of this project; the main demo is the Sports-QA research comparison.
