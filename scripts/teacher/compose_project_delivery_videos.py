#!/usr/bin/env python3
"""Compose concise Teacher-task and honest single-rollout Student FSM videos."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = ROOT / "data/teacher/videos"
DEFAULT_DELIVERY = ROOT.parent / "project_delivery"
FPS = 25
WIDTH = 1280
HEIGHT = 720


TASKS = {
    1: "BOX APPROACH",
    2: "BIMANUAL GRASP AND LIFT",
    3: "LOADED BOX CARRY",
    4: "BOX PLACEMENT IN BIN",
}


def _writer(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    return imageio.get_writer(
        path,
        fps=FPS,
        codec="libx264",
        quality=8,
        pixelformat="yuv420p",
        macro_block_size=None,
        ffmpeg_log_level="warning",
    )


def _append(writer, frame: np.ndarray) -> None:
    writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def _centered_text(frame: np.ndarray, text: str, y: int, scale: float) -> None:
    font = cv2.FONT_HERSHEY_DUPLEX
    thickness = 2
    width = cv2.getTextSize(text, font, scale, thickness)[0][0]
    cv2.putText(
        frame,
        text,
        ((WIDTH - width) // 2, y),
        font,
        scale,
        (245, 248, 252),
        thickness,
        cv2.LINE_AA,
    )


def _task_card(task: int, *, frames: int = 30) -> list[np.ndarray]:
    result = []
    for _ in range(frames):
        frame = np.full((HEIGHT, WIDTH, 3), (24, 30, 42), dtype=np.uint8)
        _centered_text(frame, f"TASK {task}", 320, 1.05)
        _centered_text(frame, TASKS[task], 382, 1.25)
        result.append(frame)
    return result


def _task_overlay(frame: np.ndarray, task: int) -> np.ndarray:
    frame = frame.copy()
    veil = np.full((82, WIDTH, 3), (12, 17, 25), dtype=np.uint8)
    frame[:82] = cv2.addWeighted(frame[:82], 0.16, veil, 0.84, 0.0)
    cv2.putText(
        frame,
        f"TASK {task}  |  {TASKS[task]}",
        (34, 52),
        cv2.FONT_HERSHEY_DUPLEX,
        0.92,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return frame


def _frames(
    path: Path,
    *,
    start: int = 0,
    end: int | None = None,
    stride: int = 1,
):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if (
                index >= start
                and (end is None or index < end)
                and (index - start) % stride == 0
            ):
                yield frame
            index += 1
            if end is not None and index >= end:
                break
    finally:
        capture.release()


def compose_teacher(delivery: Path) -> dict:
    output = delivery / "videos/01_teacher_demonstrations_four_tasks.mp4"
    # Task 1 and Task 2 are non-overlapping segments of one physical Teacher
    # trajectory. Carry and Place retain their full successful phase rollouts.
    sources = {
        1: (SOURCE_DIR / "01_production_teacher.raw-step-0.mp4", 4, 105, 1),
        2: (SOURCE_DIR / "01_production_teacher.raw-step-0.mp4", 105, 315, 2),
        3: (SOURCE_DIR / "03_carry.raw-step-0.mp4", 4, None, 2),
        4: (SOURCE_DIR / "03_place_high.raw-step-0.mp4", 4, None, 2),
    }
    count = 0
    with _writer(output) as writer:
        for task in range(1, 5):
            for frame in _task_card(task):
                _append(writer, frame)
                count += 1
            path, start, end, stride = sources[task]
            for frame in _frames(path, start=start, end=end, stride=stride):
                _append(writer, _task_overlay(frame, task))
                count += 1
    return {
        "output": str(output),
        "frames": count,
        "seconds": count / FPS,
        "content": "four physical Teacher task demonstrations",
    }


def _result_card(episode: dict, *, frames: int = 75) -> list[np.ndarray]:
    reached = set(episode.get("transitions", []))
    statuses = {
        1: "SUCCESS" if "approach" in reached else "FAILED",
        2: "SUCCESS" if "grasp_lift" in reached else "NOT REACHED",
        3: "SUCCESS" if "carry" in reached else (
            "FAILED" if "grasp_lift" in reached else "NOT REACHED"
        ),
        4: "SUCCESS" if "place" in reached else "NOT REACHED",
    }
    success = episode.get("reason") == "full_task_success"
    result = []
    for _ in range(frames):
        frame = np.full((HEIGHT, WIDTH, 3), (24, 30, 42), dtype=np.uint8)
        color = (90, 220, 130) if success else (90, 110, 255)
        title = "ROLLOUT RESULT: SUCCESS" if success else "ROLLOUT RESULT: FAILURE"
        width = cv2.getTextSize(title, cv2.FONT_HERSHEY_DUPLEX, 1.15, 2)[0][0]
        cv2.putText(
            frame, title, ((WIDTH - width) // 2, 230),
            cv2.FONT_HERSHEY_DUPLEX, 1.15, color, 2, cv2.LINE_AA,
        )
        lines = [f"TASK {task}: {statuses[task]}" for task in range(1, 5)]
        for index, line in enumerate(lines):
            _centered_text(frame, line, 315 + index * 58, 0.82)
        reason = str(episode.get("reason", "unknown")).replace("_", " ").upper()
        _centered_text(frame, f"FINAL EVENT: {reason}", 585, 0.72)
        result.append(frame)
    return result


def compose_student(delivery: Path, raw_video: Path, metrics_path: Path) -> dict:
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    episodes = metrics.get("episode_results", [])
    if len(episodes) != 1:
        raise ValueError("Student delivery video requires exactly one FSM episode")
    episode = episodes[0]
    transitions = episode.get("transition_steps", {})
    end = int(episode.get("steps", 0))
    if end <= 0:
        raise ValueError("FSM metrics do not contain a positive episode length")
    approach_end = min(end, int(transitions.get("approach", end)) + 1)
    grasp_end = min(end, int(transitions.get("grasp_lift", end)) + 1)
    carry_end = min(end, int(transitions.get("carry", end)) + 1)
    segments = [(1, 0, approach_end)]
    if approach_end < end:
        segments.append((2, approach_end, grasp_end))
    if grasp_end < end:
        segments.append((3, grasp_end, carry_end if carry_end > grasp_end else end))
    if carry_end < end and "carry" in transitions:
        segments.append((4, carry_end, end))

    output = delivery / "videos/02_student_fsm_single_rollout.mp4"
    count = 0
    with _writer(output) as writer:
        for task, start, stop in segments:
            for frame in _task_card(task):
                _append(writer, frame)
                count += 1
            # Raw RecordVideo is 50 FPS; stride 2 preserves real-time motion in
            # the 25 FPS delivery file.
            for frame in _frames(raw_video, start=start, end=stop, stride=2):
                _append(writer, _task_overlay(frame, task))
                count += 1
        for frame in _result_card(episode):
            _append(writer, frame)
            count += 1
    return {
        "output": str(output),
        "frames": count,
        "seconds": count / FPS,
        "source_raw_video": str(raw_video),
        "source_metrics": str(metrics_path),
        "result": episode.get("reason"),
        "transitions": episode.get("transitions", []),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delivery", type=Path, default=DEFAULT_DELIVERY)
    parser.add_argument("--student_raw_video", type=Path)
    parser.add_argument("--student_metrics", type=Path)
    args = parser.parse_args()
    delivery = args.delivery.expanduser().resolve()
    results = {"teacher": compose_teacher(delivery)}
    if args.student_raw_video is not None or args.student_metrics is not None:
        if args.student_raw_video is None or args.student_metrics is None:
            parser.error("--student_raw_video and --student_metrics are required together")
        results["student"] = compose_student(
            delivery,
            args.student_raw_video.expanduser().resolve(),
            args.student_metrics.expanduser().resolve(),
        )
    manifest = delivery / "video_manifest.json"
    manifest.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
