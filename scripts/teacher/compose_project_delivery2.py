#!/usr/bin/env python3
"""Build the incremental project_delivery2 presentation assets."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np


FPS = 25
WIDTH = 1280
HEIGHT = 720
BACKGROUND = (30, 35, 47)
WHITE = (245, 248, 252)
GREEN = (105, 220, 145)
ORANGE = (80, 175, 255)
RED = (95, 105, 245)


def writer(path: Path):
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


def append(output, frame: np.ndarray) -> None:
    output.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def text_center(frame: np.ndarray, value: str, y: int, scale: float, color=WHITE) -> None:
    font = cv2.FONT_HERSHEY_DUPLEX
    thickness = 2
    width = cv2.getTextSize(value, font, scale, thickness)[0][0]
    cv2.putText(
        frame,
        value,
        ((WIDTH - width) // 2, y),
        font,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def card(title: str, subtitle: str = "", *, seconds: float = 1.4, color=WHITE):
    for _ in range(round(FPS * seconds)):
        frame = np.full((HEIGHT, WIDTH, 3), BACKGROUND, dtype=np.uint8)
        text_center(frame, title, 330, 1.18, color)
        if subtitle:
            text_center(frame, subtitle, 395, 0.72, WHITE)
        yield frame


def video_frames(path: Path, start: int, end: int, stride: int = 2):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    index = 0
    try:
        while index < end:
            ok, frame = capture.read()
            if not ok:
                break
            if index >= start and (index - start) % stride == 0:
                yield cv2.resize(frame, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
            index += 1
    finally:
        capture.release()


def overlay(frame: np.ndarray, title: str, subtitle: str = "") -> np.ndarray:
    frame = frame.copy()
    veil = np.full((94, WIDTH, 3), (12, 17, 25), dtype=np.uint8)
    frame[:94] = cv2.addWeighted(frame[:94], 0.18, veil, 0.82, 0.0)
    cv2.putText(
        frame,
        title,
        (30, 43),
        cv2.FONT_HERSHEY_DUPLEX,
        0.82,
        WHITE,
        2,
        cv2.LINE_AA,
    )
    if subtitle:
        cv2.putText(
            frame,
            subtitle,
            (31, 75),
            cv2.FONT_HERSHEY_DUPLEX,
            0.56,
            (205, 214, 225),
            1,
            cv2.LINE_AA,
        )
    return frame


def emit_segment(output, path: Path, start: int, end: int, title: str, subtitle: str = "") -> int:
    count = 0
    for frame in video_frames(path, start, end, stride=2):
        append(output, overlay(frame, title, subtitle))
        count += 1
    return count


def result_card(*, seconds: float = 2.0):
    yield from card(
        "ROLLOUT RESULT: FULL TASK SUCCESS",
        "Approach > Grasp & Lift > Stabilize > Carry > Place",
        seconds=seconds,
        color=GREEN,
    )


def full_fsm(delivery: Path, raw: Path, episode: dict) -> dict:
    output_path = delivery / "videos/01_student_full_fsm_success.mp4"
    steps = int(episode["steps"])
    transitions = episode["transition_steps"]
    boundaries = [
        (0, int(transitions["approach"]), "TASK 1 | BOX APPROACH"),
        (int(transitions["approach"]), int(transitions["grasp_lift"]), "TASK 2 | BIMANUAL GRASP AND LIFT"),
        (int(transitions["grasp_lift"]), int(transitions["loaded_stand"]), "LOADED STABILIZATION"),
        (int(transitions["loaded_stand"]), int(transitions["carry"]), "TASK 3 | LOADED BOX CARRY"),
        (int(transitions["carry"]), steps, "TASK 4 | BOX PLACEMENT IN BIN"),
    ]
    count = 0
    with writer(output_path) as output:
        for frame in card("STUDENT POLICY | FULL FSM SUCCESS", "One continuous physics rollout"):
            append(output, frame)
            count += 1
        for start, end, title in boundaries:
            count += emit_segment(output, raw, start, end, title)
        for frame in result_card():
            append(output, frame)
            count += 1
    return {"path": str(output_path), "seconds": count / FPS, "content": "actual Student full-task success"}


def stabilization_and_carry(delivery: Path, raw: Path, episode: dict) -> dict:
    output_path = delivery / "videos/02_loaded_stabilization_and_carry.mp4"
    t = episode["transition_steps"]
    count = 0
    with writer(output_path) as output:
        for frame in card("LOADED STABILIZATION AND CARRY", "Hierarchical upper/lower-body control"):
            append(output, frame)
            count += 1
        count += emit_segment(
            output,
            raw,
            int(t["grasp_lift"]),
            int(t["handoff_survival"]),
            "2 s HANDOFF SURVIVAL",
            "Hands and upper body locked | zero walking command",
        )
        count += emit_segment(
            output,
            raw,
            int(t["handoff_survival"]),
            int(t["loaded_stand"]),
            "5 s LOADED STAND",
            "Live lower-body balance | grasp target retained",
        )
        count += emit_segment(
            output,
            raw,
            int(t["loaded_stand"]),
            int(t["carry"]),
            "LOADED BOX CARRY",
            "23-D Student command | hands remain locked",
        )
        for frame in card("CARRY GATE: SUCCESS", "Staging target reached with the box retained", seconds=2.0, color=GREEN):
            append(output, frame)
            count += 1
    return {"path": str(output_path), "seconds": count / FPS, "content": "actual Student stabilization and Carry"}


def before_after(delivery: Path, raw: Path, episode: dict, baseline_raw: Path, baseline_metrics: dict) -> dict:
    output_path = delivery / "videos/03_carry_before_after_comparison.mp4"
    baseline_episode = baseline_metrics["episode_results"][0]
    baseline_end = int(baseline_episode["steps"])
    baseline_grasp = int(baseline_episode["transition_steps"]["grasp_lift"])
    t = episode["transition_steps"]
    count = 0
    with writer(output_path) as output:
        for frame in card("STUDENT CARRY | BEFORE AND AFTER", "Same Unitree G1 43-DOF physics environment"):
            append(output, frame)
            count += 1
        for frame in video_frames(baseline_raw, max(0, baseline_grasp - 35), baseline_end, stride=1):
            append(output, overlay(frame, "BEFORE | DIRECT CARRY TRANSITION", "Box dropped shortly after grasp"))
            count += 1
        for frame in card("METHOD UPDATE", "2 s handoff + 5 s loaded stand + split-body Carry", seconds=1.5, color=ORANGE):
            append(output, frame)
            count += 1
        count += emit_segment(
            output,
            raw,
            int(t["grasp_lift"]),
            int(t["carry"]),
            "AFTER | HIERARCHICAL CARRY",
            "Box retained through stabilization and walking",
        )
        for frame in card("20-ROLLOUT RESULT", "P(Carry | Grasp): 0% > 64.3%", seconds=2.2, color=GREEN):
            append(output, frame)
            count += 1
    return {"path": str(output_path), "seconds": count / FPS, "content": "actual Student Carry before/after comparison"}


def latest_place_teacher(delivery: Path, raw: Path, metrics: dict) -> dict:
    """Compose the latest provenance-checked Task 4 Teacher rollout."""
    results = metrics.get("results", [])
    if len(results) != 1 or results[0].get("reason") != "place_success":
        raise ValueError("latest Place video requires one verified place_success rollout")
    output_path = delivery / "videos/03_latest_place_teacher_demonstration.mp4"
    source_steps = int(results[0]["steps"])
    count = 0
    with writer(output_path) as output:
        for frame in card(
            "TEACHER DEMONSTRATION | TASK 4",
            "Box Placement in Bin",
            seconds=1.6,
        ):
            append(output, frame)
            count += 1
        # Repeat each frame once to make the short release-and-settle motion
        # easier to inspect in a presentation.
        for frame in video_frames(raw, 0, source_steps + 1, stride=1):
            annotated = overlay(
                frame,
                "TASK 4 | BOX PLACEMENT IN BIN",
                "Physical Teacher rollout used by the latest 500-demo collection",
            )
            append(output, annotated)
            append(output, annotated)
            count += 2
        for frame in card(
            "TEACHER RESULT: PLACE SUCCESS",
            "500 accepted demonstrations from 651 physical rollouts",
            seconds=2.4,
            color=GREEN,
        ):
            append(output, frame)
            count += 1
    return {
        "path": str(output_path),
        "seconds": count / FPS,
        "content": "latest physical Teacher box-placement success",
    }


def progress_image(delivery: Path) -> dict:
    image = np.full((HEIGHT, WIDTH, 3), BACKGROUND, dtype=np.uint8)
    cv2.putText(image, "G1 43-DOF LOCO-MANIPULATION | ADDITIONAL RESULTS", (45, 72), cv2.FONT_HERSHEY_DUPLEX, 0.92, WHITE, 2, cv2.LINE_AA)
    boxes = [
        ("500 STRICT DEMOS", "Approach + Grasp"),
        ("DAgger", "Student failure correction"),
        ("SPLIT-BODY CARRY", "100 Teacher successes"),
        ("PLACE TEACHER", "500 successes"),
        ("FULL STUDENT FSM", "20 physics rollouts"),
    ]
    x_positions = [38, 286, 534, 782, 1030]
    for index, ((title, subtitle), x) in enumerate(zip(boxes, x_positions)):
        color = (68, 83, 106) if index < 4 else (57, 112, 79)
        cv2.rectangle(image, (x, 170), (x + 210, 330), color, -1)
        cv2.putText(image, title, (x + 14, 225), cv2.FONT_HERSHEY_DUPLEX, 0.52, WHITE, 1, cv2.LINE_AA)
        cv2.putText(image, subtitle, (x + 14, 275), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (215, 223, 233), 1, cv2.LINE_AA)
        if index < len(boxes) - 1:
            cv2.arrowedLine(image, (x + 212, 250), (x + 244, 250), ORANGE, 2, tipLength=0.25)
    metrics = [
        ("Approach", "100%"),
        ("Grasp | Approach", "70.0%"),
        ("Handoff | Grasp", "100%"),
        ("Loaded Stand | Handoff", "92.9%"),
        ("Carry | Grasp", "64.3%"),
        ("Place | Carry", "55.6%"),
        ("Full Task", "25.0%"),
    ]
    for index, (name, value) in enumerate(metrics):
        column = index % 4
        row = index // 4
        x = 64 + column * 300
        y = 440 + row * 105
        cv2.putText(image, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (205, 214, 225), 1, cv2.LINE_AA)
        cv2.putText(image, value, (x, y + 42), cv2.FONT_HERSHEY_DUPLEX, 0.83, GREEN if name != "Full Task" else ORANGE, 2, cv2.LINE_AA)
    output = delivery / "images/additional_research_pipeline.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), image)
    return {"path": str(output), "width": WIDTH, "height": HEIGHT}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delivery", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--latest-place-raw", type=Path, required=True)
    parser.add_argument("--latest-place-metrics", type=Path, required=True)
    args = parser.parse_args()
    delivery = args.delivery.resolve()
    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    episodes = metrics["episode_results"]
    if len(episodes) != 1 or episodes[0]["reason"] != "full_task_success":
        raise ValueError("delivery video requires one verified full_task_success episode")
    place_metrics = json.loads(args.latest_place_metrics.read_text(encoding="utf-8"))
    results = {
        "video_1": full_fsm(delivery, args.raw, episodes[0]),
        "video_2": stabilization_and_carry(delivery, args.raw, episodes[0]),
        "video_3": latest_place_teacher(delivery, args.latest_place_raw, place_metrics),
        "representative_image": progress_image(delivery),
        "provenance": {
            "student_fsm_raw_video": str(
                (delivery / "sources/student_hierarchical_fsm_success.raw-step-0.mp4").resolve()
            ),
            "student_fsm_metrics": str(
                (delivery / "sources/student_hierarchical_fsm_success.metrics.json").resolve()
            ),
            "latest_place_teacher_raw_video": str(
                (delivery / "sources/latest_place_teacher_success.raw-step-0.mp4").resolve()
            ),
            "latest_place_teacher_metrics": str(
                (delivery / "sources/latest_place_teacher_success.metrics.json").resolve()
            ),
        },
    }
    manifest = delivery / "video_manifest.json"
    manifest.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    shutil.copy2(
        args.latest_place_metrics,
        delivery / "sources/latest_place_teacher_success.metrics.json",
    )
    shutil.copy2(
        args.latest_place_raw,
        delivery / "sources/latest_place_teacher_success.raw-step-0.mp4",
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
