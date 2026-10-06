#!/usr/bin/env python3
"""Compose three honest 30-second weekly-report videos from Isaac Lab captures."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
VIDEO_DIR = ROOT / "data/teacher/videos"
FPS = 25
WIDTH = 1280
HEIGHT = 720


def _card(title: str, subtitle: str, *, frames: int) -> list[np.ndarray]:
    result = []
    for _ in range(frames):
        frame = np.full((HEIGHT, WIDTH, 3), (24, 30, 42), dtype=np.uint8)
        cv2.putText(
            frame,
            title,
            (70, 315),
            cv2.FONT_HERSHEY_DUPLEX,
            1.25,
            (245, 248, 252),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            subtitle,
            (72, 365),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (126, 211, 255),
            2,
            cv2.LINE_AA,
        )
        result.append(frame)
    return result


def _overlay(
    frame: np.ndarray,
    title: str,
    subtitle: str,
    footer: str,
) -> np.ndarray:
    frame = frame.copy()
    top = frame[:104].copy()
    top[:] = (12, 17, 25)
    frame[:104] = cv2.addWeighted(frame[:104], 0.18, top, 0.82, 0.0)
    bottom = frame[662:].copy()
    bottom[:] = (12, 17, 25)
    frame[662:] = cv2.addWeighted(frame[662:], 0.22, bottom, 0.78, 0.0)
    cv2.putText(
        frame,
        title,
        (34, 44),
        cv2.FONT_HERSHEY_DUPLEX,
        0.86,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        subtitle,
        (35, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.54,
        (126, 211, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        footer,
        (35, 699),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (235, 240, 246),
        1,
        cv2.LINE_AA,
    )
    return frame


def _frames(path: Path, *, start: int = 0, end: int | None = None):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if index >= start and (end is None or index < end):
                yield frame
            index += 1
            if end is not None and index >= end:
                break
    finally:
        capture.release()


def _writer(path: Path):
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


def compose_teacher() -> dict:
    output = VIDEO_DIR / "01_production_teacher_30s.mp4"
    sources = [
        (VIDEO_DIR / "01_teacher_clip_a75.raw-step-0.mp4", 4, None, "Demo A"),
        (VIDEO_DIR / "01_teacher_clip_b75.raw-step-0.mp4", 4, None, "Demo B"),
        (VIDEO_DIR / "01_production_teacher.raw-step-0.mp4", 412, 670, "Demo C"),
    ]
    count = 0
    with _writer(output) as writer:
        for frame in _card(
            "G1 43-DOF PRODUCTION TEACHER",
            "Representative x/y-retargeted Box grasp and lift rollouts",
            frames=50,
        ):
            _append(writer, frame)
            count += 1
        for path, start, end, label in sources:
            for frame in _frames(path, start=start, end=end):
                frame = _overlay(
                    frame,
                    f"PRODUCTION TEACHER  |  {label}",
                    "Box x/y retargeting  |  Bilateral physical grasp  |  No Box teleport",
                    "Corpus gate: 500/500 accepted, >=12 cm lift, 100-step hold  |  Filming cutoff: 75 hold steps",
                )
                _append(writer, frame)
                count += 1
        for frame in _card(
            "500 / 500 STRICT DEMONSTRATIONS",
            "Production corpus complete  |  Aggregate teacher success rate: 100%",
            frames=25,
        ):
            _append(writer, frame)
            count += 1
    return {"output": str(output), "frames": count, "seconds": count / FPS}


def compose_bc2() -> dict:
    source = VIDEO_DIR / "02_bc2_student.raw-step-0.mp4"
    output = VIDEO_DIR / "02_bc2_student_grasp_lift_30s.mp4"
    count = 0
    with _writer(output) as writer:
        for index, frame in enumerate(_frames(source, start=0, end=750)):
            episode = min(3, index // 261 + 1)
            frame = _overlay(
                frame,
                f"BC2  |  GRASP & LIFT STUDENT POLICY  |  Rollout {episode}",
                "517-D QR/proprio observation  |  43-D joint target  |  No privileged student input",
                "Independent physical certification: 20/20  |  >=12 cm lift  |  100-step bilateral hold",
            )
            _append(writer, frame)
            count += 1
    return {"output": str(output), "frames": count, "seconds": count / FPS}


def compose_carry_place() -> dict:
    carry = VIDEO_DIR / "03_carry.raw-step-0.mp4"
    place = VIDEO_DIR / "03_place_high.raw-step-0.mp4"
    output = VIDEO_DIR / "03_carry_place_phasewise_30s.mp4"
    count = 0
    with _writer(output) as writer:
        for frame in _card(
            "BC3 CARRY + BC4 PLACE v4",
            "Phase-wise physical rollouts from provenance-checked snapshots",
            frames=50,
        ):
            _append(writer, frame)
            count += 1
        for frame in _frames(carry, start=4):
            frame = _overlay(
                frame,
                "BC3  |  CARRY",
                "Start: certified BC2 bilateral-hold physical snapshot",
                "Box teleport: False  |  Loaded grasp lock: Active  |  Target: Bin staging pose",
            )
            _append(writer, frame)
            count += 1
        for frame in _card(
            "PHYSICAL SNAPSHOT HANDOFF",
            "BC3 carry-at-staging state -> BC4 Place v4",
            frames=38,
        ):
            _append(writer, frame)
            count += 1
        for frame in _frames(place, start=4):
            frame = _overlay(
                frame,
                "BC4  |  PLACE v4",
                "Start: physical carry-at-staging snapshot  |  Loaded-arm offset decay",
                "Release: Detected  |  Box settled in Bin  |  Result: PLACE SUCCESS",
            )
            _append(writer, frame)
            count += 1
        for frame in _card(
            "PHASE-WISE PHYSICAL SUCCESS",
            "Full BC1 -> BC2 -> BC3 -> BC4 end-to-end FSM rollout still pending",
            frames=50,
        ):
            _append(writer, frame)
            count += 1
    return {"output": str(output), "frames": count, "seconds": count / FPS}


def main() -> None:
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    results = {
        "format": "unitree_g1_43d_weekly_video_manifest_v1",
        "fps": FPS,
        "resolution": [WIDTH, HEIGHT],
        "videos": [compose_teacher(), compose_bc2(), compose_carry_place()],
        "disclosure": (
            "Videos 1 and 2 are independent grasp/lift rollouts. Video 3 is a "
            "phase-wise snapshot replay, not a full end-to-end FSM success."
        ),
    }
    manifest = VIDEO_DIR / "weekly_video_manifest.json"
    manifest.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
