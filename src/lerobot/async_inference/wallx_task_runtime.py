import importlib
import logging
import os
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .helpers import RawObservation

DEFAULT_FIXED_IMAGE_KEY = "observation.images.fixed"
FIXED_IMAGE_KEY_ENV = "LEROBOT_WALLX_FIXED_IMAGE_KEY"
PLANNER_API_KEY_ENV = "SILICONFLOW_API_KEY"
SUBTASK_TIMEOUT_SEC = 15.0


@dataclass
class WallXSubTask:
    plan_id: str
    subtask_id: int
    digit: str
    target_xy: tuple[int, int]
    instruction_en: str
    instruction_zh: str
    timeout_sec: float = SUBTASK_TIMEOUT_SEC


class WallXTaskOrchestrator:
    def __init__(self, logger: logging.Logger, fixed_image_key: str | None = None):
        self.logger = logger
        self.fixed_image_key = fixed_image_key or os.getenv(FIXED_IMAGE_KEY_ENV, DEFAULT_FIXED_IMAGE_KEY)
        self._resolved_fixed_key: str | None = None
        self._plan_id: str | None = None
        self._result_str: str | None = None
        self._subtasks: list[WallXSubTask] = []
        self._current_index = 0
        self._current_started_at: float | None = None
        self._bootstrapped = False
        self._finished = False

    def _candidate_fixed_keys(self) -> list[str]:
        candidates = [self.fixed_image_key]
        if self.fixed_image_key.startswith("observation.images."):
            candidates.append(self.fixed_image_key.removeprefix("observation.images."))
        candidates.extend(
            [
                DEFAULT_FIXED_IMAGE_KEY,
                "fixed",
                "observation.images.face_view",
                "face_view",
            ]
        )
        return list(dict.fromkeys(candidates))

    def _extract_fixed_frame(self, raw_observation: RawObservation) -> np.ndarray:
        for key in self._candidate_fixed_keys():
            frame = raw_observation.get(key)
            if frame is None:
                continue
            self._resolved_fixed_key = key
            return np.asarray(frame)
        raise RuntimeError("Missing fixed image in observation.")

    def _save_temp_frame(self, frame: np.ndarray) -> str:
        with tempfile.NamedTemporaryFile(prefix="wallx_fixed_", suffix=".png", delete=False) as handle:
            temp_path = Path(handle.name)

        if not cv2.imwrite(str(temp_path), frame):
            raise RuntimeError(f"Failed to save fixed frame to temporary path: {temp_path}")

        return str(temp_path)

    def _run_high_level(self, temp_img_path: str) -> tuple[str, list[tuple[str, tuple[int, int]]]]:
        api_key = os.getenv(PLANNER_API_KEY_ENV)
        if not api_key:
            raise RuntimeError(f"{PLANNER_API_KEY_ENV} is not set.")

        try:
            high_level = importlib.import_module("wall_point.high_level")
        except Exception as exc:
            raise RuntimeError("Failed to import wall_point.high_level.") from exc

        try:
            result_str, planning_list = high_level.process_inference(temp_img_path, api_key)
        except Exception as exc:
            raise RuntimeError("high_level.process_inference failed.") from exc

        if not planning_list:
            raise RuntimeError("Planner returned an empty planning_list.")

        return result_str, planning_list

    def _build_subtasks(self, planning_list: list[tuple[str, tuple[int, int]]]) -> list[WallXSubTask]:
        plan_id = self._plan_id or uuid.uuid4().hex
        subtasks: list[WallXSubTask] = []
        for subtask_id, (digit, target_xy) in enumerate(planning_list):
            instruction_en = f"grab the number {digit} block and place it at the red dot."
            subtasks.append(
                WallXSubTask(
                    plan_id=plan_id,
                    subtask_id=subtask_id,
                    digit=str(digit),
                    target_xy=(int(target_xy[0]), int(target_xy[1])),
                    instruction_en=instruction_en,
                    instruction_zh=f"抓取数字{digit}积木并放到红点处。",
                )
            )
        return subtasks

    def bootstrap_from_first_observation(self, raw_observation: RawObservation):
        if self._bootstrapped:
            return

        fixed_frame = self._extract_fixed_frame(raw_observation)
        temp_img_path = self._save_temp_frame(fixed_frame)

        try:
            result_str, planning_list = self._run_high_level(temp_img_path)
        finally:
            Path(temp_img_path).unlink(missing_ok=True)

        self._plan_id = uuid.uuid4().hex
        self._result_str = result_str
        self._subtasks = self._build_subtasks(planning_list)
        self._current_index = 0
        self._current_started_at = time.monotonic()
        self._bootstrapped = True

        self.logger.info("WallX fixed first-frame bootstrap succeeded.")
        self.logger.info("WallX planner result_str: %s", result_str)
        self.logger.info("WallX planner planning_list: %s", planning_list)

        current_subtask = self.current_subtask()
        if current_subtask is not None:
            self.logger.info(
                "WallX subtask start | subtask_id=%s digit=%s target_xy=%s instruction_en=%s",
                current_subtask.subtask_id,
                current_subtask.digit,
                current_subtask.target_xy,
                current_subtask.instruction_en,
            )

    def _draw_red_dot_on_fixed(self, frame: np.ndarray, target_xy: tuple[int, int]) -> None:
        cv2.circle(frame, target_xy, 3, (0, 0, 255), -1)
        cv2.circle(frame, target_xy, 4, (255, 255, 255), 1)

    def current_subtask(self) -> WallXSubTask | None:
        if self._finished or not self._subtasks:
            return None
        if self._current_index >= len(self._subtasks):
            return None
        return self._subtasks[self._current_index]

    def update_and_overlay(
        self, raw_observation: RawObservation
    ) -> tuple[RawObservation, WallXSubTask | None, bool]:
        if not self._bootstrapped:
            self.bootstrap_from_first_observation(raw_observation)

        subtask_changed = False
        current_subtask = self.current_subtask()
        if current_subtask is None:
            self._finished = True
            return raw_observation, None, False

        now = time.monotonic()
        if self._current_started_at is not None and now - self._current_started_at >= current_subtask.timeout_sec:
            self.logger.info(
                "WallX subtask end | subtask_id=%s done_reason=timeout",
                current_subtask.subtask_id,
            )
            self._current_index += 1
            self._current_started_at = now
            subtask_changed = True

            current_subtask = self.current_subtask()
            if current_subtask is None:
                self._finished = True
                self.logger.info("WallX all subtasks finished.")
                return raw_observation, None, True

            self.logger.info(
                "WallX subtask start | subtask_id=%s digit=%s target_xy=%s instruction_en=%s",
                current_subtask.subtask_id,
                current_subtask.digit,
                current_subtask.target_xy,
                current_subtask.instruction_en,
            )

        fixed_frame = self._extract_fixed_frame(raw_observation)
        self._draw_red_dot_on_fixed(fixed_frame, current_subtask.target_xy)

        if self._resolved_fixed_key is not None:
            raw_observation[self._resolved_fixed_key] = fixed_frame

        raw_observation["task"] = current_subtask.instruction_en
        raw_observation["__wallx_meta"] = {
            "plan_id": current_subtask.plan_id,
            "subtask_id": current_subtask.subtask_id,
            "digit": current_subtask.digit,
            "target_xy": [current_subtask.target_xy[0], current_subtask.target_xy[1]],
            "instruction_en": current_subtask.instruction_en,
            "timeout_sec": current_subtask.timeout_sec,
        }

        return raw_observation, current_subtask, subtask_changed

    def is_finished(self) -> bool:
        return self._finished
