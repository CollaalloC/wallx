from __future__ import annotations

import logging
from types import SimpleNamespace

import numpy as np
import pytest


def _make_observation() -> dict[str, np.ndarray]:
    return {
        "observation.images.fixed": np.zeros((32, 32, 3), dtype=np.uint8),
        "observation.images.handeye": np.full((32, 32, 3), 7, dtype=np.uint8),
    }


def test_run_high_level_requires_api_key(monkeypatch):
    from lerobot.async_inference.wallx_task_runtime import WallXTaskOrchestrator

    monkeypatch.delenv("SILICONFLOW_API_KEY", raising=False)
    orchestrator = WallXTaskOrchestrator(logger=logging.getLogger("wallx_test"))

    with pytest.raises(RuntimeError, match="SILICONFLOW_API_KEY"):
        orchestrator._run_high_level("/tmp/fixed.png")


def test_run_high_level_rejects_empty_planning_list(monkeypatch):
    from lerobot.async_inference.wallx_task_runtime import WallXTaskOrchestrator

    monkeypatch.setenv("SILICONFLOW_API_KEY", "test-key")
    orchestrator = WallXTaskOrchestrator(logger=logging.getLogger("wallx_test"))

    dummy_module = SimpleNamespace(process_inference=lambda _path, _key: ("18", []))
    monkeypatch.setattr(
        "lerobot.async_inference.wallx_task_runtime.importlib.import_module",
        lambda _name: dummy_module,
    )

    with pytest.raises(RuntimeError, match="planning_list"):
        orchestrator._run_high_level("/tmp/fixed.png")


def test_update_and_overlay_bootstraps_and_switches_subtasks(monkeypatch):
    from lerobot.async_inference.wallx_task_runtime import WallXTaskOrchestrator

    monotonic_values = iter([100.0, 100.0, 116.0])
    monkeypatch.setattr(
        "lerobot.async_inference.wallx_task_runtime.time.monotonic",
        lambda: next(monotonic_values),
    )

    orchestrator = WallXTaskOrchestrator(logger=logging.getLogger("wallx_test"))
    monkeypatch.setattr(orchestrator, "_save_temp_frame", lambda _frame: "/tmp/fixed.png")
    monkeypatch.setattr(
        orchestrator,
        "_run_high_level",
        lambda _path: ("18", [("1", (10, 12)), ("8", (20, 12))]),
    )

    first_observation = _make_observation()
    updated_observation, current_subtask, subtask_changed = orchestrator.update_and_overlay(first_observation)

    assert subtask_changed is False
    assert current_subtask is not None
    assert current_subtask.subtask_id == 0
    assert (
        updated_observation["task"]
        == "grab the number 1 block and place it at the red dot."
    )
    assert updated_observation["__wallx_meta"] == {
        "plan_id": current_subtask.plan_id,
        "subtask_id": 0,
        "digit": "1",
        "target_xy": [10, 12],
        "instruction_en": "grab the number 1 block and place it at the red dot.",
        "timeout_sec": 15.0,
    }
    assert tuple(updated_observation["observation.images.fixed"][12, 10]) == (0, 0, 255)
    assert np.array_equal(
        updated_observation["observation.images.handeye"],
        np.full((32, 32, 3), 7, dtype=np.uint8),
    )

    next_observation = _make_observation()
    updated_observation, current_subtask, subtask_changed = orchestrator.update_and_overlay(next_observation)

    assert subtask_changed is True
    assert current_subtask is not None
    assert current_subtask.subtask_id == 1
    assert updated_observation["__wallx_meta"]["digit"] == "8"
    assert tuple(updated_observation["observation.images.fixed"][12, 20]) == (0, 0, 255)
