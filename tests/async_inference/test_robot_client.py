# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit-tests for the `RobotClient` action-queue logic (pure Python, no gRPC).

We monkey-patch `lerobot.robots.utils.make_robot_from_config` so that
no real hardware is accessed. Only the queue-update mechanism is verified.
"""

from __future__ import annotations

import pickle
import time
from queue import Queue
from types import SimpleNamespace

import pytest
import torch

# Skip entire module if grpc is not available
pytest.importorskip("grpc")

# -----------------------------------------------------------------------------
# Test fixtures
# -----------------------------------------------------------------------------


@pytest.fixture()
def robot_client(tmp_path):
    """Fresh `RobotClient` instance for each test case (no threads started).
    Uses DummyRobot."""
    # Import only when the test actually runs (after decorator check)
    from lerobot.async_inference.configs import RobotClientConfig
    from lerobot.async_inference.robot_client import RobotClient
    from tests.mocks.mock_robot import MockRobotConfig

    test_config = MockRobotConfig()
    test_config.calibration_dir = tmp_path / "calibration"

    # gRPC channel is not actually used in tests, so using a dummy address
    test_config = RobotClientConfig(
        robot=test_config,
        server_address="localhost:9999",
        policy_type="test",
        pretrained_name_or_path="test",
        actions_per_chunk=20,
    )

    client = RobotClient(test_config)

    # Initialize attributes that are normally set in start() method
    client.chunks_received = 0
    client.available_actions_size = []

    yield client

    if client.robot.is_connected:
        client.stop()


# -----------------------------------------------------------------------------
# Helper utilities for tests
# -----------------------------------------------------------------------------


def _make_actions(start_ts: float, start_t: int, count: int):
    """Generate `count` consecutive TimedAction objects starting at timestep `start_t`."""
    from lerobot.async_inference.helpers import TimedAction

    fps = 30  # emulates most common frame-rate
    actions = []
    for i in range(count):
        timestep = start_t + i
        timestamp = start_ts + i * (1 / fps)
        action_tensor = torch.full((6,), timestep, dtype=torch.float32)
        actions.append(TimedAction(action=action_tensor, timestep=timestep, timestamp=timestamp))
    return actions


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------


def test_update_action_queue_discards_stale(robot_client):
    """`_update_action_queue` must drop actions with `timestep` <= `latest_action`."""

    # Pretend we already executed up to action #4
    robot_client.latest_action = 4

    # Incoming chunk contains timesteps 3..7 -> expect 5,6,7 kept.
    incoming = _make_actions(start_ts=time.time(), start_t=3, count=5)  # 3,4,5,6,7

    robot_client._aggregate_action_queues(incoming)

    # Extract timesteps from queue
    resulting_timesteps = [a.get_timestep() for a in robot_client.action_queue.queue]

    assert resulting_timesteps == [5, 6, 7]


@pytest.mark.parametrize(
    "weight_old, weight_new",
    [
        (1.0, 0.0),
        (0.0, 1.0),
        (0.5, 0.5),
        (0.2, 0.8),
        (0.8, 0.2),
        (0.1, 0.9),
        (0.9, 0.1),
    ],
)
def test_aggregate_action_queues_combines_actions_in_overlap(
    robot_client, weight_old: float, weight_new: float
):
    """`_aggregate_action_queues` must combine actions on overlapping timesteps according
    to the provided aggregate_fn, here tested with multiple coefficients."""
    from lerobot.async_inference.helpers import TimedAction

    robot_client.chunks_received = 0

    # Pretend we already executed up to action #4, and queue contains actions for timesteps 5..6
    robot_client.latest_action = 4
    current_actions = _make_actions(
        start_ts=time.time(), start_t=5, count=2
    )  # actions are [torch.ones(6), torch.ones(6), ...]
    current_actions = [
        TimedAction(action=10 * a.get_action(), timestep=a.get_timestep(), timestamp=a.get_timestamp())
        for a in current_actions
    ]

    for a in current_actions:
        robot_client.action_queue.put(a)

    # Incoming chunk contains timesteps 3..7 -> expect 5,6,7 kept.
    incoming = _make_actions(start_ts=time.time(), start_t=3, count=5)  # 3,4,5,6,7

    overlap_timesteps = [5, 6]  # properly tested in test_aggregate_action_queues_discards_stale
    nonoverlap_timesteps = [7]

    robot_client._aggregate_action_queues(
        incoming, aggregate_fn=lambda x1, x2: weight_old * x1 + weight_new * x2
    )

    queue_overlap_actions = []
    queue_non_overlap_actions = []
    for a in robot_client.action_queue.queue:
        if a.get_timestep() in overlap_timesteps:
            queue_overlap_actions.append(a)
        elif a.get_timestep() in nonoverlap_timesteps:
            queue_non_overlap_actions.append(a)

    queue_overlap_actions = sorted(queue_overlap_actions, key=lambda x: x.get_timestep())
    queue_non_overlap_actions = sorted(queue_non_overlap_actions, key=lambda x: x.get_timestep())

    assert torch.allclose(
        queue_overlap_actions[0].get_action(),
        weight_old * current_actions[0].get_action() + weight_new * incoming[-3].get_action(),
    )
    assert torch.allclose(
        queue_overlap_actions[1].get_action(),
        weight_old * current_actions[1].get_action() + weight_new * incoming[-2].get_action(),
    )
    assert torch.allclose(queue_non_overlap_actions[0].get_action(), incoming[-1].get_action())


@pytest.mark.parametrize(
    "chunk_size, queue_len, expected",
    [
        (20, 12, False),  # 12 / 20 = 0.6  > g=0.5 threshold, not ready to send
        (20, 8, True),  # 8  / 20 = 0.4 <= g=0.5, ready to send
        (10, 5, True),
        (10, 6, False),
    ],
)
def test_ready_to_send_observation(robot_client, chunk_size: int, queue_len: int, expected: bool):
    """Validate `_ready_to_send_observation` ratio logic for various sizes."""

    robot_client.action_chunk_size = chunk_size

    # Clear any existing actions then fill with `queue_len` dummy entries ----
    robot_client.action_queue = Queue()

    dummy_actions = _make_actions(start_ts=time.time(), start_t=0, count=queue_len)
    for act in dummy_actions:
        robot_client.action_queue.put(act)

    assert robot_client._ready_to_send_observation() is expected


@pytest.mark.parametrize(
    "g_threshold, expected",
    [
        # The condition is `queue_size / chunk_size <= g`.
        # Here, ratio = 6 / 10 = 0.6.
        (0.0, False),  # 0.6 <= 0.0 is False
        (0.1, False),
        (0.2, False),
        (0.3, False),
        (0.4, False),
        (0.5, False),
        (0.6, True),  # 0.6 <= 0.6 is True
        (0.7, True),
        (0.8, True),
        (0.9, True),
        (1.0, True),
    ],
)
def test_ready_to_send_observation_with_varying_threshold(robot_client, g_threshold: float, expected: bool):
    """Validate `_ready_to_send_observation` with fixed sizes and varying `g`."""
    # Fixed sizes for this test: ratio = 6 / 10 = 0.6
    chunk_size = 10
    queue_len = 6

    robot_client.action_chunk_size = chunk_size
    # This is the parameter we are testing
    robot_client._chunk_size_threshold = g_threshold

    # Fill queue with dummy actions
    robot_client.action_queue = Queue()
    dummy_actions = _make_actions(start_ts=time.time(), start_t=0, count=queue_len)
    for act in dummy_actions:
        robot_client.action_queue.put(act)

    assert robot_client._ready_to_send_observation() is expected


def test_receive_actions_drops_stale_subtask_payload(robot_client):
    """Stale action payloads should be dropped instead of reaching the queue."""
    from lerobot.async_inference.helpers import TimedAction
    from lerobot.transport import services_pb2

    action = TimedAction(timestamp=time.time(), timestep=5, action=torch.zeros(6))
    payload = {
        "subtask_id": 1,
        "digit": "8",
        "target_xy": [100, 200],
        "actions": [action],
    }

    class _Stub:
        def GetActions(self, _):
            robot_client.shutdown_event.set()
            return services_pb2.Actions(data=pickle.dumps(payload))

    robot_client.stub = _Stub()
    robot_client.start_barrier = SimpleNamespace(wait=lambda: None)
    robot_client.orchestrator = SimpleNamespace(current_subtask=lambda: SimpleNamespace(subtask_id=2))

    robot_client.receive_actions()

    assert robot_client.action_queue.empty()


def test_receive_actions_rejects_non_dict_payload(robot_client):
    """Malformed payloads should be rejected without mutating the queue."""
    from lerobot.transport import services_pb2

    bad_payload = [1, 2, 3]

    class _Stub:
        def GetActions(self, _):
            robot_client.shutdown_event.set()
            return services_pb2.Actions(data=pickle.dumps(bad_payload))

    robot_client.stub = _Stub()
    robot_client.start_barrier = SimpleNamespace(wait=lambda: None)
    robot_client.orchestrator = SimpleNamespace(current_subtask=lambda: SimpleNamespace(subtask_id=0))

    robot_client.receive_actions()

    assert robot_client.action_queue.empty()


def test_receive_actions_accepts_current_subtask_payload(robot_client):
    """Current-subtask payloads should still enter the aggregation path."""
    from lerobot.async_inference.helpers import TimedAction
    from lerobot.transport import services_pb2

    action = TimedAction(timestamp=time.time(), timestep=5, action=torch.zeros(6))
    payload = {
        "subtask_id": 2,
        "digit": "8",
        "target_xy": [100, 200],
        "actions": [action],
    }

    class _Stub:
        def GetActions(self, _):
            robot_client.shutdown_event.set()
            return services_pb2.Actions(data=pickle.dumps(payload))

    robot_client.stub = _Stub()
    robot_client.start_barrier = SimpleNamespace(wait=lambda: None)
    robot_client.orchestrator = SimpleNamespace(current_subtask=lambda: SimpleNamespace(subtask_id=2))

    robot_client.receive_actions()

    assert robot_client.action_queue.qsize() == 1


def test_validate_action_payload_extracts_digit_and_target_xy(robot_client):
    """Payload validation should extract task metadata alongside the actions list."""
    action = _make_actions(start_ts=time.time(), start_t=5, count=1)[0]
    payload = {
        "subtask_id": 2,
        "digit": "8",
        "target_xy": [100, 200],
        "actions": [action],
    }

    validated_payload = robot_client._validate_action_payload(payload)

    assert validated_payload == (2, "8", [100, 200], [action])


def test_control_loop_observation_clears_queue_on_subtask_transition(robot_client, monkeypatch):
    """A subtask switch should flush stale queued actions and force a must-go observation."""
    captured = {}
    robot_client.latest_action = 7
    robot_client.action_queue.put(_make_actions(start_ts=time.time(), start_t=8, count=1)[0])
    robot_client.must_go.clear()
    robot_client.robot.get_observation = lambda: {"joint1": 0.0}

    updated_observation = {"joint1": 0.0, "task": "planned"}
    robot_client.orchestrator = SimpleNamespace(
        update_and_overlay=lambda raw: (updated_observation, SimpleNamespace(subtask_id=1), True),
        is_finished=lambda: False,
    )

    monkeypatch.setattr(
        robot_client,
        "send_observation",
        lambda obs: captured.setdefault("observation", obs) or True,
    )

    returned_observation = robot_client.control_loop_observation(task="legacy-task")

    assert returned_observation is updated_observation
    assert robot_client.action_queue.empty()
    assert captured["observation"].must_go is True
    assert captured["observation"].get_observation() is updated_observation


def test_control_loop_observation_stops_when_orchestrator_finishes(robot_client, monkeypatch):
    """A finished orchestrator should stop the client loop."""
    captured = {}
    robot_client.robot.get_observation = lambda: {"joint1": 0.0}

    updated_observation = {"joint1": 0.0, "task": "planned"}
    robot_client.orchestrator = SimpleNamespace(
        update_and_overlay=lambda raw: (updated_observation, None, False),
        is_finished=lambda: True,
    )

    monkeypatch.setattr(
        robot_client,
        "send_observation",
        lambda obs: captured.setdefault("observation", obs) or True,
    )

    returned_observation = robot_client.control_loop_observation(task="legacy-task")

    assert returned_observation is updated_observation
    assert robot_client.shutdown_event.is_set() is True
    assert captured["observation"].get_observation() is updated_observation
