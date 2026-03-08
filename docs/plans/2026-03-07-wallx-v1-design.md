# Wall-X V1 Async Deployment Design

**Date:** 2026-03-07

**Goal:** Extend the existing LeRobot async inference client/server pipeline so a Wall-X deployment can bootstrap a high-level math plan from the first `fixed` camera frame, execute one timed subtask at a time for 15 seconds, overlay only the `fixed` image with the training-style red dot, and stop the client cleanly when all subtasks are finished.

## Scope

- Preserve the current async client/server inference lifecycle.
- Keep the current `robot_client.py -> policy_server.py` data flow intact.
- Reuse `wall_point.high_level.process_inference(img_path, api_key)` as the planner entrypoint.
- Do not add multiprocess orchestration.
- Do not implement success detection, visual verification, or policy-driven `done`.
- Do not modify the handeye image stream.

## Confirmed Runtime Assumptions

- The robot camera configuration exposes two camera streams:
  - `handeye`
  - `fixed`
- The high-level planner must use only the first `fixed` frame.
- Planner target coordinates are in the original pixel coordinate system of that `fixed` input frame.
- The online overlay style must match training exactly:

```python
cv2.circle(frame, target, 3, (0, 0, 255), -1)
cv2.circle(frame, target, 4, (255, 255, 255), 1)
```

- The per-subtask English instruction is fixed as:

```text
grab the number {digit} block and place it at the red dot.
```

## Architecture

### 1. New helper: `src/lerobot/async_inference/wallx_task_runtime.py`

Add a single helper module to contain all new Wall-X runtime state.

It will define:

- `WallXSubTask`
  - `plan_id: str`
  - `subtask_id: int`
  - `digit: str`
  - `target_xy: tuple[int, int]`
  - `instruction_en: str`
  - `instruction_zh: str`
  - `timeout_sec: float = 15.0`
- `WallXTaskOrchestrator`
  - bootstrap from the first observation
  - extract the `fixed` frame
  - save a temporary planner image
  - read `SILICONFLOW_API_KEY`
  - call `wall_point.high_level.process_inference`
  - build ordered subtasks
  - track the active subtask with `time.monotonic()`
  - draw the red dot onto the original `fixed` frame every step
  - inject `raw_observation["task"]`
  - inject `raw_observation["__wallx_meta"]`
  - report when the subtask changed and when the full plan finished

The helper centralizes all new behavior so `robot_client.py` and `policy_server.py` only need small, local changes.

### 2. Client behavior

`src/lerobot/async_inference/robot_client.py` remains the owner of:

- robot observation capture
- gRPC observation send
- action queue management
- local action execution

The only behavioral additions are:

- create the orchestrator in `__init__`
- defer planner bootstrap until the first real observation
- pass each raw observation through `orchestrator.update_and_overlay(...)`
- on subtask switch:
  - clear `action_queue`
  - set `must_go`
- on orchestrator completion:
  - stop the client loop
- accept action payload dicts instead of bare `list[TimedAction]`
- discard stale action chunks whose `subtask_id` does not match the current subtask

### 3. Server behavior

`src/lerobot/async_inference/policy_server.py` keeps the current policy inference pipeline unchanged:

1. `raw_observation_to_observation(...)`
2. `preprocessor(...)`
3. `policy.predict_action_chunk(...)`
4. `postprocessor(...)`

The only enhancements are:

- log incoming `__wallx_meta` in `SendObservations()`
- copy the raw observation before conversion
- strip `__wallx_meta` before preprocessing so policy inputs stay clean
- package the result of `GetActions()` as:

```python
{
    "subtask_id": ...,
    "digit": ...,
    "target_xy": ...,
    "actions": action_chunk,
}
```

## Data Flow

### Bootstrap

1. Client connects and sends policy instructions exactly as today.
2. Client captures the first real observation.
3. Orchestrator extracts `observation.images.fixed` from that raw observation.
4. Orchestrator saves the first `fixed` frame to a temporary image.
5. Orchestrator reads `SILICONFLOW_API_KEY`.
6. Orchestrator calls `wall_point.high_level.process_inference(temp_img_path, api_key)`.
7. The returned `planning_list` becomes an ordered list of `WallXSubTask`.
8. The first subtask starts immediately and records `start_monotonic`.

### Per frame

1. Client calls `self.robot.get_observation()`.
2. Orchestrator updates the active subtask state.
3. Orchestrator overlays the red dot onto the original `fixed` frame only.
4. Orchestrator writes:

```python
raw_observation["task"] = "grab the number {digit} block and place it at the red dot."
raw_observation["__wallx_meta"] = {
    "plan_id": ...,
    "subtask_id": ...,
    "digit": ...,
    "target_xy": [x, y],
    "instruction_en": ...,
    "timeout_sec": 15.0,
}
```

5. Client constructs `TimedObservation` and sends it.
6. Server strips `__wallx_meta`, runs inference, then returns the wrapped action payload.
7. Client checks the payload `subtask_id`:
  - match: aggregate into the queue
  - mismatch: drop and log warning

### Subtask completion

- Completion is driven only by `time.monotonic()`.
- When `elapsed >= 15.0`:
  - current subtask ends with `done_reason=timeout`
  - orchestrator advances to the next subtask
  - client clears any queued actions
  - client sets `must_go`
- When no subtasks remain:
  - orchestrator reports finished
  - client stops the main loop

## Error Handling

These conditions should stop the client instead of continuing in a degraded state:

- missing `fixed` frame during bootstrap
- missing `SILICONFLOW_API_KEY`
- planner exception from `process_inference(...)`
- empty `planning_list`
- missing `fixed` frame in later observations

These conditions should drop the current server response and continue:

- malformed action payload
- payload missing expected keys
- payload `subtask_id` not equal to the active subtask id

## Logging

Required runtime logs:

- fixed first-frame bootstrap success
- planner `result_str`
- planner `planning_list`
- current subtask start:
  - `subtask_id`
  - `digit`
  - `target_xy`
  - `instruction_en`
- current subtask end with `done_reason=timeout`
- action payload `subtask_id`
- stale action chunk discard warning
- all subtasks finished

## File Plan

### New file

- `src/lerobot/async_inference/wallx_task_runtime.py`

### Modified files

- `src/lerobot/async_inference/robot_client.py`
- `src/lerobot/async_inference/policy_server.py`

### Optional minimal adjustment

- `wall_point/high_level.py`
  - avoid relying on its hardcoded `__main__` API key path during deployment
  - keep `process_inference(...)` as the reusable API

### Tests

- `tests/async_inference/test_robot_client.py`
- `tests/async_inference/test_policy_server.py`
- optionally add a focused helper test if needed

## Validation Plan

- unit-test stale subtask action payload rejection
- unit-test payload wrapping in `GetActions()`
- unit-test `__wallx_meta` stripping before preprocessing
- unit-test orchestrator bootstrap and timeout advancement
- run the focused async inference tests

## References

- Git official docs for worktree workflow: `git worktree` manual on git-scm.com
- Community guidance for repository-local worktree hygiene: technical community discussions around ignoring `.worktrees/` to avoid dirty status pollution

## Open Points

- The current environment skips the existing async inference test modules, so they are useful as smoke checks but not yet as behavioral proof.
- If `fixed` drops even one runtime frame, the implementation will stop immediately rather than fallback, matching the stricter failure policy requested for v1.
