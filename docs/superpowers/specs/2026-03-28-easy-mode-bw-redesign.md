# Easy Mode Redesign: B&W Single-Channel + Position Randomization

## Summary

Redesign easy mode to (1) remove fixed dancer positions, (2) unify dancer shapes to a single shape, (3) replace red-colored RGB rendering with black-and-white single-channel rendering directly in the pipeline (no wrapper).

## Motivation

- Position fixing is unnecessary — the agent should learn from dance motions, not positional cues.
- Dancer shapes are semantically meaningless — unifying simplifies the observation.
- Single-channel B&W reduces observation size (99x99x1 vs 99x99x3) and avoids `tensordot` overhead.
- Wrapper-based grayscale conversion is undesirable due to throughput cost (renders RGB first, then converts).

## Design

### Easy Mode Behavior Changes

| Property | Current Easy Mode | New Easy Mode |
|----------|-------------------|---------------|
| Dancer colors | All red | N/A (B&W) |
| Dancer shapes | First N shapes | Single shape repeated |
| Dancer positions | Fixed (first N) | Random shuffle (same as normal) |
| Observation shape | (99, 99, 3) | (99, 99, 1) |
| Rendering | RGB via tensordot | Direct B&W mask (0=black bg, 255=white shape) |

Normal mode remains completely unchanged: RGB (99, 99, 3).

### Implementation Approach: Pipeline-Internal Branching

Branch inside the rendering pipeline based on `self._easy_mode` rather than using a wrapper. This avoids the overhead of generating RGB and then converting.

### Changes (all in `ballet_environment.py`)

#### 1. `__init__`
- Store `self._easy_mode` flag (parsed from level_name).
- Set observation space shape to `(99, 99, 1)` when easy mode, `(99, 99, 3)` otherwise.

#### 2. `_generate_template()`
- Normal mode: existing logic — `tensordot(mask, color_rgb, 0)` producing (9, 9, 3).
- Easy mode: skip tensordot, return `(mask * 255).astype(uint8).reshape(9, 9, 1)` producing (9, 9, 1).

#### 3. `_reset()`
- Easy mode shape assignment: `shapes = [DANCER_SHAPES[0]] * self._num_dancers`.
- Remove position fixing — let positions go through normal random shuffle logic.
- Remove color assignment — colors are irrelevant in B&W.
- Generate single-channel versions of `_CHAR_TO_TEMPLATE_BASE` (agent, wall, floor) for easy mode.

#### 4. `_get_obs()`
- Normal mode: `np.zeros((99, 99, 3), dtype=np.uint8)` — unchanged.
- Easy mode: `np.zeros((99, 99, 1), dtype=np.uint8)` — place (9, 9, 1) templates.

#### 5. `render()`
- Easy mode: expand (99, 99, 1) to (99, 99, 3) via `np.repeat(..., 3, axis=2)` for visualization only.

### Files Modified
- `gym_balletenv/envs/ballet_environment.py` — sole file modified.

### Files NOT Modified
- `ballet_environment_core.py` — pycolab game logic unchanged.
- Wrappers — not involved.
- `__init__.py` — v0/v1 registration unchanged.

### Testing
- Update `ballet_environment_test.py` to verify easy mode observation shape is (99, 99, 1).
- Verify normal mode observation shape remains (99, 99, 3).

## Performance Characteristics

Easy mode B&W rendering is strictly faster than RGB:
- No `tensordot` call in template generation.
- 1/3 memory for observation array allocation.
- No wrapper overhead.
