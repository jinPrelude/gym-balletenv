# Easy Mode Redesign: B&W Single-Channel + Position Randomization

## Summary

Redesign easy mode to (1) remove fixed dancer positions, (2) unify dancer shapes to a single shape ("triangle"), (3) replace red-colored RGB rendering with black-and-white single-channel rendering directly in the pipeline (no wrapper).

## Motivation

- Position fixing is unnecessary — the agent should learn from dance motions, not positional cues.
- Dancer shapes are semantically meaningless — unifying simplifies the observation.
- Single-channel B&W reduces observation size (99x99x1 vs 99x99x3) and avoids `tensordot` overhead.
- Wrapper-based grayscale conversion is undesirable due to throughput cost (renders RGB first, then converts).

## Design

### Easy Mode Behavior Changes

| Property | Current Easy Mode | New Easy Mode |
|----------|-------------------|---------------|
| Dancer colors | All red | Dummy value passed to core (ignored in rendering) |
| Dancer shapes | First N shapes | "triangle" repeated N times |
| Dancer positions | Fixed (first N) | Random shuffle from full DANCER_POSITIONS pool (same as normal) |
| Observation shape | Tuple(Box(99,99,3), Discrete(14)) | Tuple(Box(99,99,1), Discrete(14)) |
| Rendering | RGB via tensordot | Direct B&W mask (0=black bg, 255=white shape) |

Normal mode remains completely unchanged: Tuple(Box(99,99,3), Discrete(14)).

### Implementation Approach: Pipeline-Internal Branching

Branch inside the rendering pipeline based on `self._easy_mode` rather than using a wrapper. This avoids the overhead of generating RGB and then converting.

### Changes (all in `ballet_environment.py`)

#### 1. `__init__`
- Store `self._easy_mode` flag (parsed from level_name).
- Set the `Box` component of the `Tuple` observation space to shape `(99, 99, 1)` when easy mode, `(99, 99, 3)` otherwise. The `Discrete(14)` language component is unchanged.

#### 2. `_generate_template()`
- Normal mode: existing logic — `tensordot(mask, color_rgb, 0)` producing (9, 9, 3).
- Easy mode: skip tensordot, return `(mask * 255).astype(uint8).reshape(9, 9, 1)` producing (9, 9, 1).
- The function needs to accept context about whether it's in easy mode. Options: pass `self._easy_mode` or make it a method. Since it's currently a module-level function, add an `easy_mode=False` parameter.

#### 3. `_reset()`
- Easy mode shape assignment: `shapes = ["triangle"] * self._num_dancers` (DANCER_SHAPES[0] is "triangle").
- Remove position fixing — use the full `DANCER_POSITIONS` list, shuffle all 8, take first N. This matches normal mode behavior exactly.
- Color: still pass a dummy color string (e.g. `"red"`) to `ballet_core.make_game()` since the core expects it for its `char_to_color_shape` mapping. The color will be ignored during B&W template generation.
- Create local single-channel copies of `_CHAR_TO_TEMPLATE_BASE` templates (agent, wall, floor). Do NOT mutate the module-level constant. Convert each (9,9,3) template to (9,9,1) by taking any single channel and keeping the 255/0 structure.

#### 4. `_get_obs()`
- Normal mode: `np.zeros((99, 99, 3), dtype=np.uint8)` — unchanged.
- Easy mode: `np.zeros((99, 99, 1), dtype=np.uint8)` — place (9, 9, 1) templates.

#### 5. `render()`
- Easy mode: before composing the 99x200 canvas, expand `self.curr_img_obs` from (99,99,1) to (99,99,3) via `np.repeat(..., 3, axis=2)`. Then the existing canvas allocation `np.zeros((99, 200, 3))` and slice assignment `canvas[:, :99, :] = img` work without modification.

### Files Modified
- `gym_balletenv/envs/ballet_environment.py` — sole file modified.

### Files NOT Modified
- `ballet_environment_core.py` — pycolab game logic unchanged.
- Wrappers — not involved.
- `__init__.py` — v0/v1 registration unchanged (no observation shape embedded in registration).

### Testing
- Verify easy mode observation Box shape is (99, 99, 1).
- Verify normal mode observation Box shape remains (99, 99, 3).
- Verify easy mode dancers all have the same visual template (unified shape).
- Verify easy mode dancer positions are randomized across multiple resets with different seeds.
- Verify `render()` returns a valid (99, 200, 3) RGB array in easy mode.
- Verify easy mode observation pixel values are only 0 or 255 (pure B&W).
- Verify all existing normal mode tests still pass (regression).

## Performance Characteristics

Easy mode B&W rendering is strictly faster than RGB:
- No `tensordot` call in template generation.
- 1/3 memory for observation array allocation.
- No wrapper overhead.
