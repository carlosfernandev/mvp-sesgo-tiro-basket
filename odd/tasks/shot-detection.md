# Shot detection for multi-shot session videos

## Objective
Allow `analisis_tiro.py` to analyze a video containing SEVERAL shots (e.g. a
15 s session with 10 shots), not only a clip containing exactly one.

## Problem
`detectar_fases()` finds ONE global elbow-angle minimum and ONE global maximum
across the entire input. With a single shot that is correct. With N shots it
collapses all N into 4 phases spanning the whole clip.

Measured on a synthetic 15 s / 900-frame signal containing 10 shots:

```
preparacion   26 frames   0.4 s
elevacion     14 frames   0.2 s
liberacion    16 frames   0.3 s
seguimiento  844 frames  14.1 s   <-- shots 2..10 land here
```

Only the first shot is segmented. The remaining nine are absorbed into
`seguimiento`.

## Why now
The single-shot path works and is hand-validatable. Multi-shot is an
architectural gap, not a bug in the existing logic. The existing per-shot
phase logic is NOT thrown away: it becomes a subroutine applied once per
detected shot window.

## Scope
In scope:
- `detectar_tiros()`: find shot windows from the elbow-angle signal.
- Apply the existing `detectar_fases()` to each window independently.
- A multi-shot output path that does not alter the single-shot path.

Out of scope (deliberately deferred):
- Rewriting or retuning the existing phase-boundary logic.
- Automatic release-frame detection (MediaPipe does not track the ball).
- Any schema change to the existing single-shot CSV. Deferred: the user's
  original contract fixed the exact columns
  `frame,timestamp,ángulo_codo,ángulo_rodilla,ángulo_hombro,fase_asignada`.
  A `tiro_id` column would be the natural addition, but that is a contract
  change and needs the user's decision.

## Design constraints
- NO new third-party dependency. `scipy` is not installed and must stay that
  way for this MVP; peak detection is hand-rolled on numpy/pandas.
- The single-shot path must stay byte-identical in behavior.
- Limits are expressed in MILLIMECONDS, not frames, for the reason already
  documented in the file: a fixed frame count is silently wrong across fps
  (5 frames = 83 ms at 60 fps but 167 ms at 30 fps).

## Tasks
- [x] T1. `detectar_tiros()`: locate shot windows (dip + following snap peak).
- [x] T2. Filter candidate dips by prominence and minimum separation.
- [x] T3. Multi-shot path: run existing `detectar_fases()` per window.
- [x] T4. Tests for the detector and the multi-shot path.
- [ ] T5. Resolve the false positive: `MAX_SHOT_SPAN_MS = 3000.0` alarms on
      every valid session video because it was calibrated for a single shot.
      Deliberately NOT implemented: needs the user's decision.

## Acceptance criteria
- A 10-shot synthetic signal yields 10 windows, not 1.
- A single-shot signal still yields exactly 1 window.
- The existing 44 tests keep passing.
- The single-shot CSV keeps its exact original columns.

## Known honesty constraints
- **Nothing here is validated on real footage.** Every number below is a
  provisional guess until the user supplies annotated real shots.
- The detector will be fooled by a dribble deep enough to exceed the shot
  dip. Prominence and minimum-separation filters are mitigations, not proof.
- A 15 s session video has more occlusion and more NaN gaps than a 3 s clip,
  so per-frame pose quality is a bigger factor in practice than in the
  single-shot case.

## Verification
- `.venv/bin/python -m unittest test_analisis_tiro` — all pass.
- Synthetic 10-shot signal yields 10 windows.
- Single-shot signal yields 1 window.
- Single-shot CSV columns unchanged.

## Progress
T1-T4 implemented and independently verified by the orchestrator:
- 56 tests pass (44 pre-existing + 12 new).
- 10-shot / 15 s signal -> 10 windows. 3-shot -> 3. 1-shot -> 1.
- Single-shot CSV columns are byte-identical to the original contract; no
  `tiro_id` added.
- `VentanaTiro` is a frozen dataclass with a half-open iloc range
  (`inicio` inclusive, `fin` exclusive) plus `valle` and `pico` frames.
- `detectar_fases_por_tiro()` returns `(VentanaTiro, DataFrame)` pairs.
- Constants added: `MIN_DIP_PROMINENCE_DEG = 40.0`,
  `MIN_SHOT_SEPARATION_MS = 600.0` (both documented as provisional).

## Behavioral notes (verified, not bugs)
- Windows TILE the signal: `fin[i] == inicio[i+1]`, last window ends at `n`.
- Consequence: `preparacion` for shots 2..N absorbs the whole inter-shot dead
  time (measured 60 frames on the 10-shot signal). It is not a per-shot
  "set" phase; it means "everything between the previous release and this dip".
- Consequence: `seguimiento` is absent on every window except the last, because
  the window ends at the snap peak. Documented in the docstring.

## Known limitation (reported, not hidden)
A dribble that fully mimics a shot cycle (dip ~72 deg rising to ~168 deg) is
detected as a false shot. Prominence and separation cannot separate them: a
prominent dribble IS prominent. Mitigation needs a real signal the elbow angle
does not carry (e.g. ball tracking), or hand-verified session boundaries.

## Next step
Two decisions pending with the user, both contract-level rather than technical:
1. T5: make the span plausibility check shot-count-aware, or leave it.
2. Whether a multi-shot output needs a `tiro_id` column (a change to the
   user's originally fixed CSV schema).
