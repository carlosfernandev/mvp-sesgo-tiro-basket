# Shooter selection by elbow excursion

## Objective
When a video contains more than one person, automatically analyze the person
who is actually shooting instead of whichever person MediaPipe ranked first.

## Problem
`procesar_video()` hard-codes `num_poses=1` and appends only
`result.pose_landmarks[0]`. With two people in frame there is no notion of a
"target person": the code silently analyzes the most prominent pose, which may
be an assistant, a rebounder, or a bystander.

Verified: line 750 `frames_landmarks.append(list(result.pose_landmarks[0]))`.
Raising `num_poses` alone changes nothing, because index `[0]` is still
discarded, and it fails SILENTLY — no error, no behavior change.

## Approach
Select the shooter by **elbow-angle excursion**: the person who shoots has by
far the largest elbow-angle range. It is intrinsic to shooting, needs no ball
tracking, and does not break on a pass or a rebound the way "closest person to
the ball" does.

## Scope
In scope:
- Multi-pose capture and storage.
- Per-person elbow excursion scoring, with a detection-quality gate.
- Side selection performed WITHIN the selected person.
- Human-readable report of the election (every candidate and its score).

Out of scope:
- Ball tracking. Separate problem, separate cost.
- Any change to `detectar_fases()`, `detectar_tiros()`, or the CSV contract.
- Retro-fitting the side-selection fix beyond what multi-person requires.

## Design constraints
- The single-person path must behave EXACTLY as before.
- The election must be AUDITABLE: print every candidate's score and the
  reason it was accepted or rejected. This is the "confidence alongside the
  value" rule applied to a decision, after the visibility, side and global
  extrema lessons.
- Limits in degrees / milliseconds, never frames.

## Tasks
- [x] T1. Capture N poses per frame; store per-person frame series.
- [x] T2. Per-person elbow excursion, robust to single-frame glitches.
- [x] T3. Detection-quality gate so a garbage pose cannot win the election.
- [x] T4. Side selection restricted to the selected person.
- [x] T5. Report the election to stdout.
- [x] T6. Tests, including the garbage-wins case.
- [x] T7. Document the unvalidated-on-real-footage status.

## Constants chosen
- `MAX_DETECTED_POSES = 2` — shooter + one other; each extra pose costs
  inference on every frame.
- `MIN_SHOOTER_VALID_FRACTION = 0.5` — a person flickering in and out can show
  a huge random range from a few glitched frames. A share, so fps-independent.
- `MIN_SHOOTER_VISIBILITY = 0.5` — mirrors the pose-confidence bar already
  configured in the PoseLandmarker options; one consistent threshold.
- `MIN_SHOOTER_EXCURSION_DEG = 30.0` — just under `MIN_DIP_PROMINENCE_DEG` (40).
  Below it, no candidate cleared the bar and the result is an honest
  "no shot", never a least-bad winner.

## Acceptance criteria
- Two synthetic people, one shooting one standing -> the shooter wins.
- A garbage pose with a huge random angle range -> REJECTED, not selected.
- One person only -> identical behavior to today.
- All existing 56 tests keep passing.
- CSV columns unchanged.

## Known honesty constraints
- **No real two-person video exists yet.** Selection logic can be proven on
  synthetic landmark data; whether it survives real footage is UNKNOWN. The
  garbage gate is a mitigation against a failure mode we can demonstrate
  synthetically, not proof that real occlusion is handled.
- Pose ranking stability is assumed: MediaPipe may swap person indices between
  frames. Per-person series are assumed consistent. If it does not swap, the
  excursion of a single series is meaningless. This MUST be stated.

## Verification
- `.venv/bin/python -m unittest test_analisis_tiro` — all pass.
- Synthetic 2-person selection chooses the shooter.
- Synthetic garbage pose is rejected.
- Single-person path unchanged.

## Progress
Implemented and independently verified by the orchestrator:
- 71 tests pass (56 pre-existing + 15 new). The 9 shooter-election tests pass
  by name, including `test_garbage_pose_with_huge_range_is_rejected`,
  `test_shooter_wins_over_bystander`, `test_no_shot_reports_none_without_bogus_winner`
  and `test_single_person_behaves_like_current_path`.
- The hard-coded `result.pose_landmarks[0]` discard is gone; both duplicated
  `num_poses=1` option blocks were replaced by one `_pose_landmarker_options()`
  helper reading `MAX_DETECTED_POSES`.
- `_select_side` body is unchanged; its caller now passes only the elected
  person's series, so it can no longer mix visibility across people.
- The election prints every candidate with valid-frame count, mean visibility,
  excursion, verdict and reason, then names the winner. This is the
  auditability requirement: a human must be able to tell a correct election
  from a wrong one.

## OPEN RISK — person index stability (guard NOT implemented)
`PoseLandmarker` in VIDEO mode re-detects per frame and does not associate
identities across frames. With `num_poses=2` the ordering is per-frame
confidence, so indices can plausibly swap: person 0 in frame 1 and person 0 in
frame 200 may be different people. A swapped series mixes two people and its
excursion is meaningless.

A swap guard was deliberately NOT implemented. Rationale: with zero real
two-person footage, any threshold would be a guess that could silently reject
valid videos, trading one silent failure for another. The assumption is
instead stated in the function docstrings and in every printed election report.

Scope of the risk: it only exists when two or more people are in frame. With
a single person there is no ambiguity and the path is unchanged.

Cheapest resolution: run one real two-person clip and read the printed
election. If indices visibly swap, the guard becomes calibratable with
evidence instead of guesswork.

## Next step
Obtain a real two-person clip and inspect the election report. Until then the
election remains UNVALIDATED on real footage.
