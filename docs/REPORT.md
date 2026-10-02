# Report: basketball player identity prototype (hoopid 0.1)

Date: 2026-10-02. Everything called "measured" below was obtained on the attached video
`data/raw/VIDEO-2026-10-02-14-05-11.mp4` (SHA-256 `1a8750aa…7a6c`) against the manual ground
truth in `gt/`. Anything not measured is marked as an **assumption**.
Russian original: `docs/ru/REPORT.ru.md`.

## 0. Summary

| | Baseline<br>(detector + tracker + nearest Re-ID match) | Persistent-identity system<br>(automatic, 2 passes) | System + 17 manual corrections |
|---|---|---|---|
| Wrong identity assignments (whole video, detections) | **3915** of 7953 | **0** | **0** |
| — of which on unregistered people | 2853 | 0 | 0 |
| Precision of confirmed IDs | 50.8% | 100% | 100% |
| Coverage: share of registered players' visible time with the correct confirmed ID | 79.2% | **90.9%** | **95.1%** |
| Same, test half only | 63.7% | 89.7% | 93.8% |
| Online pass (no look-ahead), test half | — | 85.3%, 0 wrong | — |
| player_id switches for the same person | 25 | 0 | 0 |
| IDF1 (by player_id, false IDs on other people included) | 0.619 | 0.953 | 0.975 |
| Recovery after disappearing ≥0.5 s (correct ID within 3 s) | 33/41 (but with 2308 errors on the test half) | 35/41 | — |

Main finding: on this clip the system never assigned a wrong identity. The price: 9% of
registered players' visible time remains UNKNOWN. The 95% target **is not reached** without
manual correction; with correction it is, but the correction density (17 operations per 55 s)
**does not fit** the "10 minutes per hour of footage" target (see §8). The video is short (55 s),
so independent validation is **insufficient**: the test half is 27 s of the same gym, the same
people and the same lighting.

## 1. Environment (actual)

The work was **not done on the target machine**; the RTX 5080 was not available. Cloud container:

* CPU: Intel Xeon @ 2.10 GHz, 4 vCPU, no GPU (no NVIDIA driver or CUDA);
* RAM: 15 GB; OS: Linux 6.18; Python 3.11.15;
* onnxruntime 1.30.0 (CPUExecutionProvider), OpenCV 5.0.0, NumPy 2.4.6, SciPy 1.17.1, Flask 3.1.3, system ffmpeg.

So PyTorch/CUDA compatibility with the RTX 5080 (Blackwell, sm_120) was **not checked** here.
The prototype deliberately does not depend on PyTorch: all models run through ONNX Runtime.
On the RTX 5080, installing `onnxruntime-gpu` with CUDA 12.8+ should be enough (**assumption**:
sm_120 support must be verified on site with
`python -c "import onnxruntime as o; print(o.get_available_providers())"`).
None of the speed figures below represent RTX 5080 performance.

## 2. Video audit

| Parameter | Value (measured) |
|---|---|
| Container / codec | MP4, H.264 Baseline, yuv420p; AAC audio |
| Coded resolution | 848×480, −90° rotation matrix → **displayed 480×848 (portrait)** |
| Duration / frames | 54.625 s / 1638 frames |
| FPS | **variable**: 29.99 on average; PTS step 31.7–63.3 ms; 87 intervals > 40 ms (dropped frames). Exact per-frame PTS are used (`frame_timestamps`) |
| Camera | handheld phone, almost static: total drift ≈ 14 px in X and 42 px in Y over the clip, no zoom (scale 0.999–1.001), ORB inliers per frame: median 1066, minimum 638 |
| Cuts | none (minimum histogram correlation between neighbouring frames 0.992) |
| Person size | box height: median 85 px, 10th percentile 42 px, 90th 153 px, maximum 469 px (child next to the camera) |

Near and far examples: `docs/img/annotated_system.jpg`; per-track crops: `gt/units.json` and the labelling sheets.

**Readability of identity cues:**

* **Numbers:** no uniforms, this is pickup play in casual clothes. OCR (RapidOCR) on the 400 largest crops
  of the participants read **0 digits**. Only fragments of the "ROAD TO NATIONALS" print on one
  child's T-shirt were recognised (`results/ocr.json`). The team/number cue is implemented but not applicable to this video.
* **Shoes:** the shoe strip with floor colour removed is available in 45% of detections. But most players
  wear similar bright pink sneakers, so the cue is weak (the correct player ranks first by it in only 49% of cases).
* **Hair:** a head of ≥ ~10 px only in boxes ≥ 140 px, i.e. 13% of detections.
* **Tattoos:** with a forearm 5–10 px wide they are not legible; the cue is **disabled**. Absence of a
  visible tattoo is not treated as evidence of its absence.
* **Skin tone:** **not extracted** at all. In these frames it is dominated by lighting (back-lit windows,
  floor glare); the risk of it acting as a demographic proxy outweighs the benefit, and the brief allows it only as a weak auxiliary cue.
* Super-resolution and generative restoration were **not used**.

**Hard episodes (from the ground truth):**

* up to 10–12 people in frame at once, 9 of them regular participants on the near court; the background has people on neighbouring courts and by the windows;
* 424 track segments, many of them "contact" segments (boxes overlap with IoU ≥ 0.1);
* 41 cases of "disappeared for ≥ 0.5 s and returned" for registered players (19 in tuning, 17 in test);
* the tracker silently jumps from one person to another (per ground truth at least in tracks 1, 5, 24, 58, 62, 63); example: track 24, frames 408→413, `docs/img/tracker_jump_t24.jpg`;
* two children in dark T-shirts and two more in grey ones look very similar; during the first labelling pass I wrongly split one adult in a black logo T-shirt into two people (see §7).

**Recording limitations:** low resolution (480 px wide), portrait frame, motion blur, strong floor
glare and back-light from windows, identical clothing and shoe colours, no uniforms or numbers,
55 s of footage, one court, one day.

## 3. What was implemented

Code: the `hoopid/` package. Commands: `README.md`.

| Module | File | What it does |
|---|---|---|
| Video and PTS reading | `video.py` | streaming frame-by-frame reading, exact PTS (VFR), rotation from metadata |
| Detector | `detect.py` | YOLOX-m (COCO person), ONNX, threshold 0.25, NMS 0.5, optional tiling |
| Camera | `camera.py` | ORB + RANSAC similarity between frames (people masked out), unreliability flag, cut detector |
| Features | `features.py` | Re-ID (YoutuReID 768-d), HSV histograms of torso/shorts/shoes (floor pixels removed)/hair, quality: size, truncation, occlusion, sharpness |
| Precompute | `precompute.py` | one streaming pass, shards of 600 frames on disk (never holds a match in RAM) |
| MOT tracker | `tracker.py` | own ByteTrack-style implementation, Kalman filter, camera-motion compensation, LOST state with prediction (flagged as a prediction), reset on cuts |
| Identity manager | `identity.py` | segments, joint assignment (Hungarian algorithm), best-vs-second margin check, exclusivity, second pass, reason log |
| Player registry | `registry.py`, `register.py` | `players.json`: team, number, confirmed samples with quality and view, source, history |
| Numbers | `ocr.py` | RapidOCR on the torso; a number is confirmed only with ≥3 agreeing frames at confidence ≥0.8 |
| Manual correction | `corrections.py`, `ui.py` | assign / unknown / split / merge / set_attr, re-run (≈2.6 s) |
| Interface | `ui.py` | local Flask app: pick a frame and click a player, registration, view and delete samples, review segments, corrections, operator time logging |
| Export | `export.py` | `tracks.csv`, `assignments.jsonl`, annotated video |
| Evaluation | `evaluate.py`, `experiments.py`, `calibrate.py`, `correction_sim.py` | metrics against ground truth, thresholds tuned on the tuning half only, ablations |
| Coach stats | `ball.py`, `stats.py`, `dashboard.py` | ball detection, time with ball, distance/speed estimates, dashboard video (§10) |

Entity separation: `det_id` (a box in a frame) → `track_id` (short tracker trajectory) →
`segment` (part of a track between risk events) → `player_id` (persistent identity).
One `player_id` is linked to many tracks and segments.

### Persistent identity rules (as implemented)

* Identity is decided **per segment**. A track is cut: (a) on box contact (IoU ≥ 0.1);
  (b) after a gap > 5 frames; (c) on a sharp appearance change: Re-ID similarity of 6-observation windows < 0.55,
  or < 0.8 together with a change of the best player with a margin in both windows; (d) on a manual split.
  So a tracker jump during a crossing cannot "drag" a confirmed ID onto another person.
* **Joint assignment:** on every frame, the Hungarian algorithm over all visible unconfirmed segments and
  players, with dummy "nobody" columns at the threshold level.
* Confirmation requires all of: score ≥ `abs_thr`, accumulated quality ≥ `n_min`,
  **margin over the second player ≥ `margin`** (+ `contact_extra_margin` for contact segments), and no other visible segment
  equally similar to that player (`col_margin`).
* A player confirmed on a live segment is never given to a second visible segment.
* When in doubt the status is UNKNOWN (UNCERTAIN) with the three best candidates and their scores.
  If nobody registered is similar, the status is UNREGISTERED.
* LOST: while the tracker keeps a track without observations (≤ 30 frames), `tracks.csv` gets a row
  `observed=0, predicted=1, status=LOST`; in the video it is a dashed red box. A prediction is never counted as an observation.
* Camera: when the motion estimate is unreliable the prediction is frozen; after a cut all tracks are closed.
  Re-entry priors (`reentry_*`) are disabled if the camera was unreliable between disappearance and return.
* Uncertain observations are **never added** to a player's memory: the gallery consists only of user-confirmed samples.
* Second pass (recorded video): decision over the whole segment, including later frames, plus a global check
  that one player is never on two time-overlapping segments. The online decision is kept in `tracks.csv`
  (`player_id_online`, `status_online`), every change goes to `assignments.jsonl` (`pass: offline, event: revised`).
* The score is a `score` (logit of an uncalibrated logistic model), **not a probability**.
* The online pass uses a 6-frame buffer (≈0.2 s) to detect appearance jumps, so it is
  "near-online" with ~0.2 s latency.

## 4. Models and licences

Details: `docs/MODELS_AND_LICENSES.md`, full inventory: `docs/RESOURCES.md`. In short: YOLOX-m
(Apache-2.0, Megvii, weights v0.1.1rc0), YoutuReID ONNX (Apache-2.0, OpenCV Zoo), RapidOCR (Apache-2.0).
The tracker and identity manager were written from scratch to avoid AGPL code (Ultralytics, BoxMOT).
Risk: the Re-ID and COCO weights were trained on datasets with their own terms (Market-1501/MSMT17 are
research-only); for commercial use a lawyer must check this.

## 5. Evaluation method

* **Ground truth:** 456 track pieces (each up to 1.5 s, no contact inside) were labelled by hand → 8514
  detections with labels: 9 participants, `OTHER` (2853 detections of other people), `AMBIG` (561 detections excluded:
  two people in the box or impossible to judge). 842 detections (9%) were not labelled: 477 belonged to no track and 365 lie in pieces shorter than 4 frames.
  Method and change log: `gt/README.md`.
* **Split:** tuning = frames 0–819 (27.3 s), test = 820–1637 (27.3 s). Registration (4 samples
  for each of 9 players), fitting the score model and tuning thresholds happened **on the tuning half only**. The test
  half was computed with frozen parameters.
* Metrics are computed only on detections labelled in the ground truth. Ground-truth boxes are the same detector's boxes,
  so the metrics measure identification **given detection**. Players missed by the detector are not counted.
  That is why HOTA is not reported: localisation matches the ground truth by construction and HOTA would be inflated.
  IDF1 is reported.
* "Coverage" = share of registered players' detections where the system showed the correct `player_id` with status CONFIRMED.
  At ~30 fps, detections correspond to visible time.

## 6. Results

### 6.1 Baseline vs system

| Variant | Split | Correct | Wrong | Precision | Coverage | UNKNOWN | ID sw | IDF1 | Recov. ≤3 s |
|---|---|---|---|---|---|---|---|---|---|
| Baseline | tuning | 2624 | 1607 | 62.0% | 91.1% | 0 | 13 | 0.738 | 19/19 |
| Baseline | **test** | 1414 | **2308** | 38.0% | 63.7% | 0 | 11 | 0.497 | 10/17 |
| System (2 passes) | tuning | 2647 | 0 | 100% | 91.9% | 8.1% | 0 | 0.958 | 15/19 |
| System (2 passes) | **test** | 1991 | **0** | 100% | **89.7%** | 10.3% | 0 | 0.946 | 15/17 |
| System, online only | test | 1894 | 0 | 100% | 85.3% | 14.7% | 0 | 0.921 | 16/17 |

The baseline gives every track an ID without ever refusing, so it also labels bystanders
(2853 errors on people on neighbouring courts and by the windows) and never fixes tracker jumps.

### 6.2 Feature contribution (parameters tuned separately for each set, tuning half only)

| Feature set | Tuning: coverage / wrong | Test: coverage / wrong | Test online: coverage / wrong |
|---|---|---|---|
| clothing colour only (no Re-ID) | 86.0% / **26** (constraint not met) | 75.0% / 0 | 70.5% / 0 |
| **Re-ID** (chosen) | 91.9% / 0 | 89.7% / 0 | 85.3% / 0 |
| Re-ID + torso and shorts colour | 91.9% / 0 | 82.9% / 0 | 78.6% / 0 |
| + shoes | 91.9% / 0 | 82.2% / 0 | 74.9% / 0 |
| + hair | 91.9% / 0 | 85.9% / 0 | 80.7% / 0 |

On the tuning half the Re-ID sets are **indistinguishable**. Following the rule "an indistinguishable cue is not enabled by default",
the simplest set was chosen: Re-ID only. The test half confirmed the choice: the extra colour cues
lowered coverage by 4–7 points. Clothing colour alone, without Re-ID, already makes errors on the tuning half.

### 6.3 Component ablations (Re-ID, same parameters)

| Disabled | Tuning: cov. / wrong | Test: cov. / wrong | Test online: wrong | Conclusion |
|---|---|---|---|---|
| appearance-change split | 91.3% / **16** | 89.8% / 0 | 2 | needed: catches silent tracker jumps |
| contact segmentation | 97.2% / 0 | 96.5% / 0 | **12** | +5–7 points of coverage, but 12 wrong online on test. Kept as a safeguard |
| motion priors (continuity/reentry) | no change | no change | — | indistinguishable → off by default |
| tracker appearance gate | no change | no change | — | indistinguishable → off by default |

### 6.4 Speed and memory (CPU machine from §1, not an RTX 5080)

| Stage | Time for 54.6 s of video | Share |
|---|---|---|
| YOLOX-m detector 640×640 | 367.6 s | 59% |
| Features (Re-ID on every box + histograms) | 215.2 s | 35% |
| Camera motion (ORB) | 36.2 s | 6% |
| **Precompute total** | **623.2 s → 2.63 fps, 0.088× real time** | |
| Tracking + identity + export (`pipeline`) | 1.5–2.6 s | |
| Peak RSS | 730 MB during precompute, 276 MB during identity | |
| VRAM | **not measured** (no GPU) | |

Real time on an RTX 5080 is **not promised** and was not measured. FP16 and TensorRT have not been enabled yet; that is the next step after a check on the target machine.

## 7. Remaining errors and their causes

1. **UNKNOWN on 9% of visible time** (462 detections): "contact unresolved" 300; too few good observations 121
   (often short, small segments: median height 85 px vs 105 px overall); "below threshold" 35; "ambiguous between two players" 6.
   Examples: `docs/img/unknown_examples.jpg`.
2. **Not recovered within 3 s:** 6 of 41 returns; 4 of them are one adult in a black T-shirt. He is mostly
   seen small and in contact with others, and his registration comes from a single episode (frames 28–103).
3. **Tracker jumps inside a track.** The appearance-change split fired 10 times. The system catches jumps with that split
   (`docs/img/tracker_jump_t24.jpg`: P01 → P06 at frame 413 is reassigned correctly). But with similar clothing
   the split may not fire. This is the main **potential** source of wrong assignments on longer recordings.
4. **An error in my ground truth (fixed, disclosed):** during the first labelling pass, from small crops, I split one
   adult in a black logo T-shirt and beige trousers into two people. With ground truth v1 the test half showed
   13 "wrong" detections for the system (PLAYER_09 on the "other" person). Checking large crops (`docs/img/gt_fix_same_person.jpg`)
   showed the same dreadlocks, T-shirt, trousers and shoes in all those appearances. It is one person, and the system was right.
   The change only affected the test half and did not influence parameter selection.
   Results against ground truth v1 are kept in `results/experiments_gt_v1_before_gt_fix.json`. Lesson: ground truth from 40–80 px crops
   makes mistakes itself; hard pairs need the full video.

## 8. Manual correction

Interface: `python -m hoopid.ui …` → `/review`. Operations: assign a player_id to a segment (only within the
shown frame range), mark UNKNOWN, split a track at a frame, merge two tracks,
fix team or number, re-run. A re-run takes 2.6 s: detections and features come from the cache.

Simulated session (`hoopid/correction_sim.py`, operator = ground truth): there are no wrong assignments,
so nothing to fix; reaching ≥95% coverage takes **17 "assign" operations** on the largest UNCERTAIN segments. Result: 95.1% on the whole video
(93.8% on the test half), 0 wrong. It was verified that an assignment does not spread beyond the given range.

**Correction time was not measured**: there was no live operator. The interface logs `seconds_spent` per
operation for a real measurement. **Estimate (assumption):** at 15–30 s per operation this is 4–9 minutes per 55 s of video,
i.e. hours per hour of footage. The "≤10 minutes per hour" target **is not met** on such dense footage;
most corrections are short segments after contacts.

## 9. Next steps with the largest impact

1. **Longer video from a stable camera (ideally 1080p+, landscape) and 2–3 matches** for real independent validation.
   This is the main gap now: 27 s of test is not enough to claim reliability.
2. **Re-ID fine-tuned on basketball crops** (or a stronger model, e.g. OSNet/CLIP-ReID based,
   after a licence check) plus **registration from several episodes**. This will remove most UNKNOWN after contacts.
3. **Jersey numbers**: in official games this is the strongest cue; the module is ready.
4. **Linking segments across a contact by trajectory**: feet and direction of motion before and after the crossing.
   The ablation suggests +5–7 points of coverage, but it can only be enabled after testing on more footage,
   because without segmentation the online pass produced 12 wrong assignments.
5. Then GPU: `onnxruntime-gpu` and FP16 on the RTX 5080, measure speed and VRAM.

## 10. Coach panel (addendum)

Video `results/coach/coach_dashboard.mp4`: on the left the original frame with P01–P09 boxes and detected balls
(orange circles), on the right a panel with running totals for every player and a mini-map (top-down view, estimated).
Final numbers: `results/coach/coach_stats.csv` and `.json`. Command:

```bash
python -m hoopid.ball  --video V --out runs/full/ball.npz          # ball search (full frame + zoomed tiles)
python -m hoopid.stats --tracks results/system_corrected/tracks.csv --ball runs/full/ball.npz \
    --out results/coach --video V
```

The statistics are computed on the run **after manual correction** (95.1% coverage). Only frames
with a confirmed ID are counted, so all values are lower bounds.

| Player | On screen, s | With ball, s | Touches | Distance, m | of which running, m | Max speed, m/s | Moving |
|---|---|---|---|---|---|---|---|
| P01 | 40.7 | 3.8 | 6 | 51.6 | 26.9 | 2.7 | 61% |
| P02 | 26.2 | 0.0 | 0 | 12.8 | 3.3 | 1.0 | 35% |
| P03 | 32.7 | 0.5 | 2 | 48.2 | 28.6 | 3.4 | 77% |
| P04 | 24.4 | 9.9 | 5 | 10.1 | 0.0 | 1.5 | 35% |
| P05 | 15.9 | 0.4 | 1 | 9.9 | 2.4 | 1.3 | 44% |
| P06 | 13.3 | 0.0 | 0 | 3.6 | 1.2 | 2.0 | 50% |
| P07 | 5.8 | 0.0 | 0 | 7.5 | 3.7 | 2.9 | 98% |
| P08 | 8.6 | 0.0 | 0 | 14.2 | 10.6 | 3.6 | 88% |
| P09 | 8.6 | 0.0 | 0 | 3.4 | 0.0 | 1.5 | 59% |

How far this can be trusted:

* **Numbers:** P01–P09 are registration IDs. There are no jersey numbers in the video; if the coach enters a number
  in the registry, it can be shown on the panel.
* **With ball** = frames where a detected ball is at the player's hands, feet (dribbling) or overhead, with
  an unambiguous nearest player. Spot check of 30 frames: in 27 the ball really is with that player,
  in 3 it cannot be judged from the crop (player cut by the frame edge or ball off to the side), no clearly wrong ones.
  **Recall is low:** a ball was found in only 643 of 1638 frames (the COCO "sports ball" class sees a 10–25 px basketball
  poorly, especially with motion blur or overhead during a shot). So time with ball is **underestimated**.
  This is a warm-up: many players have their own ball, so this is "time with a ball", not game possession.
* **Distance and speed are estimates without court calibration.** Depth = f·H/h and lateral offset = (u−cx)·H/h,
  where h is the box height, H is the player's height (1.75 m by default for everyone; the coach can set `height_m` per player
  in `match/players.json`), and f is the focal length from an assumed phone field of view of 65°. Expected error
  ±30%. For children of ~1.5 m the default height overestimates distance by about 15%. Crouching, jumping and
  partially visible boxes create outliers: steps faster than 9 m/s are dropped. Distance while
  the player is off screen or UNKNOWN is not counted.
* **Sprints** (≥4.5 m/s for longer than 0.7 s) were not found in this clip.

Metric numbers require court calibration: 4 court-marking points with known dimensions and
a floor homography. Too few markings are visible in this portrait clip to do that reliably.
A tripod camera covering the whole court is needed.
