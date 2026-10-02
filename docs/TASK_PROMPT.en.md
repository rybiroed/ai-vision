English translation of the original brief (`docs/TASK_PROMPT.md`, Russian).

Test video: `data/raw/VIDEO-2026-10-02-14-05-11.mp4` (original, not re-encoded; SHA-256 in `data/raw/SHA256SUMS`).

---

You are a computer vision engineer. Build a working prototype that preserves the identity of basketball players in the attached video. Do not stop at recommendations: write the code, run the processing and deliver validation results.

1. Goal

The user registers players as PLAYER_01, PLAYER_02 and so on. The system tracks each player, keeps their persistent Player ID through crossings and restores it after temporary disappearance.

The top priority is minimising wrong identity assignment. When information is insufficient, show UNKNOWN; do not guess.

Do not promise error-free recognition. Separate results measured on the attached video from unconfirmed assumptions.

2. Environment

Target machine:

* NVIDIA RTX 5080, 16 GB VRAM;
* 96 GB RAM;
* Ryzen 9950X;
* Windows/WSL or Linux — check the actual environment.

First check GPU availability, PyTorch/CUDA compatibility and the existing project. Do not replace the working environment and do not delete existing files. Use a separate environment and project directory.

If you work on another machine, state its specifications explicitly. Do not present that machine's results as RTX 5080 performance.

3. Video audit

Before choosing models:

1. Find the attached file.
2. Measure resolution, duration, FPS, codec and available timestamps.
3. Determine whether the camera is static, and whether there is panning, zoom or editing cuts.
4. Extract examples of near and far players.
5. Assess the readability of numbers, shoes, hair and tattoos.
6. Find episodes with crossings, screens, leaving the frame and returning.
7. State the limitations of the source recording.

Do not use upscaling or generative restoration as evidence that a number, tattoo or other detail exists.

If the video is unavailable, say so; do not substitute another file.

4. Scope of the first prototype

Implement:

* player detection;
* exclusion of referees, spectators and people outside the playing area;
* short trajectories;
* registration of persistent Player IDs;
* re-identification;
* annotated video and data export;
* manual correction;
* error evaluation.

Do not include automatic counting of shots, passes, possessions and points in the first stage. Do not build a full commercial website.

Registration belongs to a specific match. Do not assume appearance, shoes and uniform persist between matches.

5. Architecture

Separate:

* detection_id — a detection in a single frame;
* track_id — a temporary trajectory;
* player_id — a persistent registered identity.

One player_id can be linked to several consecutive track_ids.

Modules:

1. Video and timestamp reading.
2. Player detector.
3. MOT tracker.
4. Appearance feature extraction.
5. Team and number recognition.
6. Persistent identity manager.
7. Review and manual correction.
8. Export and evaluation.

Use ready-made models as a base. Justify the choice and check the licences of code, weights and dependencies. Record model versions and sources.

6. Player registration

Build a simple local interface:

* choose a frame and a player;
* assign PLAYER_XX;
* enter team and number if known;
* add several confirmed examples from different angles;
* view and delete a wrong example.

For each player store:

* team and number;
* confirmed images;
* appearance features;
* quality and view angle of each example;
* registration source;
* history of confirmations and corrections.

Do not register a confident identity from a single blurry frame.

7. Identification cues

Use the available cues:

A. Team and jersey number.

* Keep recognition candidates and their confidence.
* Combine results from several good-quality frames.
* Do not allow confirmation from a single weak OCR result.
* Account for possible duplicate numbers and roster errors.

B. Overall body appearance — Re-ID.

* Use several views.
* Assess image quality and occlusion separately.

C. Shoes.

* Main colours, sole contrast and visible pattern.
* Isolate shoe regions where possible.
* Do not use floor colour as part of the shoe feature.

D. Hair.

* Colour, length and hairstyle silhouette.
* Use only when the head is large enough.

E. Tattoos.

* Consider location and visual features only when actually legible.
* Do not treat the absence of a visible tattoo as evidence of its absence.
* If quality is insufficient, disable this cue and explain why.

F. Visible skin tone.

* Only as an auxiliary cue for exposed skin.
* Account for lighting, shadows and white balance.
* Do not determine race or ethnicity.
* Do not assign identity based on this cue alone.

G. Motion and geometry.

* Previous position, direction, speed, time absent.
* With calibration, position on the court.
* With camera movement, compensate or reset an unreliable prediction.

Do not set arbitrary fixed feature weights as the final decision. Weight depends on visibility, quality and reliability. Tune thresholds on labelled episodes.

8. Persistent identity rules

Mandatory rules:

* A Player ID is not handed to another player because of proximity.
* One Player ID is not assigned to two different simultaneously visible players.
* To match several players use joint assignment, not independent nearest-neighbour choice.
* Check the best candidate and its margin over the second.
* When in doubt keep UNKNOWN and the candidates.
* On full occlusion show the LOST status; do not present a position prediction as an observation.
* After long absence reduce trust in the motion prediction.
* After a video cut do not carry over by the old frame's coordinates.
* Do not add uncertain observations to a player's confirmed memory.
* Log the reasons for every ID assignment and change.

States: UNREGISTERED, CONFIRMED, UNCERTAIN, LOST.

For recorded video implement a second pass: later readable frames may refine the identity of an earlier fragment. Keep the original decision and the changes for review.

9. Manual correction

The user must be able to:

* assign the correct player_id to a fragment;
* split a trajectory at the moment of a jump;
* merge fragments of the same player;
* mark an interval as UNKNOWN;
* fix a number or team;
* re-process the affected section.

Do not propagate a correction across the whole trajectory automatically if it contains several players.

10. Quality validation

First build a baseline: detector + tracker without extended memory. Then compare it with the persistent identity system on the same episodes.

Prepare a manual ground-truth labelling:

* several normal episodes;
* at least 10 hard episodes, if present;
* crossings of similar players;
* disappearance and return;
* unreadable numbers;
* substitutions and camera movement, if present.

Separate tuning and validation episodes. Do not tune thresholds on the final test. If the video is short, state explicitly that independent validation is insufficient.

Measure:

* ID switches;
* wrong assignments of a registered identity;
* share of correctly confirmed observations;
* share of UNKNOWN;
* coverage: what part of visible time was identified correctly;
* recovery after disappearance;
* manual correction time;
* processing time and peak VRAM.

With sufficient ground truth add IDF1 and HOTA. Do not compute metrics from your own predictions instead of ground truth.

Check the contribution of cues: compare the baseline, adding Re-ID, numbers and additional details. If a cue worsens the result or is indistinguishable, do not enable it by default.

Target: 95% correctly confirmed visible time and no more than 10 minutes of correction per hour of footage. This is a goal, not a criterion that may be declared met without measurement.

11. Performance

Keep the original video. A downscaled copy is allowed for the detector, but analyse numbers and details on crops of the original.

Do not load the whole match into RAM. Process a stream or chunks with correct state carried between them.

Use FP16 and optimisation only after obtaining a verifiable baseline result. Do not promise real-time processing before measuring.

12. Deliverables

Provide:

* source code;
* README with installation and launch instructions;
* model and threshold configuration;
* annotated video;
* players.json — player registry;
* tracks.csv — time, coordinates, track_id, player_id, status and scores;
* assignments.jsonl — log of assignments and corrections;
* a report comparing the baseline and improved versions;
* examples of remaining errors;
* actual speed and memory figures.

Call an uncalibrated estimate a score, not a probability. Do not present it as the probability of correct identification.

13. Order of work

1. Environment and video audit.
2. Short baseline run.
3. Player registration and persistent identities.
4. Handling hard episodes.
5. Manual correction.
6. Comparative validation.
7. Processing the full video.
8. Final report.

Do not stop after the plan if you can continue implementing. Make routine decisions yourself. If blocked by a missing file, an unavailable model or lack of resources, state the specific reason and continue with independent parts.

In the final answer report: what was implemented, which commands to run, where the results are, which errors remain and which next step gives the biggest improvement.
