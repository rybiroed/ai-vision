# Business plan: "personal player video and stats" built on hoopid

This is my view as an engineer on how to turn the prototype into a product. Technical figures come
from `docs/REPORT.md` (measured on one 55-second clip). All market figures are **hypotheses to
validate**, not research data. Russian original: `docs/ru/BUSINESS_PLAN.ru.md`.

## 1. Problem and customer

Amateur and youth basketball (clubs, AAU-style leagues, camps, schools) is filmed on phones.
Parents, coaches and players want **video of their own player**: possession clips, minutes on court,
and later simple stats. Today this is done by hand: hours of viewing or a video analyst.

Key point: such footage has **no uniforms or numbers** (like our clip) or barely legible numbers.
Solutions that rely on jersey numbers do not work here. Keeping identity by appearance within one match,
confirmed by a human, is what the prototype can do.

Segments (in order of entry):
1. **Coaches and academies** (B2B): player reviews after practices and games, a report per player.
2. **Parents and players** (B2C through the academy): a personal highlight reel and minutes on court.
3. **Tournament and camp organisers**: a "video of every participant" package as a paid add-on.

## 2. Product stages

| Stage | What we sell | Technology | Done when |
|---|---|---|---|
| **MVP (3 months)** | "Player reel": upload a recording → tag players in 3–5 frames → get a separate reel per player and time on court | current prototype + GPU service + operator review | on 10+ real matches: 0 wrong assignments after review, ≤ 15 min of operator work per match |
| **v1 (6–9 months)** | + presence stats, heat map (court calibration), coach export | court homography, fine-tuned Re-ID, numbers when uniforms exist | ≥ 95% coverage without corrections on independent matches |
| **v2 (12+ months)** | shots, passes, possessions (currently out of scope) | ball and event detection | separate validation |

Product principle inherited from the prototype: **"I don't know" is better than someone else's name**.
Doubtful fragments go to a human reviewer, not into the client's reel.

## 3. Revenue model (hypotheses)

* **Academies/teams:** monthly subscription per team (N matches processed) plus a fee per extra match.
* **Tournaments/camps:** per participant per event (reel and minutes) or revenue share with the organiser.
* **Parents:** one-off purchase of their child's reel through the academy or tournament.

Prices must be validated with interviews and a pilot. Starting hypothesis: a player reel must cost clearly less
than an hour of a video analyst's work, otherwise the client will make it by hand.

## 4. Unit economics: what is known and what is not

| Item | Measured | Still to measure |
|---|---|---|
| Compute | CPU 4 vCPU: 0.088× real time, i.e. ~11.4 CPU-hours per hour of video. Detector 59% of the time, Re-ID 35% | speed on RTX 5080 / cloud GPU with FP16. I expect a many-fold speed-up (**not measured**) |
| Operator review | 17 operations per 55 s of dense video (model with a perfect operator) | real operator time. The UI already logs `seconds_spent`. **This is the main margin risk now**: on dense footage corrections take hours per hour of recording |
| Storage | source 8.9 MB per 55 s (≈ 0.6 GB/h at this quality) | retention policy: delete sources after N days |

Conclusion: the product is viable only if (1) GPU processing is cheap and (2) operator work drops
to minutes per match. Point 2 depends on Re-ID quality and on filming: a stable camera, landscape frame, 1080p.
So the first investment is data and fine-tuning, not the interface.

## 5. How to cut operator work (technical plan)

1. Filming requirements for clients: tripod, landscape frame, whole court, 1080p/30.
   This is cheap and has the biggest effect on results.
2. Fine-tune Re-ID on basketball crops from pilot matches (with consent) and take registration from several episodes.
3. Link segments across contacts using foot trajectory and direction of motion: +5–7 points of coverage in the ablation.
4. Uniforms and numbers where they exist: the module is ready.
5. Review queue ordered by value: the longest UNKNOWN segments first (17 operations closed the gap 91% → 95%).

## 6. Competition (to verify)

There are mature products for automatic filming and video analytics in sport: smart cameras with automatic
tracking and video-analysis platforms for teams. They are strong where they have their own camera and uniforms with numbers.
hoopid's positioning: **any phone clip, no uniforms, with a "we won't mix up your child" guarantee**
thanks to human review. A review of specific products, prices and features is needed before launch;
I do not list them here so as not to present unverified data as fact.

## 7. Law and ethics (mandatory)

* **Children** are in the frame. Consent from parents and the organiser, a clear privacy policy and
  deletion on request are required. Requirements depend on jurisdiction (e.g. GDPR in the EU, COPPA and state biometric laws in the US,
  152-FZ in Russia). Legal advice is needed **before** the pilot.
* The system **does not recognise faces** and keeps no biometrics between matches: identity lives within
  one match and is built from clothing. This is both a product and a legal advantage, and must be kept.
* Do not infer race or ethnicity, do not use skin colour as a cue (the prototype does not extract it).
* Model licences: the Re-ID and detector weights were trained on datasets with restrictions. Commercial use needs own fine-tuning (see `MODELS_AND_LICENSES.md`).

## 8. 90-day plan

| Weeks | Action | Result |
|---|---|---|
| 1–2 | Run the prototype on an RTX 5080 (speed, VRAM); 3–5 full-match recordings from 1–2 academies on a tripod | real speed and operator-work figures |
| 3–6 | Label 2–3 matches, independent validation; fine-tune Re-ID; link across contacts | coverage and errors on unseen matches |
| 7–10 | Service: upload → registration → review → player reels; operator time logging | MVP for the pilot |
| 11–13 | Paid pilot with 2–3 academies | willingness to pay, NPS, actual margin |

**Go / stop decision** after the pilot: 0 wrong names in delivered reels, ≤ 15 minutes of
review per match, at least 2 of 3 pilot clients willing to pay.
