# Manual ground truth

Russian original: `docs/ru/GT_README.ru.md`.

## Method

1. Tracker tracks (configuration at labelling time) were cut into pieces ("units"): at box contacts
   (IoU ≥ 0.05), at gaps > 3 frames, and no longer than 45 frames (1.5 s). Unit membership: `units.json`
   (detection indices in the precompute `runs/full/precompute`).
2. Every unit of 4+ frames (456 units, 8514 detections) was labelled by hand from a sheet of 6 crops
   sampled evenly across the unit. Labels:
   * participant name (`tank`, `navy_man`, `black_print`, `rtn_kid`, `olive_man`, `red_shorts`, `gray_man`,
     `old_tank`, `black_khaki_man`): descriptive, unrelated to real names;
   * `OTHER`: anyone else (neighbouring courts, by the windows, passers-by);
   * `AMBIG`: two people visible in the unit, or impossible to judge from crops. Such units are excluded from metrics.
3. `det_labels.csv`: a label per detection; `unit_labels.txt`: unit labels;
   `registration_map.json`: which `PLAYER_XX` corresponds to which label.
4. 842 of 9356 detections (9%) are not labelled: 477 belong to no track, 365 lie in units shorter than 4 frames. They are not part of the metrics.

Labelling was done by one annotator (the prototype's author, an AI assistant) from crops of 40–470 px. There was no
independent cross-check. Labelling errors are possible; one was found and fixed (see the log).

## Split

* tuning: frames 0–819 (registration, score model, thresholds, feature selection);
* test: frames 820–1637 (final evaluation only).

## Change log

* **v1**: initial labelling. A man in a black logo T-shirt and beige trousers was labelled as two
  people: `black_khaki_man` (frames 0–300) and `dread_man` (frames 819+), decided from small crops.
* **v2**: after the system assigned PLAYER_09 to `dread_man` on the test half, large crops of all
  appearances were re-checked (`docs/img/gt_fix_same_person.jpg`): the same dreadlocks, logo T-shirt,
  trousers and shoes everywhere. Units 418, 419, 420, 424, 492, 494 were renamed to `black_khaki_man`, and 635, 636, 639
  moved from `AMBIG` to `black_khaki_man`.
  All these units are in the test half; the change did not affect tuning or parameter selection.
  Metrics against v1 are kept in `results/experiments_gt_v1_before_gt_fix.json`.
