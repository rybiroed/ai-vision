"""Persistent identity manager.

Units
  detection  : one box in one frame (det_id)
  track      : short-term tracker output (track_id)
  segment    : part of a track between "risk events" (contact with another
               player, a long unobserved gap, or a manual split). Identity is
               decided per segment, so a tracker swap during a crossing cannot
               drag a confirmed Player ID onto the wrong person.
  player     : registered persistent identity (player_id)

Evidence
  For every observation o and registered player p we compute a per-feature
  similarity (Re-ID cosine, colour-histogram similarities of torso/legs/shoes/
  hair). A calibrator (logistic model fitted on the tuning episodes, see
  calibrate.py) turns them into one evidence value L(o,p) (a logit, i.e. an
  uncalibrated *score*, not a probability). Missing features contribute
  nothing. Each observation is weighted by its quality q(o) (size, occlusion,
  truncation). Segment evidence E(s,p) is the q-weighted mean of L(o,p).

Decision (online pass, frame by frame)
  * joint Hungarian assignment between visible segments and players;
  * accept s->p only if E >= abs_thr, enough evidence (n_eff >= n_min), row
    margin to the second-best player >= margin, and column margin to any other
    visible segment >= col_margin;
  * a player already CONFIRMED on another live segment is never given to a
    second segment (exclusivity), the newcomer stays UNCERTAIN;
  * a CONFIRMED segment keeps its player unless its evidence falls below
    abs_thr - hysteresis or another player overtakes it by margin;
  * after a contact/gap boundary the next segment of the same track gets only a
    small prior toward the previous player and must re-confirm.
Second pass (offline, recorded video)
  * whole-segment evidence (including later frames) and a global greedy
    assignment that forbids the same player on time-overlapping segments;
  * a contact segment inherits a player only if the segments before and after
    it on the same track agree.
Every decision is logged with its reason.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from .features import HIST_DIM
from .tracker import iou_matrix

FEATS = ["reid", "torso", "legs", "shoes", "hair"]


# ----------------------------------------------------------------------------- similarities

def similarities(pre: dict, idx: np.ndarray, gal: dict, feats=FEATS) -> dict[str, np.ndarray]:
    """sims[f] has shape (len(idx), n_players); NaN where unavailable."""
    pids = list(gal)
    out = {}
    for f in feats:
        m = np.full((len(idx), len(pids)), np.nan, np.float32)
        if f == "reid":
            X = pre["reid"][idx].astype(np.float32)
            for j, p in enumerate(pids):
                c = X @ gal[p]["reid"].T  # (n, k)
                k = min(2, c.shape[1])
                m[:, j] = np.sort(c, axis=1)[:, -k:].mean(1)
        else:
            X = pre[f][idx]
            okx = ~np.isnan(X).any(1)
            sx = np.sqrt(np.clip(np.nan_to_num(X), 0, None))
            for j, p in enumerate(pids):
                G = gal[p][f]
                okg = ~np.isnan(G).any(1)
                if not okg.any():
                    continue
                sg = np.sqrt(np.clip(G[okg], 0, None))
                bc = sx @ sg.T
                m[okx, j] = bc[okx].max(1)
        out[f] = m
    return out


def quality(pre: dict, idx: np.ndarray, cfg: dict) -> np.ndarray:
    h = pre["q_height"][idx]
    q = np.clip((h - cfg["q_h0"]) / (cfg["q_h1"] - cfg["q_h0"]), 0, 1)
    q *= np.clip(1 - pre["q_occ"][idx] / cfg["q_occ_max"], 0, 1)
    q *= np.where(pre["q_trunc"][idx] > 0, cfg["q_trunc_factor"], 1.0)
    return q.astype(np.float32)


class Calibrator:
    """Linear evidence model: L = b + sum_f w_f * (s_f - c_f) over available f."""

    def __init__(self, params: dict):
        self.p = params

    @classmethod
    def load(cls, path: str | None, default: dict) -> "Calibrator":
        if path:
            try:
                with open(path) as fh:
                    return cls(json.load(fh))
            except FileNotFoundError:
                pass
        return cls(default)

    def __call__(self, sims: dict[str, np.ndarray], enabled: list[str]) -> np.ndarray:
        n, P = next(iter(sims.values())).shape
        L = np.full((n, P), self.p.get("bias", 0.0), np.float32)
        for f in enabled:
            if f not in sims or f not in self.p["w"]:
                continue
            s = sims[f]
            contrib = self.p["w"][f] * (s - self.p["c"][f])
            L += np.nan_to_num(contrib, nan=0.0)
        return L


# ----------------------------------------------------------------------------- segments

@dataclass
class Segment:
    sid: int
    track_id: int
    first: int
    last: int
    obs: list = field(default_factory=list)  # indices into pre arrays
    kind: str = "normal"  # normal | contact
    prev_sid: int | None = None
    forced: str | None = None  # manual lock: player_id or "UNKNOWN"
    forced_source: dict | None = None


def build_segments(pre: dict, track_of: np.ndarray, cfg: dict, corrections: list[dict]) -> list[Segment]:
    frames = pre["frame"]
    boxes = pre["box"]
    tracked = np.where(track_of >= 0)[0]
    # contact: overlap with another tracked detection in the same frame
    contact = np.zeros(len(frames), bool)
    by_frame = defaultdict(list)
    for i in tracked:
        by_frame[int(frames[i])].append(i)
    for f, ids in by_frame.items():
        if len(ids) < 2:
            continue
        ids = np.array(ids)
        iou = iou_matrix(boxes[ids], boxes[ids])
        np.fill_diagonal(iou, 0)
        contact[ids] = iou.max(1) >= cfg["contact_iou"]
    splits = defaultdict(set)
    locks = []
    for c in corrections:
        if c["op"] == "split":
            splits[int(c["track_id"])].add(int(c["frame"]))
        elif c["op"] in ("assign", "unknown"):
            a, b = c["frames"]
            splits[int(c["track_id"])].update({int(a), int(b) + 1})
            locks.append(c)
    segs: list[Segment] = []
    for tid in np.unique(track_of[tracked]):
        ids = tracked[track_of[tracked] == tid]
        ids = ids[np.argsort(frames[ids])]
        cur: Segment | None = None
        last_f = None
        for i in ids:
            f = int(frames[i])
            kind = "contact" if contact[i] else "normal"
            new = (cur is None or kind != cur.kind
                   or (last_f is not None and f - last_f > cfg["gap_split_frames"])
                   or any(last_f is not None and last_f < s <= f for s in splits[int(tid)]))
            if new:
                prev = cur.sid if cur is not None else None
                cur = Segment(len(segs), int(tid), f, f, kind=kind, prev_sid=prev)
                segs.append(cur)
            # short contact blips (<= contact_min_frames) are merged into the running segment
            cur.obs.append(int(i)); cur.last = f
            last_f = f
    # merge tiny contact segments back (they are noise, not crossings)
    merged: list[Segment] = []
    for s in segs:
        if (s.kind == "contact" and len(s.obs) < cfg["contact_min_frames"] and merged
                and merged[-1].track_id == s.track_id and s.prev_sid == merged[-1].sid
                and not any(merged[-1].last < x <= s.first for x in splits[s.track_id])):
            merged[-1].obs += s.obs; merged[-1].last = s.last
            continue
        if (s.kind == "normal" and merged and merged[-1].track_id == s.track_id
                and merged[-1].kind == "normal" and s.first - merged[-1].last <= cfg["gap_split_frames"]
                and not any(merged[-1].last < x <= s.first for x in splits[s.track_id])):
            merged[-1].obs += s.obs; merged[-1].last = s.last
            continue
        merged.append(s)
    for k, s in enumerate(merged):
        s.sid = k
        s.prev_sid = k - 1 if k > 0 and merged[k - 1].track_id == s.track_id else None
    for c in locks:
        for s in merged:
            if s.track_id == int(c["track_id"]) and s.first >= c["frames"][0] and s.last <= c["frames"][1]:
                s.forced = c.get("player_id", "UNKNOWN") if c["op"] == "assign" else "UNKNOWN"
                s.forced_source = c
    return merged


# ----------------------------------------------------------------------------- manager

class IdentityManager:
    def __init__(self, pre: dict, track_out: dict, gal: dict, cfg: dict, calib: Calibrator,
                 corrections: list[dict] | None = None, enabled: list[str] | None = None):
        self.pre, self.cfg, self.gal, self.calib = pre, cfg, gal, calib
        self.pids = list(gal)
        self.enabled = enabled or cfg["features_enabled"]
        self.track_of = track_out["track_of"]
        self.lost = track_out["lost"]
        self.corrections = corrections or []
        self.log: list[dict] = []
        all_obs = np.where(self.track_of >= 0)[0].astype(np.int64)
        self.obs_index = {int(i): k for k, i in enumerate(all_obs)}
        self.all_obs = all_obs
        if len(self.pids) and len(all_obs):
            sims = similarities(pre, all_obs, gal, self.enabled)
            self.L = calib(sims, self.enabled)
            self.sims = sims
        else:
            self.L = np.zeros((len(all_obs), len(self.pids)), np.float32)
            self.sims = {}
        self.q = quality(pre, all_obs, cfg) if len(all_obs) else np.zeros(0, np.float32)
        self.segs = build_segments(pre, self.track_of, cfg, self.corrections)
        if cfg.get("appearance_split", True):
            self.segs = self._appearance_splits(self.segs)
        self.seg_of_obs = {}
        for s in self.segs:
            for i in s.obs:
                self.seg_of_obs[i] = s.sid

    # --------------------------------------------------------------- appearance change points
    def _appearance_splits(self, segs: list[Segment]) -> list[Segment]:
        """Split segments where the tracker silently jumped to another person.

        Compares mean Re-ID embeddings of the w observations before and after
        each position. A split is made where similarity is very low, or where
        it is moderately low AND the best registered player changes with a
        clear margin in both windows. Needs w observations after the jump, so
        the online pass works with a w-frame look-ahead buffer (~0.2 s).
        """
        cfg = self.cfg
        w = cfg["split_window"]
        E = self.pre["reid"].astype(np.float32)
        H = self.pre["q_height"]
        out: list[Segment] = []
        for s in segs:
            obs = [i for i in s.obs]
            valid = [i for i in obs if H[i] >= cfg["split_min_height"]]
            cuts = []
            if len(valid) >= 2 * w:
                best = None
                for k in range(w, len(valid) - w + 1):
                    a = E[valid[k - w:k]].mean(0); b = E[valid[k:k + w]].mean(0)
                    c = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
                    hard = c < cfg["split_cos_hard"]
                    soft = False
                    if not hard and c < cfg["split_cos_soft"] and self.pids:
                        ea, _ = self._evidence(valid[k - w:k]); eb, _ = self._evidence(valid[k:k + w])
                        ja, jb = int(np.argmax(ea)), int(np.argmax(eb))
                        if ja != jb:
                            ma = ea[ja] - np.delete(ea, ja).max() if len(ea) > 1 else 0
                            mb = eb[jb] - np.delete(eb, jb).max() if len(eb) > 1 else 0
                            soft = ma >= cfg["split_margin"] and mb >= cfg["split_margin"]
                    if (hard or soft) and (best is None or c < best[0]):
                        best = (c, k, "hard" if hard else "soft")
                    elif best is not None and k - best[1] >= w:
                        cuts.append(best); best = None
                if best is not None:
                    cuts.append(best)
            if not cuts:
                out.append(s); continue
            cut_frames = sorted(int(self.pre["frame"][valid[k]]) for _, k, _ in cuts)
            for c, k, kind in cuts:
                self._note(pass_="segmentation", event="appearance_split", track_id=s.track_id,
                           frame=int(self.pre["frame"][valid[k]]), reid_cos=round(c, 3), kind=kind)
            bounds = [s.first] + cut_frames + [s.last + 1]
            for a, b in zip(bounds, bounds[1:]):
                part = [i for i in obs if a <= int(self.pre["frame"][i]) < b]
                if not part:
                    continue
                ns = Segment(0, s.track_id, int(self.pre["frame"][part[0]]), int(self.pre["frame"][part[-1]]),
                             obs=part, kind=s.kind, forced=s.forced, forced_source=s.forced_source)
                ns.split_reason = "appearance" if a != s.first else getattr(s, "split_reason", None)
                out.append(ns)
        for k, s in enumerate(out):
            s.sid = k
            same = k > 0 and out[k - 1].track_id == s.track_id
            # no continuity prior across an appearance split: the person changed
            s.prev_sid = k - 1 if same and getattr(s, "split_reason", None) != "appearance" else None
        return out

    # --------------------------------------------------------------- helpers
    def _evidence(self, rows: list[int]) -> tuple[np.ndarray, float]:
        if not rows or not self.pids:
            return np.zeros(len(self.pids), np.float32), 0.0
        k = np.array([self.obs_index[i] for i in rows])
        q = self.q[k]
        n = float(q.sum())
        if n <= 1e-6:
            return np.zeros(len(self.pids), np.float32), 0.0
        return (q[:, None] * self.L[k]).sum(0) / n, n

    def _candidates(self, E: np.ndarray, k: int = 3) -> list:
        o = np.argsort(-E)[:k]
        return [[self.pids[j], round(float(E[j]), 3)] for j in o]

    def _note(self, **kw):
        self.log.append(kw)

    # --------------------------------------------------------------- online pass
    def run_online(self) -> dict[int, dict]:
        cfg, P = self.cfg, len(self.pids)
        frames = self.pre["frame"]
        seg_state: dict[int, dict] = {}
        held: dict[str, int] = {}  # player -> sid currently CONFIRMED and live
        ended: dict[str, dict] = {}  # player -> info about where/when its last segment ended
        out: dict[int, dict] = {}
        by_frame = defaultdict(list)
        for i in self.all_obs:
            by_frame[int(frames[i])].append(int(i))
        seg_end = {s.sid: s.last for s in self.segs}
        cam_frames = self.pre["cam_frame"]
        cam_ok = self.pre["cam_reliable"] & ~self.pre["cam_cut"]
        cam_bad_cum = np.concatenate([[0], np.cumsum(~cam_ok)])
        fpos = {int(f): k for k, f in enumerate(cam_frames)}
        rows_so_far: dict[int, list] = defaultdict(list)

        def release(sid, f, reason):
            st = seg_state[sid]
            pid = st["pid"]
            if pid and held.get(pid) == sid:
                held.pop(pid)
                s = self.segs[sid]
                ended[pid] = {"frame": f, "box": self.pre["box"][s.obs[-1]].copy(), "sid": sid}
                self._note(pass_="online", frame=f, event="released", segment=sid, track_id=s.track_id,
                           player_id=pid, reason=reason)

        live_prev: set = set()
        for f in sorted(by_frame):
            vis = by_frame[f]
            vis_sids = sorted({self.seg_of_obs[i] for i in vis})
            # segments that ended before this frame release their player
            for sid in list(live_prev):
                if seg_end[sid] < f and sid in seg_state:
                    release(sid, f, "segment_ended")
            live_prev = {sid for sid in live_prev if seg_end[sid] >= f} | set(vis_sids)
            for i in vis:
                rows_so_far[self.seg_of_obs[i]].append(i)
            E = np.zeros((len(vis_sids), P), np.float32)
            N = np.zeros(len(vis_sids), np.float32)
            for a, sid in enumerate(vis_sids):
                s = self.segs[sid]
                st = seg_state.setdefault(sid, {"status": "UNCERTAIN", "pid": None, "prior": None,
                                                "first_frame": f})
                e, n = self._evidence(rows_so_far[sid])
                # continuity prior from the previous segment of the same track
                if st["prior"] is None and s.prev_sid is not None and s.prev_sid in seg_state:
                    pp = seg_state[s.prev_sid]["pid"]
                    st["prior"] = ("continuity", pp) if pp else ("none", None)
                elif st["prior"] is None:
                    # re-entry prior: a player whose segment ended recently close by
                    st["prior"] = ("none", None)
                    best = None
                    for pid, info in ended.items():
                        if pid in held:
                            continue
                        gap = f - info["frame"]
                        if gap > cfg["reentry_max_gap"]:
                            continue
                        k0, k1 = fpos.get(info["frame"]), fpos.get(f)
                        if k0 is None or k1 is None or cam_bad_cum[k1 + 1] - cam_bad_cum[k0 + 1] > 0:
                            continue  # camera motion unknown in between: do not trust geometry
                        b0 = info["box"]; b1 = self.pre["box"][rows_so_far[sid][0]]
                        d = np.hypot((b0[0] + b0[2]) / 2 - (b1[0] + b1[2]) / 2, b0[3] - b1[3])
                        hh = max(b1[3] - b1[1], 10)
                        if d < cfg["reentry_max_dist_h"] * hh and (best is None or gap < best[1]):
                            best = (pid, gap)
                    if best:
                        st["prior"] = ("reentry", best[0], best[1])
                pr = st["prior"]
                if pr and pr[0] == "continuity" and pr[1] in self.pids:
                    e = e.copy(); e[self.pids.index(pr[1])] += cfg["continuity_bonus"]
                elif pr and pr[0] == "reentry" and pr[1] in self.pids:
                    e = e.copy()
                    e[self.pids.index(pr[1])] += cfg["reentry_bonus"] * np.exp(-pr[2] / cfg["reentry_tau"])
                E[a], N[a] = e, n
            # forced (manual) segments
            for a, sid in enumerate(vis_sids):
                s, st = self.segs[sid], seg_state[sid]
                if s.forced and st.get("forced_applied") is None:
                    st["forced_applied"] = True
                    if s.forced == "UNKNOWN":
                        release(sid, f, "manual_unknown") if st["pid"] else None
                        st.update(status="UNCERTAIN", pid=None)
                    else:
                        other = held.get(s.forced)
                        if other is not None and other != sid:
                            seg_state[other].update(status="UNCERTAIN", pid=None)
                            held.pop(s.forced)
                            self._note(pass_="online", frame=f, event="revoked", segment=other,
                                       player_id=s.forced, reason="manual_assign_elsewhere")
                        st.update(status="CONFIRMED", pid=s.forced)
                        held[s.forced] = sid
                    self._note(pass_="online", frame=f, event="manual", segment=sid, track_id=s.track_id,
                               player_id=s.forced, correction=s.forced_source)
            # maintain existing confirmations
            for a, sid in enumerate(vis_sids):
                st, s = seg_state[sid], self.segs[sid]
                if st["status"] != "CONFIRMED" or s.forced or not P:
                    continue
                j = self.pids.index(st["pid"])
                others = np.delete(E[a], j)
                if N[a] >= cfg["n_min"] and (E[a, j] < cfg["abs_thr"] - cfg["hysteresis"] or
                                             (len(others) and others.max() - E[a, j] >= cfg["margin"])):
                    self._note(pass_="online", frame=f, event="revoked", segment=sid, track_id=s.track_id,
                               player_id=st["pid"], reason="evidence_dropped",
                               candidates=self._candidates(E[a]), n_eff=round(float(N[a]), 2))
                    held.pop(st["pid"], None)
                    st.update(status="UNCERTAIN", pid=None)
            # joint assignment for undecided segments
            und = [a for a, sid in enumerate(vis_sids)
                   if seg_state[vis_sids[a]]["status"] != "CONFIRMED" and not self.segs[sid].forced]
            if und and P:
                M = E[und]
                big = np.concatenate([M, np.full((len(und), len(und)), cfg["abs_thr"], np.float32)], 1)
                r, c = linear_sum_assignment(-big)
                for ri, ci in zip(r, c):
                    a = und[ri]; sid = vis_sids[a]; st = seg_state[sid]; s = self.segs[sid]
                    reason = None
                    pr = st["prior"]
                    is_prior = ci < P and pr and pr[0] in ("continuity", "reentry") and pr[1] == self.pids[ci]
                    n_req = cfg["n_min_reconfirm"] if is_prior else cfg["n_min"]
                    if N[a] < n_req:
                        reason = "insufficient_evidence"
                    elif ci >= P:
                        reason = "no_player_above_threshold"
                    else:
                        e1 = E[a, ci]
                        e2 = np.delete(E[a], ci).max() if P > 1 else -np.inf
                        col = [E[b, ci] for b in range(len(vis_sids)) if b != a and N[b] >= cfg["n_min_col"]]
                        extra = cfg.get("contact_extra_margin", 1.0) if s.kind == "contact" and not is_prior else 0.0
                        if e1 - e2 < cfg["margin"] + extra:
                            reason = "ambiguous_between_players"
                        elif col and e1 - max(col) < cfg["col_margin"]:
                            reason = "ambiguous_between_segments"
                        elif self.pids[ci] in held and held[self.pids[ci]] != sid:
                            reason = "player_held_by_other_live_segment"
                    if reason is None:
                        pid = self.pids[ci]
                        st.update(status="CONFIRMED", pid=pid)
                        held[pid] = sid
                        ended.pop(pid, None)
                        self._note(pass_="online", frame=f, event="assigned", segment=sid, track_id=s.track_id,
                                   player_id=pid, score=round(float(E[a, ci]), 3),
                                   candidates=self._candidates(E[a]), n_eff=round(float(N[a]), 2),
                                   prior=pr[0] if pr else None)
                    else:
                        st["status"] = ("UNREGISTERED" if reason == "no_player_above_threshold"
                                        and N[a] >= cfg["n_min"] and E[a].max() < cfg["unregistered_thr"]
                                        else "UNCERTAIN")
                        if st.get("last_reason") != reason:
                            st["last_reason"] = reason
                            self._note(pass_="online", frame=f, event="undecided", segment=sid,
                                       track_id=s.track_id, reason=reason, candidates=self._candidates(E[a]),
                                       n_eff=round(float(N[a]), 2))
            for a, sid in enumerate(vis_sids):
                st = seg_state[sid]
                for i in by_frame[f]:
                    if self.seg_of_obs[i] == sid:
                        out[i] = {"status": st["status"], "player_id": st["pid"] or "UNKNOWN",
                                  "score": round(float(E[a].max()), 3) if P else 0.0,
                                  "margin": round(float(np.sort(E[a])[-1] - np.sort(E[a])[-2]), 3) if P > 1 else 0.0,
                                  "candidates": self._candidates(E[a]) if P else [],
                                  "n_eff": round(float(N[a]), 2), "segment": sid}
        self.online_state = seg_state
        return out

    # --------------------------------------------------------------- offline pass
    def run_offline(self, online: dict[int, dict]) -> dict[int, dict]:
        cfg, P = self.cfg, len(self.pids)
        segE, segN = {}, {}
        for s in self.segs:
            segE[s.sid], segN[s.sid] = self._evidence(s.obs)
        decided: dict[int, str] = {}
        reasons: dict[int, str] = {}
        for s in self.segs:
            if s.forced:
                decided[s.sid] = s.forced
                reasons[s.sid] = "manual"
        occupied: dict[str, list] = defaultdict(list)
        for sid, pid in decided.items():
            if pid != "UNKNOWN":
                occupied[pid].append((self.segs[sid].first, self.segs[sid].last))
        cands = []
        for s in self.segs:
            if s.sid in decided or not P:
                continue
            e, n = segE[s.sid], segN[s.sid]
            o = np.argsort(-e)
            # contact segments may stand on their own evidence, but with a stricter margin
            extra = cfg.get("contact_extra_margin", 1.0) if s.kind == "contact" else 0.0
            if n < cfg["n_min"]:
                reasons[s.sid] = "insufficient_evidence"; continue
            e1 = e[o[0]]; e2 = e[o[1]] if P > 1 else -np.inf
            if e1 < cfg["abs_thr"]:
                reasons[s.sid] = "no_player_above_threshold"; continue
            if e1 - e2 < cfg["margin"] + extra:
                reasons[s.sid] = "ambiguous_between_players"; continue
            cands.append((float(e1 - e2), float(e1), s.sid, self.pids[o[0]]))
        # most decisive first; same player cannot sit on two time-overlapping segments
        for margin, e1, sid, pid in sorted(cands, reverse=True):
            s = self.segs[sid]
            clash = [iv for iv in occupied[pid] if not (s.last < iv[0] or s.first > iv[1])]
            if clash:
                reasons[sid] = "player_taken_by_more_decisive_overlapping_segment"; continue
            # column check: overlapping segment nearly as similar to pid
            rivals = [segE[t.sid][self.pids.index(pid)] for t in self.segs
                      if t.sid != sid and not (t.last < s.first or t.first > s.last) and segN[t.sid] >= cfg["n_min_col"]]
            if rivals and e1 - max(rivals) < cfg["col_margin"]:
                reasons[sid] = "ambiguous_between_segments"; continue
            decided[sid] = pid
            reasons[sid] = "offline_assigned"
            occupied[pid].append((s.first, s.last))
        # contact segments: only when neighbours on the same track agree
        by_track = defaultdict(list)
        for s in self.segs:
            by_track[s.track_id].append(s)
        for segs in by_track.values():
            for k, s in enumerate(segs):
                if s.kind != "contact" or s.sid in decided:
                    continue
                a = decided.get(segs[k - 1].sid) if k > 0 else None
                b = decided.get(segs[k + 1].sid) if k + 1 < len(segs) else None
                if a and a == b and a != "UNKNOWN":
                    clash = [iv for iv in occupied[a] if not (s.last < iv[0] or s.first > iv[1])
                             and iv not in [(segs[k - 1].first, segs[k - 1].last), (segs[k + 1].first, segs[k + 1].last)]]
                    if not clash:
                        decided[s.sid] = a; reasons[s.sid] = "contact_sandwich"
                        occupied[a].append((s.first, s.last))
                        continue
                reasons[s.sid] = "contact_unresolved"
        out = {}
        for s in self.segs:
            pid = decided.get(s.sid)
            e, n = segE[s.sid], segN[s.sid]
            if pid and pid != "UNKNOWN":
                status = "CONFIRMED"
            elif reasons.get(s.sid) == "no_player_above_threshold" and P and e.max() < cfg["unregistered_thr"]:
                status, pid = "UNREGISTERED", None
            else:
                status, pid = "UNCERTAIN", None
            on_pids = {online[i]["player_id"] for i in s.obs if i in online}
            if on_pids != {pid or "UNKNOWN"}:
                self._note(pass_="offline", event="revised", segment=s.sid, track_id=s.track_id,
                           frames=[s.first, s.last], online=sorted(on_pids), final=pid or "UNKNOWN",
                           reason=reasons.get(s.sid), candidates=self._candidates(e) if P else [],
                           n_eff=round(float(n), 2))
            for i in s.obs:
                out[i] = {"status": status, "player_id": pid or "UNKNOWN",
                          "score": round(float(e.max()), 3) if P else 0.0,
                          "margin": round(float(np.sort(e)[-1] - np.sort(e)[-2]), 3) if P > 1 else 0.0,
                          "candidates": self._candidates(e) if P else [], "n_eff": round(float(n), 2),
                          "segment": s.sid, "reason": reasons.get(s.sid)}
        return out


# ----------------------------------------------------------------------------- baseline

def baseline_identity(pre: dict, track_out: dict, gal: dict, cfg: dict, k_first: int = 5) -> dict[int, dict]:
    """Naive comparison system: detector + tracker, each track gets the nearest
    registered player (Re-ID cosine) from its first k observations, no margin
    test, no exclusivity, no segments, no memory, never revised."""
    track_of = track_out["track_of"]
    pids = list(gal)
    out = {}
    for tid in np.unique(track_of[track_of >= 0]):
        ids = np.where(track_of == tid)[0]
        ids = ids[np.argsort(pre["frame"][ids])]
        s = similarities(pre, ids[:k_first], gal, ["reid"])["reid"].mean(0)
        j = int(np.argmax(s))
        pid = pids[j] if s[j] >= cfg["baseline_min_cos"] else "UNKNOWN"
        for i in ids:
            out[int(i)] = {"status": "CONFIRMED" if pid != "UNKNOWN" else "UNCERTAIN", "player_id": pid,
                           "score": round(float(s[j]), 3), "margin": 0.0, "candidates": [], "n_eff": 0.0,
                           "segment": -1}
    return out
