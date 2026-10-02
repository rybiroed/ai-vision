"""Minimal local web UI: registration, sample review, manual corrections.

    python -m hoopid.ui --video data/raw/VIDEO.mp4 --pre runs/full/precompute \
        --registry match/players.json --corrections match/corrections.json --run runs/full/system
    open http://127.0.0.1:5000

Local only (binds to 127.0.0.1). No accounts, no upload: this is a review
tool, not a website.
"""
from __future__ import annotations

import argparse
import csv
import html
import io
import json
import os
import time
from collections import defaultdict

import cv2
import numpy as np
from flask import Flask, Response, jsonify, redirect, request

from . import corrections as C
from .config import load_config
from .pipeline import run as run_pipeline
from .precompute import load
from .register import add_sample, det_at
from .registry import Registry, registration_problems
from .video import read_frame

PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>hoopid</title>
<style>body{{font-family:system-ui,sans-serif;margin:16px;max-width:1200px}}nav a{{margin-right:16px}}
img{{image-rendering:auto}}table{{border-collapse:collapse}}td,th{{border:1px solid #ccc;padding:3px 6px;font-size:13px}}
.st-CONFIRMED{{color:#080}}.st-UNCERTAIN{{color:#c80}}.st-UNREGISTERED{{color:#888}}.thumb{{height:72px}}
form.inline{{display:inline}}</style></head><body>
<nav><a href="/">Frames / register</a><a href="/players">Players</a><a href="/review">Review &amp; correct</a></nav>
{body}</body></html>"""


def create_app(video, pre_dir, registry, corr_path, run_dir, config):
    app = Flask(__name__)
    pre = load(pre_dir)
    cfg = load_config(config)
    pos = {int(d): k for k, d in enumerate(pre["det_id"])}
    nframes = int(pre["cam_frame"].max()) + 1
    state = {"tracks": None}

    def tracks():
        if state["tracks"] is None:
            p = os.path.join(run_dir, "tracks.csv")
            rows = list(csv.DictReader(open(p))) if os.path.exists(p) else []
            state["tracks"] = rows
        return state["tracks"]

    def page(body):
        return PAGE.format(body=body)

    @app.route("/frame/<int:f>.jpg")
    def frame_img(f):
        img = read_frame(video, f)
        by_det = {int(r["det_id"]): r for r in tracks() if r["observed"] == "1" and int(r["frame"]) == f}
        for i in np.where(pre["frame"] == f)[0]:
            x1, y1, x2, y2 = pre["box"][i].astype(int)
            r = by_det.get(int(pre["det_id"][i]))
            col = (60, 200, 60) if r and r["status"] == "CONFIRMED" else (0, 200, 255)
            cv2.rectangle(img, (x1, y1), (x2, y2), col, 1)
            lab = f"d{int(pre['det_id'][i])}" + (f" {r['player_id'].replace('PLAYER_', 'P')}" if r and r["status"] == "CONFIRMED" else "")
            cv2.putText(img, lab, (x1, max(9, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.33, col, 1)
        ok, buf = cv2.imencode(".jpg", img)
        return Response(buf.tobytes(), mimetype="image/jpeg")

    @app.route("/crop/<int:det>.jpg")
    def crop(det):
        i = pos[det]
        img = read_frame(video, int(pre["frame"][i]))
        x1, y1, x2, y2 = pre["box"][i].astype(int)
        c = img[max(0, y1):y2, max(0, x1):x2]
        ok, buf = cv2.imencode(".jpg", c)
        return Response(buf.tobytes(), mimetype="image/jpeg")

    @app.route("/")
    def index():
        f = int(request.args.get("f", 0))
        reg = Registry(registry)
        opts = "".join(f"<option>{p}</option>" for p in reg.players) + f"<option>{reg.next_id()}</option>"
        body = f"""<h2>Frame {f} / {nframes - 1}</h2>
<form><input name=f type=number value={f} min=0 max={nframes - 1}> <button>go</button>
 <a href="/?f={max(0, f - 15)}">-15</a> <a href="/?f={max(0, f - 1)}">-1</a>
 <a href="/?f={min(nframes - 1, f + 1)}">+1</a> <a href="/?f={min(nframes - 1, f + 15)}">+15</a></form>
<p>Click a player to select a detection. Register only sharp, unoccluded examples; add several moments and angles.</p>
<div style="display:flex;gap:16px"><img id=fr src="/frame/{f}.jpg" style="cursor:crosshair">
<div><form method=post action="/register" id=rf>
<input type=hidden name=frame value={f}><input type=hidden name=x id=x><input type=hidden name=y id=y>
<p>Selected: <span id=sel>none</span></p>
<p>Player <select name=player>{opts}</select></p>
<p>Team <input name=team size=8> Number <input name=number size=4></p>
<p>View <select name=view><option>front</option><option>back</option><option>side</option><option>unknown</option></select></p>
<button>Add sample</button></form></div></div>
<script>
const im=document.getElementById('fr');
im.onclick=e=>{{const r=im.getBoundingClientRect();const x=(e.clientX-r.left)*im.naturalWidth/r.width,y=(e.clientY-r.top)*im.naturalHeight/r.height;
document.getElementById('x').value=x;document.getElementById('y').value=y;
fetch(`/api/pick?frame={f}&x=${{x}}&y=${{y}}`).then(r=>r.json()).then(d=>{{document.getElementById('sel').innerHTML=d.det_id===null?'no detection here':
`det ${{d.det_id}} h=${{d.height}}px occ=${{d.occ}} <img class=thumb src="/crop/${{d.det_id}}.jpg">`+(d.warn?`<br><b>${{d.warn}}</b>`:'')}})}}
</script>"""
        return page(body)

    @app.route("/api/pick")
    def pick():
        i = det_at(pre, int(request.args["frame"]), float(request.args["x"]), float(request.args["y"]))
        if i is None:
            return jsonify({"det_id": None})
        h, occ = float(pre["q_height"][i]), float(pre["q_occ"][i])
        warn = []
        if h < cfg["identity"]["reg_min_height"]:
            warn.append(f"small ({h:.0f}px): weak example")
        if occ >= 0.3:
            warn.append("occluded")
        if pre["q_trunc"][i]:
            warn.append("cut by frame edge")
        return jsonify({"det_id": int(pre["det_id"][i]), "height": round(h), "occ": round(occ, 2),
                        "warn": "; ".join(warn)})

    @app.route("/register", methods=["POST"])
    def register():
        f = int(request.form["frame"])
        i = det_at(pre, f, float(request.form["x"] or -1), float(request.form["y"] or -1))
        if i is None:
            return redirect(f"/?f={f}")
        reg = Registry(registry)
        add_sample(reg, pre, video, request.form["player"], i, request.form["view"], "user:ui",
                   request.form.get("team") or None, request.form.get("number") or None,
                   os.path.join(os.path.dirname(registry), "samples"))
        reg.save()
        return redirect(f"/?f={f}")

    @app.route("/players", methods=["GET", "POST"])
    def players():
        reg = Registry(registry)
        if request.method == "POST":
            act = request.form["act"]
            if act == "remove":
                reg.remove_sample(request.form["player"], int(request.form["det"]), "user:ui")
            elif act == "attr":
                reg.set_attr(request.form["player"], request.form.get("team", ""), request.form.get("number", ""), "user:ui")
            elif act == "delete_player":
                reg.delete_player(request.form["player"])
            reg.save()
            return redirect("/players")
        out = ["<h2>Registered players</h2>"]
        for pid, p in reg.players.items():
            probs = registration_problems(p, cfg["identity"]["reg_min_samples"], cfg["identity"]["reg_min_height"])
            out.append(f"<h3>{pid} <small>team={html.escape(str(p['team']))} number={html.escape(str(p['number']))}</small></h3>")
            out.append(("<p style='color:#c00'>Not usable for confirmation: " + "; ".join(probs) + "</p>") if probs else
                       "<p style='color:#080'>registration OK</p>")
            out.append(f"""<form method=post class=inline><input type=hidden name=act value=attr><input type=hidden name=player value={pid}>
team <input name=team size=8 value="{html.escape(p['team'] or '')}"> number <input name=number size=4 value="{html.escape(p['number'] or '')}"><button>save</button></form>
<form method=post class=inline onsubmit="return confirm('delete {pid}?')"><input type=hidden name=act value=delete_player><input type=hidden name=player value={pid}><button>delete player</button></form><br>""")
            for s in p["samples"]:
                q = s["quality"]
                out.append(f"""<span style="display:inline-block;text-align:center;margin:4px"><img class=thumb src="/crop/{s['det_id']}.jpg"><br>
<small>f{s['frame']} {s['view']} h{q['height_px']:.0f} occ{q['occlusion']}</small><br>
<form method=post class=inline><input type=hidden name=act value=remove><input type=hidden name=player value={pid}><input type=hidden name=det value={s['det_id']}><button>remove</button></form></span>""")
        return page("".join(out))

    @app.route("/review")
    def review():
        rows = [r for r in tracks() if r["observed"] == "1" and r["track_id"] != "-1"]
        segs = defaultdict(list)
        for r in rows:
            segs[(int(r["track_id"]), int(r["segment"]))].append(r)
        reg = Registry(registry)
        opts = "".join(f"<option>{p}</option>" for p in reg.players)
        only = request.args.get("filter", "")
        out = [f"<h2>Segments ({len(segs)})</h2><p>Filter: <a href='/review'>all</a> · <a href='/review?filter=UNCERTAIN'>uncertain</a> · "
               f"<a href='/review?filter=CONFIRMED'>confirmed</a></p>"
               "<p>Assign covers only the frames shown. Split a track at the frame where the box jumps to another person. "
               "After corrections press <b>Re-run identity</b>.</p>"
               "<form method=post action=/rerun><button>Re-run identity</button></form>"
               "<table><tr><th>track</th><th>seg</th><th>frames</th><th>status</th><th>player</th><th>score</th><th>margin</th><th>samples</th><th>correct</th></tr>"]
        for (t, s), rr in sorted(segs.items(), key=lambda kv: int(kv[1][0]["frame"])):
            st = rr[0]["status"]
            if only and st != only:
                continue
            f0, f1 = int(rr[0]["frame"]), int(rr[-1]["frame"])
            pick = [rr[k]["det_id"] for k in np.linspace(0, len(rr) - 1, min(4, len(rr))).astype(int)]
            thumbs = "".join(f"<img class=thumb src='/crop/{d}.jpg'>" for d in pick)
            out.append(f"""<tr><td>{t}</td><td>{s}</td><td>{f0}–{f1}<br><a href="/?f={f0}">view</a></td><td class=st-{st}>{st}</td>
<td>{rr[0]['player_id']}</td><td>{rr[0]['id_score']}</td><td>{rr[0]['id_margin']}</td><td>{thumbs}</td><td>
<form method=post action=/correct class=inline data-t0=0 onsubmit="this.secs.value=(Date.now()-window.T0)/1000">
<input type=hidden name=secs><input type=hidden name=track value={t}><input type=hidden name=a value={f0}><input type=hidden name=b value={f1}>
<select name=player>{opts}</select><button name=op value=assign>assign</button><button name=op value=unknown>unknown</button><br>
split at <input name=split size=5><button name=op value=split>split</button>
merge with track <input name=other size=4><button name=op value=merge>merge</button></form></td></tr>""")
        out.append("</table><script>window.T0=Date.now();</script>")
        ops = C.load(corr_path)
        if ops:
            spent = sum(o.get("seconds_spent", 0) for o in ops)
            out.append(f"<h3>Corrections ({len(ops)}, {spent:.0f}s recorded)</h3><pre>" +
                       html.escape(json.dumps(ops, indent=1)) + "</pre>")
        return page("".join(out))

    @app.route("/correct", methods=["POST"])
    def correct():
        fm = request.form
        op = fm["op"]; t = int(fm["track"]); a, b = int(fm["a"]), int(fm["b"])
        secs = float(fm.get("secs") or 0) or None
        if op == "assign":
            C.add(corr_path, {"op": "assign", "track_id": t, "frames": [a, b], "player_id": fm["player"]}, "user:ui", secs)
        elif op == "unknown":
            C.add(corr_path, {"op": "unknown", "track_id": t, "frames": [a, b]}, "user:ui", secs)
        elif op == "split" and fm.get("split"):
            C.add(corr_path, {"op": "split", "track_id": t, "frame": int(fm["split"])}, "user:ui", secs)
        elif op == "merge" and fm.get("other"):
            C.add(corr_path, {"op": "merge", "track_ids": [t, int(fm["other"])], "player_id": fm["player"]}, "user:ui", secs)
        return redirect("/review")

    @app.route("/rerun", methods=["POST"])
    def rerun():
        t0 = time.time()
        run_pipeline(pre_dir, registry, corr_path, run_dir, cfg, "system", pre=pre, quiet=True)
        state["tracks"] = None
        return redirect(f"/review?rerun_s={time.time() - t0:.1f}")

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--pre", required=True)
    ap.add_argument("--registry", default="match/players.json")
    ap.add_argument("--corrections", default="match/corrections.json")
    ap.add_argument("--run", default="runs/full/system")
    ap.add_argument("--config", default="configs/default.json")
    ap.add_argument("--port", type=int, default=5000)
    a = ap.parse_args()
    create_app(a.video, a.pre, a.registry, a.corrections, a.run, a.config).run("127.0.0.1", a.port)


if __name__ == "__main__":
    main()
