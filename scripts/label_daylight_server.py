#!/usr/bin/env python3
"""Loopback-only COCO bbox reviewer for the daylight PP-YOLOE+ set.

Run on the server and expose through an SSH tunnel only::

    python3 scripts/label_daylight_server.py --dataset runs/training/daylight_v1
    ssh -L 8765:127.0.0.1:8765 ppvehicle-server

The tool never uploads images and writes only the selected COCO annotation
file.  Detector suggestions are drafts; only an explicit Save becomes GT.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse


CATEGORIES = [
    {"id": 1, "name": "sepeda_motor"},
    {"id": 2, "name": "mobil_penumpang"},
    {"id": 3, "name": "kendaraan_sedang"},
    {"id": 4, "name": "bus_besar"},
    {"id": 5, "name": "truk_berat"},
    {"id": 6, "name": "pejalan_kaki"},
    {"id": 7, "name": "sepeda"},
]


HTML = r'''<!doctype html>
<meta charset="utf-8"><title>Daylight COCO reviewer</title>
<style>
body{font:14px system-ui;margin:0;background:#10151b;color:#e8eef4}header{padding:10px 16px;background:#18232e;display:flex;gap:10px;align-items:center;flex-wrap:wrap}button,select{font:14px;padding:6px;background:#253443;color:#fff;border:1px solid #526779;border-radius:4px}button:hover{background:#31516a}.ok{color:#64e6a1}.warn{color:#ffd166}.layout{display:flex;gap:14px;padding:14px}.stage{position:relative;width:min(75vw,960px);background:#000}.stage img{display:block;width:100%;height:auto}.stage canvas{position:absolute;inset:0;width:100%;height:100%;cursor:crosshair}.side{width:310px}.row{margin:8px 0}.hint{color:#9fb0bf;font-size:12px;line-height:1.4}.boxrow{padding:6px;margin:3px 0;background:#1b2834;border-left:4px solid #52b5ff;cursor:pointer}.boxrow.sel{background:#3b2d19;border-left-color:#ffd166}.boxrow.bad{border-left-color:#ff6b6b}.stat{padding:8px;background:#1b2834;margin:8px 0;line-height:1.5}.kbd{font-family:monospace;background:#2a3743;padding:1px 4px;border-radius:3px}
</style>
<header>
 <b>Daylight COCO reviewer</b>
 <label>split <select id="split"><option>val</option><option>train</option></select></label>
 <button id="prev">← Prev</button><button id="next">Next →</button>
 <button id="save">Save ground truth</button><span id="msg" class="warn">Draft belum disimpan</span>
</header>
<div class="layout"><div class="stage"><img id="image"><canvas id="canvas"></canvas></div><aside class="side">
 <div class="stat" id="stat"></div>
 <div class="row"><label>class <select id="cls"></select></label> <button id="del">Delete</button></div>
 <div class="hint">Klik bbox untuk memilih. Drag area kosong untuk menambah bbox. Pilih class lalu simpan. Bus/truck harus diberi kelas resmi secara manual; suggestion bukan ground truth.</div>
 <hr><div id="boxes"></div>
</aside></div>
<script>
const CATS=[['1','sepeda_motor'],['2','mobil_penumpang'],['3','kendaraan_sedang'],['4','bus_besar'],['5','truk_berat'],['6','pejalan_kaki'],['7','sepeda']];
const $=id=>document.getElementById(id), img=$('image'), cv=$('canvas'), ctx=cv.getContext('2d');
let split='val', images=[], index=0, boxes=[], selected=-1, suggestions=false, drag=null;
CATS.forEach(c=>{let o=document.createElement('option');o.value=c[0];o.textContent=c[1];$('cls').append(o)});
function q(url,opts){return fetch(url,opts).then(r=>{if(!r.ok)throw Error(r.status);return r.json()})}
async function loadList(){images=await q('/api/images?split='+split);index=Math.min(index,Math.max(0,images.length-1));await loadImage()}
async function loadImage(){if(!images.length)return;let im=images[index];img.src='/api/image?split='+split+'&name='+encodeURIComponent(im.file_name);await new Promise(r=>img.onload=r);cv.width=img.naturalWidth;cv.height=img.naturalHeight;let s=await q('/api/boxes?split='+split+'&id='+im.id);boxes=s.boxes;suggestions=s.suggestions;selected=-1;draw();render()}
function draw(){ctx.clearRect(0,0,cv.width,cv.height);boxes.forEach((b,i)=>{let [x,y,w,h]=b.bbox;ctx.strokeStyle=i===selected?'#ffd166':(b.suggestion?'#61dafb':'#67e8a9');ctx.lineWidth=i===selected?4:2;ctx.strokeRect(x,y,w,h);ctx.fillStyle='rgba(0,0,0,.7)';ctx.fillRect(x,y,Math.max(90,ctx.measureText(b.label||'').width+10),18);ctx.fillStyle=ctx.strokeStyle;ctx.font='12px system-ui';ctx.fillText((b.label||'unknown')+(b.score?' '+Math.round(b.score*100)+'%':''),x+4,y+13)})}
function render(){let im=images[index]||{};$('stat').innerHTML=`<b>${index+1}/${images.length}</b> · ${im.file_name||''}<br>boxes: ${boxes.length} · ${suggestions?'draft suggestions':'reviewed/edited'}<br><span class="hint">${suggestions?'Belum menjadi GT sampai Save ground truth ditekan.':''}</span>`;$('boxes').innerHTML=boxes.map((b,i)=>`<div class="boxrow ${i===selected?'sel':''} ${b.category_id?'':'bad'}" data-i="${i}">${i+1}. ${b.label||'UNMAPPED'} ${b.score?'('+Math.round(b.score*100)+'%)':''}<br><span class="hint">[${b.bbox.map(x=>Math.round(x)).join(', ')}]</span></div>`).join('');document.querySelectorAll('.boxrow').forEach(e=>e.onclick=()=>{selected=+e.dataset.i; if(boxes[selected].category_id)$('cls').value=boxes[selected].category_id;draw();render()})}
cv.onmousedown=e=>{let r=cv.getBoundingClientRect(),x=(e.clientX-r.left)*cv.width/r.width,y=(e.clientY-r.top)*cv.height/r.height;let hit=-1;boxes.forEach((b,i)=>{let [a,c,w,h]=b.bbox;if(x>=a&&x<=a+w&&y>=c&&y<=c+h)hit=i});if(hit>=0){selected=hit;if(boxes[hit].category_id)$('cls').value=boxes[hit].category_id;draw();render()}else drag={x,y}}
cv.onmouseup=e=>{if(!drag)return;let r=cv.getBoundingClientRect(),x=(e.clientX-r.left)*cv.width/r.width,y=(e.clientY-r.top)*cv.height/r.height;let a=Math.min(drag.x,x),b=Math.min(drag.y,y),w=Math.abs(x-drag.x),h=Math.abs(y-drag.y);if(w>4&&h>4){boxes.push({bbox:[a,b,w,h],category_id:+$('cls').value,label:$('cls').selectedOptions[0].textContent,reviewed:true});selected=boxes.length-1;suggestions=false;draw();render()}drag=null}
$('cls').onchange=()=>{if(selected>=0){boxes[selected].category_id=+$('cls').value;boxes[selected].label=$('cls').selectedOptions[0].textContent;boxes[selected].reviewed=true;suggestions=false;draw();render()}}
$('del').onclick=()=>{if(selected>=0){boxes.splice(selected,1);selected=-1;suggestions=false;draw();render()}}
$('save').onclick=async()=>{let im=images[index];let r=await q('/api/save',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({split,id:im.id,boxes})});suggestions=false;$('msg').textContent='Tersimpan sebagai ground truth';$('msg').className='ok';await loadImage()}
$('prev').onclick=()=>{if(index>0){index--;loadImage()}};$('next').onclick=()=>{if(index+1<images.length){index++;loadImage()}};$('split').onchange=()=>{split=$('split').value;index=0;loadList()};document.onkeydown=e=>{if(e.key==='ArrowRight')$('next').click();if(e.key==='ArrowLeft')$('prev').click();if(e.key==='Delete')$('del').click()};loadList();
</script>'''


def atomic_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


class App(BaseHTTPRequestHandler):
    dataset: Path
    suggestions: dict

    def send_json(self, data: object, status=HTTPStatus.OK):
        raw = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def body(self):
        n = int(self.headers.get("Content-Length", "0"))
        return json.loads(self.rfile.read(n))

    def payload(self, split):
        p = self.dataset / "annotations" / f"instances_{split}.json"
        return p, json.loads(p.read_text(encoding="utf-8"))

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if u.path == "/":
            raw = HTML.encode()
            self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8"); self.send_header("Content-Length", str(len(raw))); self.end_headers(); self.wfile.write(raw); return
        if u.path == "/api/images":
            split = q.get("split", ["val"])[0]
            _, d = self.payload(split)
            self.send_json(d.get("images", [])); return
        if u.path == "/api/image":
            split, name = q.get("split", ["val"])[0], unquote(q.get("name", [""])[0])
            # COCO metadata may store paths as images/<split>/<file>, while
            # the route root already points at images/<split>.
            name = Path(name).name
            root = (self.dataset / "images" / split).resolve(); path = (root / name).resolve()
            if root not in path.parents or not path.is_file(): self.send_error(404); return
            raw = path.read_bytes(); self.send_response(200); self.send_header("Content-Type", "image/jpeg"); self.send_header("Content-Length", str(len(raw))); self.end_headers(); self.wfile.write(raw); return
        if u.path == "/api/boxes":
            split, iid = q.get("split", ["val"])[0], int(q.get("id", ["0"])[0]); _, d = self.payload(split)
            anns = [a for a in d.get("annotations", []) if int(a.get("image_id", -1)) == iid]
            if anns:
                out = [{"bbox": a["bbox"], "category_id": int(a["category_id"]), "label": next((c["name"] for c in d.get("categories", CATEGORIES) if int(c["id"]) == int(a["category_id"])), "unknown"), "reviewed": True} for a in anns]
                self.send_json({"boxes": out, "suggestions": False}); return
            preds = [p for p in self.suggestions.get("predictions", []) if p.get("split") == split and int(p.get("image_id", -1)) == iid]
            self.send_json({"boxes": [{"bbox": p["bbox"], "category_id": p.get("category_id"), "label": p.get("detector_label", "unknown"), "score": p.get("score"), "suggestion": True} for p in preds], "suggestions": bool(preds)}); return
        self.send_error(404)

    def do_POST(self):
        if urlparse(self.path).path != "/api/save": self.send_error(404); return
        try:
            b = self.body(); split, iid = str(b["split"]), int(b["id"]); p, d = self.payload(split)
            if split not in {"train", "val"}: raise ValueError("invalid split")
            valid = []
            for x in b.get("boxes", []):
                cat = int(x["category_id"]); bx = [float(v) for v in x["bbox"]]
                if cat not in {c["id"] for c in CATEGORIES} or len(bx) != 4 or bx[2] <= 0 or bx[3] <= 0: raise ValueError("invalid bbox/category")
                valid.append({"id": 0, "image_id": iid, "category_id": cat, "bbox": [round(v, 2) for v in bx], "area": round(bx[2] * bx[3], 2), "iscrowd": 0, "review_status": "REVIEWED_GROUND_TRUTH"})
            d["categories"] = d.get("categories") or CATEGORIES
            d["annotations"] = [a for a in d.get("annotations", []) if int(a.get("image_id", -1)) != iid]
            next_id = max([int(a.get("id", 0)) for a in d["annotations"]] + [0]) + 1
            for a in valid: a["id"] = next_id; next_id += 1; d["annotations"].append(a)
            for im in d.get("images", []):
                if int(im.get("id", -1)) == iid: im["review_status"] = "REVIEWED"
            d["annotation_status"] = "GROUND_TRUTH_REVIEWED"
            atomic_json(p, d); self.send_json({"saved": True, "annotations": len(valid)})
        except Exception as e:
            self.send_json({"error": str(e)}, HTTPStatus.BAD_REQUEST)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", type=Path, required=True); ap.add_argument("--port", type=int, default=8765); args = ap.parse_args()
    App.dataset = args.dataset.resolve(); sug = App.dataset / "predictions_suggestions.json"; App.suggestions = json.loads(sug.read_text(encoding="utf-8")) if sug.exists() else {"predictions": []}
    print(f"reviewer listening on http://127.0.0.1:{args.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), App).serve_forever()


if __name__ == "__main__":
    main()
