"""
Lookzi Virtual Try-On
=====================
Start:  python app.py
UI:     http://127.0.0.1:7860
API:    http://127.0.0.1:7860/api/tryon  (POST)
Docs:   http://127.0.0.1:7860/docs
"""

from __future__ import annotations

import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import io
import gc
import os as _os_mod
import sys
import uuid
import base64
import random
import logging
import argparse
import traceback
import time
import json as _json
import signal
from logging.handlers import RotatingFileHandler
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("lookzi")

for pkg in ["torch", "gradio", "fastapi", "PIL", "fashn_vton"]:
    try:
        __import__(pkg if pkg != "PIL" else "PIL.Image")
    except ImportError:
        sys.exit(f"[ERROR] '{pkg}' not found. Run:  pip install -e .")

import torch
import gradio as gr
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse, PlainTextResponse
import uvicorn

ROOT    = Path(__file__).parent
WEIGHTS = ROOT / "weights"
OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(exist_ok=True)

# ── Server tracking & admin key ───────────────────────────────────────────
SERVER_START_TIME = time.time()
_log_file_path: str | None = None


def _setup_file_logging(log_path: str):
    """Replace stream handlers with a rotating file handler (no duplicate lines)."""
    global _log_file_path
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    for h in root.handlers[:]:
        if isinstance(h, logging.StreamHandler) and not isinstance(h, RotatingFileHandler):
            root.removeHandler(h)
    fh = RotatingFileHandler(
        log_path, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    fh.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")
    )
    root.addHandler(fh)
    _log_file_path = log_path
    logger.info("File logging: %s", log_path)


def _load_admin_key() -> str:
    """Load or auto-generate the admin panel secret key (saved in server_config.json)."""
    config_path = ROOT / "server_config.json"
    try:
        if config_path.exists():
            cfg = _json.loads(config_path.read_text(encoding="utf-8"))
            if key := cfg.get("admin_key"):
                return key
    except Exception:
        pass
    import secrets
    key = secrets.token_urlsafe(16)
    try:
        config_path.write_text(_json.dumps({"admin_key": key}, indent=2), encoding="utf-8")
    except Exception:
        pass
    return key


ADMIN_KEY: str = _load_admin_key()

# ── Category map: UI label → pipeline value ───────────────────────────────
CATEGORY_MAP = {
    "Upper":   "tops",
    "Lower":   "bottoms",
    "Overall": "one-pieces",
}

# ── Default inference settings ────────────────────────────────────────────
DEFAULT_STEPS     = 30
DEFAULT_GUIDANCE  = 1.5
DEFAULT_SEG_FREE  = True

# ── Pipeline singleton ────────────────────────────────────────────────────
_pipeline = None

def get_pipeline():
    global _pipeline
    if _pipeline is not None:
        return _pipeline
    if not (WEIGHTS / "model.safetensors").exists():
        raise RuntimeError("Model weights not found. Run: python scripts/download_weights.py --weights-dir ./weights")
    from fashn_vton import TryOnPipeline
    logger.info("Loading Lookzi model...")
    _pipeline = TryOnPipeline(weights_dir=str(WEIGHTS))
    logger.info("Model ready on: %s", _pipeline.device)
    return _pipeline


# ── Core inference ────────────────────────────────────────────────────────
def run_tryon(
    person_image:       Image.Image,
    garment_image:      Image.Image,
    category:           str,
    garment_photo_type: str   = "model",
    num_timesteps:      int   = DEFAULT_STEPS,
    guidance_scale:     float = DEFAULT_GUIDANCE,
    seed:               int   = -1,
    segmentation_free:  bool  = DEFAULT_SEG_FREE,
) -> tuple[Image.Image | None, str]:

    if person_image is None:
        return None, "Please upload a person photo."
    if garment_image is None:
        return None, "Please upload a garment image."

    api_category = CATEGORY_MAP.get(category, "tops")
    actual_seed  = random.randint(0, 2**31) if seed < 0 else int(seed)

    try:
        pipe = get_pipeline()
        t0   = time.time()
        output = pipe(
            person_image=person_image.convert("RGB"),
            garment_image=garment_image.convert("RGB"),
            category=api_category,
            garment_photo_type=garment_photo_type,
            num_timesteps=int(num_timesteps),
            guidance_scale=float(guidance_scale),
            segmentation_free=bool(segmentation_free),
            seed=actual_seed,
        )
        elapsed = time.time() - t0

        result = output.images[0]
        fname  = f"{uuid.uuid4().hex[:8]}_{api_category}.png"
        result.save(OUTPUTS / fname)
        logger.info("Done %.1fs -> %s", elapsed, fname)

        vram = ""
        if torch.cuda.is_available():
            used  = torch.cuda.memory_reserved(0) / 1e9
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            vram  = f" | VRAM {used:.1f}/{total:.0f} GB"

        return result, f"Done in {elapsed:.1f}s{vram}"

    except torch.cuda.OutOfMemoryError:
        gc.collect()
        torch.cuda.empty_cache()
        return None, "Out of memory. Try reducing Steps or restart."
    except Exception as e:
        logger.error(traceback.format_exc())
        return None, f"Error: {e}"


# ── FastAPI REST ──────────────────────────────────────────────────────────
api = FastAPI(
    title="Lookzi Virtual Try-On API",
    version="1.0.0",
    description="Lookzi — AI-powered Virtual Try-On. Local, fast, private.",
)


@api.get("/api/health")
def health():
    info: dict = {"status": "ok", "brand": "Lookzi", "model_loaded": _pipeline is not None}
    if torch.cuda.is_available():
        prop  = torch.cuda.get_device_properties(0)
        total = prop.total_memory / 1e9
        used  = torch.cuda.memory_reserved(0) / 1e9
        info["gpu"] = {"name": prop.name, "vram_total_gb": round(total, 1),
                       "vram_free_gb": round(total - used, 2)}
    return info


@api.post("/api/tryon")
async def api_tryon(
    person_image:       UploadFile = File(..., description="Person photo"),
    garment_image:      UploadFile = File(..., description="Garment image"),
    category:           str  = Form("Upper",  description="Upper | Lower | Overall"),
    garment_photo_type: str  = Form("model",  description="model | flat-lay"),
    return_base64:      bool = Form(False),
):
    """
    Lookzi Virtual Try-On endpoint.
    - **category**: `Upper` / `Lower` / `Overall`
    - **garment_photo_type**: `model` or `flat-lay`
    """
    try:
        person_pil  = Image.open(io.BytesIO(await person_image.read())).convert("RGB")
        garment_pil = Image.open(io.BytesIO(await garment_image.read())).convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"Invalid image: {e}")

    if category not in CATEGORY_MAP:
        raise HTTPException(422, f"category must be one of: {list(CATEGORY_MAP.keys())}")

    result, msg = run_tryon(person_pil, garment_pil, category, garment_photo_type)
    if result is None:
        raise HTTPException(500, msg)

    buf = io.BytesIO()
    result.save(buf, format="PNG")
    buf.seek(0)

    if return_base64:
        return JSONResponse({"status": "ok", "message": msg,
                             "image_base64": base64.b64encode(buf.getvalue()).decode()})
    return StreamingResponse(buf, media_type="image/png", headers={"X-Lookzi-Info": msg})


# ── Admin panel ───────────────────────────────────────────────────────────
_ADMIN_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Lookzi Admin</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
    background: #0a0a0a; color: #e0e0e0;
    font-family: 'Inter', 'Segoe UI', sans-serif;
    padding: 32px 24px; max-width: 960px; margin: 0 auto;
}
h1 {
    font-size: 1.8rem; font-weight: 800; margin-bottom: 6px;
    background: linear-gradient(135deg, #fff 0%, #a0a0ff 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
}
.sub { color: #333; font-size: 0.78rem; letter-spacing: 2px; text-transform: uppercase; margin-bottom: 32px; }
h2 { font-size: 0.72rem; color: #444; font-weight: 700; text-transform: uppercase;
     letter-spacing: 2px; margin: 28px 0 12px; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 10px; }
.stat { background: #0f0f0f; border: 1px solid #1e1e1e; border-radius: 12px; padding: 16px; }
.stat .lbl { font-size: 0.68rem; color: #444; text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 8px; }
.stat .val { font-size: 1.05rem; font-weight: 700; color: #fff; word-break: break-all; }
.val.g { color: #4ade80; } .val.y { color: #facc15; } .val.r { color: #f87171; }
.badge { display: inline-block; padding: 3px 11px; border-radius: 100px; font-size: 0.72rem; font-weight: 700; }
.badge.ok { background: #052e16; color: #4ade80; border: 1px solid #14532d; }
.badge.err { background: #2d0000; color: #f87171; border: 1px solid #7f1d1d; }
.card { background: #0f0f0f; border: 1px solid #1e1e1e; border-radius: 14px; padding: 20px; }
.btn { display: inline-block; padding: 11px 26px; border: none; border-radius: 10px;
       font-size: 0.88rem; font-weight: 700; cursor: pointer;
       transition: opacity 0.15s, transform 0.1s; letter-spacing: 0.5px; }
.btn:hover { opacity: 0.85; transform: translateY(-1px); }
.btn:active { transform: translateY(0); }
.btn-restart { background: linear-gradient(135deg, #dc2626, #b91c1c); color: #fff; }
.btn-sm { background: #1a1a1a; border: 1px solid #2a2a2a; color: #666;
          padding: 6px 14px; font-size: 0.72rem; margin-left: 8px; }
#logs-box {
    background: #060606; border: 1px solid #1a1a1a; border-radius: 10px;
    padding: 16px; font-family: 'Cascadia Code', 'Consolas', monospace;
    font-size: 0.75rem; line-height: 1.6; color: #6b7280;
    max-height: 460px; overflow-y: auto;
    white-space: pre-wrap; word-break: break-all; margin-top: 10px;
}
.timer { color: #2a2a2a; font-size: 0.72rem; margin-left: 14px; vertical-align: middle; }
#toast { position: fixed; bottom: 24px; right: 24px; padding: 12px 20px;
         border-radius: 10px; font-size: 0.85rem; display: none; z-index: 999; }
</style>
</head>
<body>
<h1>Lookzi Admin</h1>
<div class="sub">Server Management Panel</div>

<h2>System Status</h2>
<div class="grid">
  <div class="stat"><div class="lbl">Server</div><div class="val" id="s-status">...</div></div>
  <div class="stat"><div class="lbl">Uptime</div><div class="val" id="s-uptime">...</div></div>
  <div class="stat"><div class="lbl">Model</div><div class="val" id="s-model">...</div></div>
  <div class="stat"><div class="lbl">GPU</div><div class="val" id="s-gpu">...</div></div>
  <div class="stat"><div class="lbl">VRAM Used</div><div class="val" id="s-vram">...</div></div>
  <div class="stat"><div class="lbl">VRAM Free</div><div class="val" id="s-vram-free">...</div></div>
</div>

<h2>Actions</h2>
<div class="card" style="display:flex; align-items:center; gap:12px; flex-wrap:wrap;">
  <button class="btn btn-restart" onclick="doRestart()">&#x1F504; Restart Server</button>
  <button class="btn btn-sm" onclick="refreshAll()">&#x21BB; Refresh Now</button>
  <span class="timer" id="timer">auto-refresh in 15s</span>
</div>

<h2>Server Logs <button class="btn btn-sm" onclick="loadLogs()">&#x21BB;</button></h2>
<div id="logs-box">Loading...</div>

<div id="toast"></div>
<script>
const KEY = new URLSearchParams(location.search).get('key') || '';

function toast(msg, ok) {
    const t = document.getElementById('toast');
    t.textContent = msg;
    t.style.background = ok ? '#052e16' : '#2d0000';
    t.style.color = ok ? '#4ade80' : '#f87171';
    t.style.border = '1px solid ' + (ok ? '#14532d' : '#7f1d1d');
    t.style.display = 'block';
    setTimeout(() => t.style.display = 'none', 3500);
}

async function fetchStatus() {
    try {
        const [h, s] = await Promise.all([
            fetch('/api/health').then(r => r.json()),
            fetch('/admin/status?key=' + KEY).then(r => r.json())
        ]);
        document.getElementById('s-status').innerHTML = '<span class="badge ok">ONLINE</span>';
        document.getElementById('s-uptime').textContent = s.uptime_human || '-';
        const ml = h.model_loaded;
        document.getElementById('s-model').textContent = ml ? 'Loaded ✓' : 'Not loaded';
        document.getElementById('s-model').className = 'val ' + (ml ? 'g' : 'y');
        if (h.gpu) {
            document.getElementById('s-gpu').textContent = h.gpu.name.replace('NVIDIA GeForce ', '');
            const used = h.gpu.vram_total_gb - h.gpu.vram_free_gb;
            const pct = used / h.gpu.vram_total_gb;
            document.getElementById('s-vram').textContent = used.toFixed(1) + ' GB';
            document.getElementById('s-vram').className = 'val ' + (pct > 0.88 ? 'r' : 'g');
            document.getElementById('s-vram-free').textContent = h.gpu.vram_free_gb.toFixed(1) + ' GB';
            document.getElementById('s-vram-free').className = 'val ' + (h.gpu.vram_free_gb < 1 ? 'r' : 'g');
        }
    } catch(e) {
        document.getElementById('s-status').innerHTML = '<span class="badge err">OFFLINE</span>';
        ['s-uptime','s-model','s-gpu','s-vram','s-vram-free'].forEach(id => {
            document.getElementById(id).textContent = '-';
        });
    }
}

async function loadLogs() {
    try {
        const r = await fetch('/admin/logs?key=' + KEY);
        if (!r.ok) { document.getElementById('logs-box').textContent = '[Access denied]'; return; }
        const text = await r.text();
        const box = document.getElementById('logs-box');
        box.textContent = text || '(no logs yet)';
        box.scrollTop = box.scrollHeight;
    } catch(e) { document.getElementById('logs-box').textContent = 'Error: ' + e; }
}

async function doRestart() {
    if (!confirm('Restart the Lookzi server?\\n\\nThis will interrupt active requests.\\nThe wrapper script will bring it back online in ~20 seconds.')) return;
    try {
        const r = await fetch('/admin/restart?key=' + KEY, {method:'POST'});
        const d = await r.json();
        toast(d.message || 'Restarting...', true);
    } catch(e) { toast('Error: ' + e, false); }
}

function refreshAll() { fetchStatus(); loadLogs(); }

let cd = 15;
setInterval(() => {
    cd--;
    document.getElementById('timer').textContent = 'auto-refresh in ' + cd + 's';
    if (cd <= 0) { cd = 15; refreshAll(); }
}, 1000);

refreshAll();
</script>
</body>
</html>"""


@api.get("/admin")
async def admin_page(key: str = ""):
    if key != ADMIN_KEY:
        raise HTTPException(403, "Access denied. Add ?key=YOUR_ADMIN_KEY to the URL.")
    return HTMLResponse(_ADMIN_HTML)


@api.get("/admin/status")
async def admin_status(key: str = ""):
    if key != ADMIN_KEY:
        raise HTTPException(403, "Access denied")
    uptime = int(time.time() - SERVER_START_TIME)
    h, m, s = uptime // 3600, (uptime % 3600) // 60, uptime % 60
    return {
        "uptime_seconds": uptime,
        "uptime_human": f"{h}h {m}m {s}s",
        "model_loaded": _pipeline is not None,
    }


@api.get("/admin/logs")
async def admin_logs(key: str = "", lines: int = 100):
    if key != ADMIN_KEY:
        raise HTTPException(403, "Access denied")
    if _log_file_path and Path(_log_file_path).exists():
        with open(_log_file_path, encoding="utf-8", errors="replace") as f:
            all_lines = f.readlines()
        return PlainTextResponse("".join(all_lines[-lines:]))
    return PlainTextResponse("(No log file configured. Start server with --log-file logs/server.log)")


@api.post("/admin/restart")
async def admin_restart(key: str = ""):
    if key != ADMIN_KEY:
        raise HTTPException(403, "Access denied")
    import threading
    def _do_exit():
        time.sleep(0.8)
        _os_mod._exit(0)   # server_runner.bat loop will restart automatically
    threading.Thread(target=_do_exit, daemon=True).start()
    return {"status": "ok", "message": "Restarting in 1s — back online in ~20 seconds"}


# ── UI CSS ────────────────────────────────────────────────────────────────
CSS = """
/* ── Global ── */
* { box-sizing: border-box; }
body, .gradio-container {
    background: #0a0a0a !important;
    font-family: 'Inter', 'Segoe UI', sans-serif !important;
    color: #f0f0f0 !important;
}

/* ── Hide ALL Gradio branding / API / Settings ── */
footer,
.footer,
#footer,
.built-with,
.show-api,
.api-docs,
.api-recorder,
button[title="Settings"],
button[aria-label="Settings"],
.settings-button,
[data-testid="settings-button"],
.gradio-footer,
.svelte-byatnx,
a[href*="gradio.app"],
div.meta-text,
.meta-text-public,
.share-button,
#component-0 > .footer { display: none !important; }

.gradio-container { max-width: 1300px !important; margin: 0 auto !important; }

/* ── Header ── */
.lookzi-header {
    text-align: center;
    padding: 36px 0 20px;
}
.lookzi-header .logo {
    font-size: 3rem;
    font-weight: 800;
    letter-spacing: -2px;
    background: linear-gradient(135deg, #ffffff 0%, #a0a0ff 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.lookzi-header .tagline {
    color: #666;
    font-size: 0.9rem;
    margin-top: 4px;
    letter-spacing: 2px;
    text-transform: uppercase;
}

/* ── Upload panels ── */
.upload-panel .svelte-1ipelgc,
.upload-panel .wrap {
    border: 2px dashed #2a2a2a !important;
    border-radius: 16px !important;
    background: #111 !important;
    transition: border-color 0.2s;
}
.upload-panel .svelte-1ipelgc:hover { border-color: #5555ff !important; }
.result-panel .wrap {
    border: 2px solid #1e1e1e !important;
    border-radius: 16px !important;
    background: #0d0d0d !important;
}

/* ── Category buttons ── */
.category-row .gradio-radio { gap: 10px !important; }
.category-row label span {
    border-radius: 100px !important;
    padding: 8px 24px !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.5px !important;
    border: 2px solid #2a2a2a !important;
    background: #111 !important;
    color: #aaa !important;
    transition: all 0.15s !important;
    cursor: pointer;
}
.category-row label.selected span,
.category-row input:checked + span {
    border-color: #5555ff !important;
    background: #1a1a4a !important;
    color: #fff !important;
}

/* ── Try On button ── */
.tryon-btn button {
    background: linear-gradient(135deg, #4444ee, #7755ff) !important;
    border: none !important;
    border-radius: 14px !important;
    font-size: 1.05rem !important;
    font-weight: 700 !important;
    letter-spacing: 1px !important;
    color: #fff !important;
    height: 56px !important;
    transition: opacity 0.2s, transform 0.1s !important;
    text-transform: uppercase !important;
}
.tryon-btn button:hover { opacity: 0.9 !important; transform: translateY(-1px) !important; }
.tryon-btn button:active { transform: translateY(0) !important; }

/* ── Clear button ── */
.clear-btn button {
    background: #1a1a1a !important;
    border: 2px solid #2a2a2a !important;
    border-radius: 14px !important;
    color: #666 !important;
    height: 56px !important;
    font-weight: 600 !important;
}

/* ── Status bar ── */
.status-box textarea {
    background: #0d0d0d !important;
    border: 1px solid #1e1e1e !important;
    border-radius: 10px !important;
    color: #4ade80 !important;
    font-size: 0.82rem !important;
    font-family: monospace !important;
}

/* ── Section labels ── */
.section-label {
    color: #555;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 2px;
    padding: 16px 0 6px;
}

/* ── Examples gallery ── */
.examples-gallery .label-wrap { color: #555 !important; font-size: 0.8rem !important; }
.examples-gallery table td { background: #111 !important; border-radius: 8px !important; }
"""


# ── Gradio UI ─────────────────────────────────────────────────────────────
def build_ui() -> gr.Blocks:
    persons_dir  = ROOT / "examples" / "data" / "persons"
    garments_dir = ROOT / "examples" / "data" / "garments"

    person_imgs  = sorted(persons_dir.glob("*.[jJpPwW]*"))  if persons_dir.exists() else []
    upper_imgs   = sorted(garments_dir.glob("upper_*"))     if garments_dir.exists() else []
    lower_imgs   = sorted(garments_dir.glob("lower_*"))     if garments_dir.exists() else []
    overall_imgs = sorted(garments_dir.glob("overall_*"))   if garments_dir.exists() else []

    with gr.Blocks(title="Lookzi — Virtual Try-On",
                   analytics_enabled=False) as demo:

        # ── Header ────────────────────────────────────────────────────────
        gr.HTML("""
        <div class="lookzi-header">
            <div class="logo">Lookzi</div>
            <div class="tagline">AI Virtual Try-On &nbsp;·&nbsp; Try Before You Buy</div>
        </div>
        """)

        # ── Main panels ───────────────────────────────────────────────────
        with gr.Row(equal_height=True):
            with gr.Column(elem_classes="upload-panel"):
                gr.HTML('<div class="section-label">Person Photo</div>')
                person_img = gr.Image(
                    label="",
                    type="pil",
                    height=480,
                    show_label=False,
                )

            with gr.Column(elem_classes="upload-panel"):
                gr.HTML('<div class="section-label">Garment</div>')
                garment_img = gr.Image(
                    label="",
                    type="pil",
                    height=480,
                    show_label=False,
                )

            with gr.Column(elem_classes="result-panel"):
                gr.HTML('<div class="section-label">Result</div>')
                result_img = gr.Image(
                    label="",
                    type="pil",
                    height=480,
                    show_label=False,
                    interactive=False,
                )

        # ── Category + photo type ─────────────────────────────────────────
        with gr.Row(elem_classes="category-row"):
            with gr.Column(scale=3):
                category = gr.Radio(
                    choices=["Upper", "Lower", "Overall"],
                    value="Upper",
                    label="Garment Type",
                    info="Upper = shirts & jackets  ·  Lower = pants & skirts  ·  Overall = dresses & jumpsuits",
                )
            with gr.Column(scale=2):
                photo_type = gr.Radio(
                    choices=["model", "flat-lay"],
                    value="model",
                    label="Garment Photo",
                    info="model = worn by someone  ·  flat-lay = product shot",
                )

        # ── Advanced controls ─────────────────────────────────────────────
        with gr.Row():
            timesteps = gr.Slider(10, 50, value=DEFAULT_STEPS, step=1,
                                  label="Steps",
                                  info="More = slower but sharper")
            guidance  = gr.Slider(0.5, 7.0, value=DEFAULT_GUIDANCE, step=0.1,
                                  label="Guidance Scale")
            seed      = gr.Number(value=-1, precision=0,
                                  label="Seed  (-1 = random)")
            seg_free  = gr.Checkbox(value=DEFAULT_SEG_FREE,
                                    label="Segmentation-Free",
                                    info="Better for loose/baggy garments")

        # ── Action buttons ────────────────────────────────────────────────
        with gr.Row():
            run_btn   = gr.Button("Try On", variant="primary",   scale=5,
                                  size="lg", elem_classes="tryon-btn")
            clear_btn = gr.Button("Clear",  variant="secondary", scale=1,
                                  size="lg", elem_classes="clear-btn")

        status = gr.Textbox(
            label="", placeholder="Ready.", interactive=False,
            lines=1, max_lines=2, show_label=False, elem_classes="status-box",
        )

        # ── Example galleries ─────────────────────────────────────────────
        gr.HTML('<div class="section-label" style="padding-top:28px">Models</div>')
        if person_imgs:
            gr.Examples(
                examples=[[str(p)] for p in person_imgs],
                inputs=[person_img],
                label="",
                examples_per_page=10,
                elem_id="persons-gallery",
            )

        with gr.Row():
            with gr.Column():
                gr.HTML('<div class="section-label">Upper Garments</div>')
                if upper_imgs:
                    gr.Examples(
                        examples=[[str(g)] for g in upper_imgs],
                        inputs=[garment_img],
                        label="",
                        examples_per_page=10,
                    )
            with gr.Column():
                gr.HTML('<div class="section-label">Lower Garments</div>')
                if lower_imgs:
                    gr.Examples(
                        examples=[[str(g)] for g in lower_imgs],
                        inputs=[garment_img],
                        label="",
                        examples_per_page=10,
                    )
            with gr.Column():
                gr.HTML('<div class="section-label">Overall / One-Piece</div>')
                if overall_imgs:
                    gr.Examples(
                        examples=[[str(g)] for g in overall_imgs],
                        inputs=[garment_img],
                        label="",
                        examples_per_page=10,
                    )

        # ── Footer ────────────────────────────────────────────────────────
        gr.HTML("""
        <div style="text-align:center;padding:32px 0 16px;color:#333;font-size:0.78rem;
                    letter-spacing:1px;text-transform:uppercase;">
            Lookzi &nbsp;·&nbsp; Powered by AI &nbsp;·&nbsp; All processing on-device
        </div>
        """)

        # ── Logic ─────────────────────────────────────────────────────────
        def infer(person, garment, cat, ptype, steps, cfg, rng, sfree):
            img, msg = run_tryon(person, garment, cat, ptype,
                                 int(steps), float(cfg), int(rng), bool(sfree))
            return img, msg

        run_btn.click(
            fn=infer,
            inputs=[person_img, garment_img, category, photo_type,
                    timesteps, guidance, seed, seg_free],
            outputs=[result_img, status],
            api_name="tryon",
        )
        clear_btn.click(
            fn=lambda: (None, None, None, ""),
            outputs=[person_img, garment_img, result_img, status],
        )

    demo.css = CSS
    return demo


# ── Tunnel (ngrok) ────────────────────────────────────────────────────────
def _start_ngrok(port: int, authtoken: str | None = None, domain: str | None = None):
    """Start ngrok tunnel and print public URL."""
    try:
        from pyngrok import ngrok, conf as ngrok_conf
    except ImportError:
        logger.error("pyngrok not installed. Run:  pip install pyngrok")
        return

    if authtoken:
        ngrok_conf.get_default().auth_token = authtoken

    options = {}
    if domain:
        options["domain"] = domain

    try:
        tunnel = ngrok.connect(port, "http", **options)
        url    = tunnel.public_url
    except Exception as e:
        logger.error("ngrok tunnel failed: %s", e)
        logger.warning("Running in LOCAL mode only (fix ngrok token to enable public access)")
        return None

    logger.info("")
    logger.info("=" * 58)
    logger.info("  LOOKZI PUBLIC URL : %s", url)
    logger.info("  API DOCS          : %s/docs", url)
    logger.info("  (istalgan qurilmadan, istalgan joydan ochish mumkin)")
    logger.info("=" * 58)
    logger.info("")
    return url


# ── Entry point ───────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description="Lookzi Virtual Try-On Server")
    p.add_argument("--host",      default="127.0.0.1")
    p.add_argument("--port",      default=7860, type=int)
    p.add_argument("--share",     action="store_true",
                   help="ngrok tunnel orqali public URL yaratish")
    p.add_argument("--authtoken", default=None,
                   help="ngrok authtoken (ngrok.com dan olish mumkin)")
    p.add_argument("--domain",    default=None,
                   help="ngrok static domain (e.g. gap-tiring-omit.ngrok-free.dev)")
    p.add_argument("--preload",   action="store_true")
    p.add_argument("--log-file",  default=None,
                   help="Write logs to file (e.g. logs/server.log). Used by server_runner.bat.")
    args = p.parse_args()

    # File logging must be set up before anything else logs
    if args.log_file:
        _setup_file_logging(args.log_file)

    if torch.cuda.is_available():
        prop = torch.cuda.get_device_properties(0)
        logger.info("GPU: %s | VRAM: %.1f GB", prop.name, prop.total_memory / 1e9)

    if args.preload:
        get_pipeline()

    demo = build_ui()
    app  = gr.mount_gradio_app(api, demo, path="/")

    # --share: ngrok tunnel
    if args.share:
        _start_ngrok(args.port, args.authtoken, args.domain)

    logger.info("Lookzi local : http://%s:%d", args.host, args.port)
    logger.info("")
    logger.info("=" * 60)
    logger.info("  ADMIN PANEL  : http://%s:%d/admin?key=%s", args.host, args.port, ADMIN_KEY)
    if args.share and args.domain:
        logger.info("  ADMIN (ngrok): https://%s/admin?key=%s", args.domain, ADMIN_KEY)
    logger.info("=" * 60)
    logger.info("")

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning",
                timeout_keep_alive=600)


if __name__ == "__main__":
    main()
