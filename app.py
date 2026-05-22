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


def _free_port(port: int):
    """Kill any process holding the given port so we can bind cleanly."""
    import subprocess as _sp
    try:
        r = _sp.run(["netstat", "-ano"], capture_output=True, text=True, timeout=5)
        for line in r.stdout.splitlines():
            if f":{port}" in line and "LISTEN" in line:
                parts = line.strip().split()
                pid = int(parts[-1])
                if pid > 4:
                    _sp.run(["taskkill", "/F", "/PID", str(pid)],
                            capture_output=True, timeout=5)
                    logger.info("Freed port %d (killed PID %d)", port, pid)
    except Exception as e:
        logger.warning("Could not free port %d: %s", port, e)


def _kill_old_ngrok():
    """Kill leftover ngrok processes from previous sessions (prevents ERR_NGROK_108)."""
    import subprocess as _sp
    try:
        r = _sp.run(["taskkill", "/F", "/IM", "ngrok.exe", "/T"],
                    capture_output=True, text=True, timeout=5)
        if "SUCCESS" in r.stdout:
            logger.info("Killed leftover ngrok.exe processes")
    except Exception:
        pass


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
_sleeping  = True    # Start in sleep mode — wake manually from admin panel

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


def unload_pipeline():
    """Remove model from GPU memory (sleep mode)."""
    global _pipeline, _sleeping
    if _pipeline is not None:
        del _pipeline
        _pipeline = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Model unloaded — sleep mode (VRAM freed)")
    _sleeping = True


def wake_pipeline():
    """Reload model onto GPU (wake mode)."""
    global _sleeping
    _sleeping = False
    get_pipeline()   # loads if not already loaded
    logger.info("Model loaded — wake mode")
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

    if _sleeping or _pipeline is None:
        # Auto-wake on first request (model loads in ~15s)
        logger.info("Auto-wake: loading model on demand...")
        wake_pipeline()
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

# ── Test review router ────────────────────────────────────────────────────
try:
    from fastapi.staticfiles import StaticFiles
    from test_review import router as _test_router
    api.include_router(_test_router)
    # Assets rasmlarini to'g'ridan-to'g'ri serve qilish (review UI uchun)
    _assets_dir = ROOT / "Assets"
    if _assets_dir.exists():
        api.mount("/Assets", StaticFiles(directory=str(_assets_dir)), name="assets")
    logger.info("Test review UI: /tests?key=<admin_key>")
except Exception as _e:
    logger.warning("Test review yuklanmadi: %s", _e)


@api.get("/api/ping")
def api_ping():
    return PlainTextResponse("ok")


@api.get("/api/health")
def health():
    info: dict = {"status": "ok", "brand": "Lookzi",
                  "model_loaded": _pipeline is not None,
                  "sleeping": _sleeping}
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

def _render_deploy_waiting(key: str, git_out: str) -> str:
    import html as _html
    k   = _html.escape(key, quote=True)
    out = _html.escape(git_out)
    admin_url = f"/admin?key={k}&msg=Deploy+muvaffaqiyatli&ok=1"
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="refresh" content="55;url={admin_url}">
<title>Deploying...</title>
<style>
* {{ box-sizing:border-box; margin:0; padding:0; }}
body {{ background:#0a0a0a; color:#e0e0e0;
       font-family:'Inter','Segoe UI',sans-serif;
       display:flex; flex-direction:column; align-items:center;
       justify-content:center; min-height:100vh; padding:32px 24px; text-align:center; }}
h1 {{ font-size:2rem; font-weight:800; margin-bottom:8px; }}
.sub {{ color:#555; font-size:0.8rem; letter-spacing:2px; text-transform:uppercase; margin-bottom:32px; }}
.spinner {{ width:52px; height:52px; border:4px solid #1e1e1e;
            border-top-color:#4ade80; border-radius:50%;
            animation:spin 0.9s linear infinite; margin:0 auto 28px; }}
@keyframes spin {{ to {{ transform:rotate(360deg); }} }}
.git-box {{ background:#060606; border:1px solid #14532d; border-radius:10px;
            padding:16px 20px; font-family:'Consolas',monospace; font-size:0.78rem;
            color:#4ade80; max-width:640px; width:100%; text-align:left;
            white-space:pre-wrap; word-break:break-all; margin-bottom:28px; }}
.status {{ color:#555; font-size:0.88rem; margin-bottom:6px; }}
.ok  {{ color:#4ade80; font-weight:700; }}
.btn {{ display:none; margin-top:20px; padding:12px 32px; background:#16a34a;
        color:#fff; border-radius:10px; text-decoration:none;
        font-weight:700; font-size:0.9rem; }}
</style>
</head>
<body>
<h1>🚀 Deploying</h1>
<div class="sub">git pull muvaffaqiyatli</div>
<div class="spinner" id="sp"></div>
<div class="git-box">{out}</div>
<p class="status" id="st">Server qayta ishga tushmoqda...</p>
<p style="color:#2a2a2a;font-size:0.8rem;margin-top:6px" id="el">0s</p>
<a class="btn" id="btn" href="{admin_url}">Admin panelga o'tish &rarr;</a>

<script>
var KEY='{k}', secs=0, done=false;
function go(){{
  if(done)return; done=true;
  document.getElementById('sp').style.cssText='border-top-color:#4ade80;animation:none';
  document.getElementById('st').innerHTML='<span class="ok">Server tayyor! Yonaltirilmoqda...</span>';
  document.getElementById('btn').style.display='inline-block';
  setTimeout(function(){{ window.location.href='{admin_url}'; }},1500);
}}
var iv=setInterval(function(){{
  if(done)return;
  secs+=3;
  document.getElementById('el').textContent=secs+'s';
  if(secs>=10) document.getElementById('btn').style.display='inline-block';
  fetch('/api/ping',{{cache:'no-store',headers:{{'ngrok-skip-browser-warning':'1'}}}})
    .then(function(r){{ if(r.ok) go(); }})
    .catch(function(){{}});
}},3000);
</script>
</body>
</html>"""

_ADMIN_CSS = """
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
.sub { color: #555; font-size: 0.78rem; letter-spacing: 2px; text-transform: uppercase; margin-bottom: 32px; }
h2 { font-size: 0.72rem; color: #444; font-weight: 700; text-transform: uppercase;
     letter-spacing: 2px; margin: 28px 0 12px; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 10px; }
.stat { background: #0f0f0f; border: 1px solid #1e1e1e; border-radius: 12px; padding: 16px; }
.stat .lbl { font-size: 0.68rem; color: #444; text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 8px; }
.stat .val { font-size: 1.05rem; font-weight: 700; color: #fff; word-break: break-all; }
.val.g { color: #4ade80; } .val.y { color: #facc15; } .val.r { color: #f87171; }
.badge { display: inline-block; padding: 3px 11px; border-radius: 100px; font-size: 0.72rem; font-weight: 700; }
.badge.ok  { background: #052e16; color: #4ade80; border: 1px solid #14532d; }
.badge.err { background: #2d0000; color: #f87171; border: 1px solid #7f1d1d; }
.card { background: #0f0f0f; border: 1px solid #1e1e1e; border-radius: 14px; padding: 20px; }
.btn { display: inline-block; padding: 11px 26px; border: none; border-radius: 10px;
       font-size: 0.88rem; font-weight: 700; cursor: pointer; text-decoration: none;
       transition: opacity 0.15s, transform 0.1s; letter-spacing: 0.5px; }
.btn:hover { opacity: 0.85; transform: translateY(-1px); }
.btn-restart { background: linear-gradient(135deg, #dc2626, #b91c1c); color: #fff; }
.btn-deploy  { background: linear-gradient(135deg, #16a34a, #15803d); color: #fff; }
.btn-sleep   { background: linear-gradient(135deg, #d97706, #b45309); color: #fff; }
.btn-wake    { background: linear-gradient(135deg, #0891b2, #0e7490); color: #fff; }
.btn-sm { background: #1a1a1a; border: 1px solid #2a2a2a; color: #666;
          padding: 6px 14px; font-size: 0.72rem; }
.msg-ok  { margin-bottom:20px; padding:12px 16px; background:#052e16;
           border:1px solid #14532d; border-radius:10px; color:#4ade80; font-size:0.88rem; }
.msg-err { margin-bottom:20px; padding:12px 16px; background:#2d0000;
           border:1px solid #7f1d1d; border-radius:10px; color:#f87171; font-size:0.88rem; }
#logs-box {
    background: #060606; border: 1px solid #1a1a1a; border-radius: 10px;
    padding: 16px; font-family: 'Cascadia Code', 'Consolas', monospace;
    font-size: 0.75rem; line-height: 1.6; color: #6b7280;
    max-height: 460px; overflow-y: auto;
    white-space: pre-wrap; word-break: break-all; margin-top: 10px;
}
.timer-bar { color: #2a2a2a; font-size: 0.72rem; text-align: right; margin-bottom: 8px; }
"""


def _render_admin_page(key: str, msg: str = "", msg_ok: bool = True) -> str:
    import html as _html

    # ── gather status ──────────────────────────────────────────────────────
    uptime_s = int(time.time() - SERVER_START_TIME)
    uh = uptime_s // 3600
    um = (uptime_s % 3600) // 60
    us = uptime_s % 60
    uptime_human = f"{uh}h {um}m {us}s"

    model_loaded = _pipeline is not None
    sleeping     = _sleeping

    model_cls = "g" if model_loaded else "y"
    model_txt = "Loaded ✓" if model_loaded else "Not loaded"
    mode_cls  = "y" if sleeping else "g"
    mode_txt  = "Sleep 💤" if sleeping else "Active ⚡"

    gpu_cards = ""
    if torch.cuda.is_available():
        try:
            props      = torch.cuda.get_device_properties(0)
            vram_total = props.total_memory / 1024 ** 3
            vram_free  = torch.cuda.mem_get_info(0)[0] / 1024 ** 3
            vram_used  = vram_total - vram_free
            pct        = vram_used / vram_total
            gpu_name   = props.name.replace("NVIDIA GeForce ", "")
            vc = "r" if pct > 0.88 else "g"
            fc = "r" if vram_free < 1 else "g"
            gpu_cards = (
                f'<div class="stat"><div class="lbl">GPU</div>'
                f'<div class="val">{_html.escape(gpu_name)}</div></div>'
                f'<div class="stat"><div class="lbl">VRAM Used</div>'
                f'<div class="val {vc}">{vram_used:.1f} GB</div></div>'
                f'<div class="stat"><div class="lbl">VRAM Free</div>'
                f'<div class="val {fc}">{vram_free:.1f} GB</div></div>'
            )
        except Exception:
            gpu_cards = '<div class="stat"><div class="lbl">GPU</div><div class="val y">N/A</div></div>'

    # ── action buttons ─────────────────────────────────────────────────────
    k = _html.escape(key, quote=True)
    sleep_btn = (
        f'<a class="btn btn-sleep" href="/admin/action?a=sleep&key={k}" '
        f'onclick="return confirm(\'Modelni GPU dan tushirish?\\nVRAM bo\\\'shaydi.\')">💤 Sleep</a>'
        if (model_loaded and not sleeping) else ""
    )
    wake_btn = (
        f'<a class="btn btn-wake" href="/admin/action?a=wake&key={k}">⚡ Wake</a>'
        if sleeping else ""
    )

    # ── message banner ─────────────────────────────────────────────────────
    msg_html = ""
    if msg:
        cls = "msg-ok" if msg_ok else "msg-err"
        msg_html = f'<div class="{cls}">{_html.escape(msg)}</div>'

    # ── logs ───────────────────────────────────────────────────────────────
    logs_txt = "(No log file configured)"
    if _log_file_path and Path(_log_file_path).exists():
        try:
            with open(_log_file_path, encoding="utf-8", errors="replace") as f:
                logs_txt = "".join(f.readlines()[-80:])
        except Exception as exc:
            logs_txt = f"Error reading logs: {exc}"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="refresh" content="15;url=/admin?key={k}">
<title>Lookzi Admin</title>
<style>{_ADMIN_CSS}</style>
</head>
<body>
<h1>Lookzi Admin</h1>
<div class="sub">Server Management Panel</div>

{msg_html}

<div class="timer-bar" id="tmr">auto-refresh in 15s</div>

<h2>System Status</h2>
<div class="grid">
  <div class="stat"><div class="lbl">Server</div><div class="val"><span class="badge ok">ONLINE</span></div></div>
  <div class="stat"><div class="lbl">Uptime</div><div class="val">{uptime_human}</div></div>
  <div class="stat"><div class="lbl">Model</div><div class="val {model_cls}">{model_txt}</div></div>
  <div class="stat"><div class="lbl">Mode</div><div class="val {mode_cls}">{mode_txt}</div></div>
  {gpu_cards}
</div>

<h2>Actions</h2>
<div class="card">
  <div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap;">
    <a class="btn btn-deploy" href="/admin/action?a=deploy&key={k}"
       onclick="return confirm('Deploy latest code?\\n\\ngit pull + server restart (~40s)')">🚀 Deploy</a>
    <a class="btn btn-restart" href="/admin/action?a=restart&key={k}"
       onclick="return confirm('Server restart?\\n~40 soniya offline bo\\'ladi.')">🔄 Restart</a>
    {sleep_btn}
    {wake_btn}
    <a class="btn btn-sm" href="/admin?key={k}">↻ Refresh</a>
  </div>
</div>

<h2>Server Logs</h2>
<div id="logs-box">{_html.escape(logs_txt)}</div>

<script>
let cd = 15;
setInterval(() => {{
  cd--;
  const t = document.getElementById('tmr');
  if (t) t.textContent = 'auto-refresh in ' + cd + 's';
}}, 1000);
const lb = document.getElementById('logs-box');
if (lb) lb.scrollTop = lb.scrollHeight;
</script>
</body>
</html>"""


@api.get("/admin")
async def admin_page(key: str = "", msg: str = "", ok: int = 1):
    if key != ADMIN_KEY:
        raise HTTPException(403, "Access denied. Add ?key=YOUR_ADMIN_KEY to the URL.")
    html_content = _render_admin_page(key, msg=msg, msg_ok=bool(ok))
    return HTMLResponse(
        html_content,
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate",
            "Pragma": "no-cache",
            "ngrok-skip-browser-warning": "1",
        },
    )


@api.get("/admin/action")
async def admin_action(a: str = "", key: str = ""):
    if key != ADMIN_KEY:
        raise HTTPException(403, "Access denied")
    from fastapi.responses import RedirectResponse
    from urllib.parse import quote

    def redir(message: str, success: bool = True) -> RedirectResponse:
        return RedirectResponse(
            f"/admin?key={key}&msg={quote(message)}&ok={1 if success else 0}",
            status_code=303,
        )

    if a == "restart":
        import threading
        def _do_exit():
            time.sleep(0.8)
            _os_mod._exit(0)
        threading.Thread(target=_do_exit, daemon=True).start()
        return redir("Restarting… Back online in ~40 seconds.")

    if a == "sleep":
        if not _pipeline:
            return redir("Model allaqachon yuklanmagan.")
        import threading
        threading.Thread(target=unload_pipeline, daemon=True).start()
        return redir("Server uxlayapti 💤 — VRAM bo'shaydi (~3s).")

    if a == "wake":
        if _pipeline:
            return redir("Model already loaded ⚡")
        import threading
        threading.Thread(target=wake_pipeline, daemon=True).start()
        return redir("Model loading… (~15s) ⚡")

    if a == "deploy":
        import subprocess, threading
        # Run git pull RIGHT NOW (synchronously) so we can show the result
        try:
            pull = subprocess.run(
                ["git", "-c", "safe.directory=*", "pull", "--ff-only", "lookzi", "main"],
                cwd=str(ROOT), capture_output=True, text=True, timeout=60,
                env={**_os_mod.environ, "GIT_TERMINAL_PROMPT": "0"},
            )
            git_out = (pull.stdout + pull.stderr).strip() or "(no output)"
            logger.info("Deploy git pull (code %d): %s", pull.returncode, git_out)
        except Exception as exc:
            logger.error("Deploy git pull exception: %s", exc)
            return redir(f"❌ git pull xato: {exc}", success=False)

        if pull.returncode != 0:
            # Pull failed — don't restart, just show error
            return redir(f"❌ git pull muvaffaqiyatsiz: {git_out}", success=False)

        # Already up to date — no restart needed
        if "Already up to date" in git_out:
            return redir("✅ Kod allaqachon yangi (Already up to date). Restart kerak emas.", success=True)

        # New code pulled — schedule restart and show waiting page
        threading.Thread(target=lambda: (time.sleep(0.6), _os_mod._exit(0)), daemon=True).start()
        return HTMLResponse(
            _render_deploy_waiting(key, git_out),
            headers={"Cache-Control": "no-store", "ngrok-skip-browser-warning": "1"},
        )

    raise HTTPException(400, f"Unknown action: {a}")


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
        _os_mod._exit(0)   # NSSM restarts automatically
    threading.Thread(target=_do_exit, daemon=True).start()
    return {"status": "ok", "message": "Restarting in 1s — back online in ~40 seconds"}


@api.post("/admin/sleep")
async def admin_sleep(key: str = ""):
    if key != ADMIN_KEY:
        raise HTTPException(403, "Access denied")
    if not _pipeline:
        return {"status": "ok", "message": "Model allaqachon yuklanmagan"}
    unload_pipeline()
    vram = ""
    if torch.cuda.is_available():
        free = torch.cuda.get_device_properties(0).total_memory / 1e9
        vram = f" | {free:.0f} GB VRAM bo'shadi"
    return {"status": "ok", "message": f"Server uxlayapti 💤{vram}"}


@api.post("/admin/wake")
async def admin_wake(key: str = ""):
    if key != ADMIN_KEY:
        raise HTTPException(403, "Access denied")
    if _pipeline:
        return {"status": "ok", "message": "Model allaqachon yuklangan ⚡"}
    import threading
    def _load():
        wake_pipeline()
    threading.Thread(target=_load, daemon=True).start()
    return {"status": "ok", "message": "Model yuklanmoqda (~15s)... ⚡"}


@api.post("/admin/deploy")
async def admin_deploy(key: str = ""):
    if key != ADMIN_KEY:
        raise HTTPException(403, "Access denied")
    import subprocess, threading
    def _do_deploy():
        time.sleep(0.3)
        try:
            logger.info("Deploy: running git pull...")
            result = subprocess.run(
                ["git", "-c", "safe.directory=*", "pull", "--ff-only", "lookzi", "main"],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
                timeout=60,
                env={**_os_mod.environ, "GIT_TERMINAL_PROMPT": "0"},
            )
            out = (result.stdout + result.stderr).strip()
            logger.info("Deploy git pull: %s", out)
            if result.returncode == 0:
                logger.info("Deploy: OK — restarting service...")
                time.sleep(1)
                _os_mod._exit(0)   # NSSM restarts with new code
            else:
                logger.error("Deploy: git pull failed (code %d): %s", result.returncode, out)
        except Exception as e:
            logger.error("Deploy error: %s", e)
    threading.Thread(target=_do_deploy, daemon=True).start()
    return {
        "status": "ok",
        "message": "git pull bajarilmoqda → server qayta ishga tushadi (~40s)"
    }


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

    # Kill leftover ngrok + free port before starting
    _kill_old_ngrok()
    _free_port(args.port)

    if torch.cuda.is_available():
        prop = torch.cuda.get_device_properties(0)
        logger.info("GPU: %s | VRAM: %.1f GB", prop.name, prop.total_memory / 1e9)

    if args.preload:
        wake_pipeline()   # --preload flag explicitly requests model load

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
