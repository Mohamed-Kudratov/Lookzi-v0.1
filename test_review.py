"""
Lookzi Test Review UI
=====================
Test natijalarini ko'rish va baholash uchun web interfeys.

Routes:
    GET  /tests                     — barcha sessionlar ro'yxati
    GET  /tests/{sid}               — session review sahifasi
    GET  /tests/{sid}/image/{name}  — natija rasmini berish
    POST /tests/{sid}/rate          — bahoni saqlash
    GET  /tests/{sid}/report        — tahlil JSON
    GET  /tests/{sid}/report.html   — tahlil HTML sahifasi
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import (
    FileResponse, HTMLResponse, JSONResponse, PlainTextResponse
)
from pydantic import BaseModel

ROOT     = Path(__file__).parent
TEST_DIR = ROOT / "test_results"

router = APIRouter()


# ── Yordamchi funksiyalar ──────────────────────────────────────────────────

def _load_meta(sid: str) -> dict:
    path = TEST_DIR / sid / "metadata.json"
    if not path.exists():
        raise HTTPException(404, f"Session '{sid}' topilmadi")
    return json.loads(path.read_text(encoding="utf-8"))


def _save_meta(sid: str, meta: dict):
    path = TEST_DIR / sid / "metadata.json"
    path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


def _check_key(request: Request, admin_key: str) -> bool:
    key = request.query_params.get("key", "")
    return key == admin_key


ISSUE_LABELS = {
    "wrong_fit":    "Kiyim noto'g'ri",
    "mask_error":   "Mask xato",
    "artifact":     "Artefakt/buzilish",
    "color_shift":  "Rang o'zgardi",
    "body_distort": "Tana buzildi",
    "bg_noise":     "Fon buzildi",
    "garment_messy":"Kiyim rasmi sifatsiz",
    "blurry":       "Loyqa/xiralashgan",
}

RATING_LABELS = {"good": "✅ Yaxshi", "mid": "⚠️ O'rta", "bad": "❌ Yomon"}


# ── Endpoints ──────────────────────────────────────────────────────────────

@router.get("/tests", response_class=HTMLResponse)
async def sessions_list(request: Request):
    """Barcha test sessionlar ro'yxati."""
    from app import ADMIN_KEY  # circular import avoid: import inside function
    if not _check_key(request, ADMIN_KEY):
        return HTMLResponse("<h2>403 — Admin key kerak: /tests?key=...</h2>", 403)

    sessions = []
    if TEST_DIR.exists():
        for d in sorted(TEST_DIR.iterdir(), reverse=True):
            mf = d / "metadata.json"
            if mf.exists():
                try:
                    m = json.loads(mf.read_text(encoding="utf-8"))
                    rated   = sum(1 for t in m.get("tests", []) if t.get("rating"))
                    total   = m.get("total", 0)
                    ok      = m.get("ok", 0)
                    sessions.append({
                        "id":      m["session_id"],
                        "date":    m.get("created_at", "")[:19].replace("T", " "),
                        "mode":    m.get("mode", "?"),
                        "total":   total,
                        "ok":      ok,
                        "rated":   rated,
                    })
                except Exception:
                    pass

    rows = ""
    for s in sessions:
        pct   = round(s["rated"] / s["total"] * 100) if s["total"] else 0
        key   = request.query_params.get("key", "")
        rows += f"""
        <tr>
          <td><a href="/tests/{s['id']}?key={key}">{s['id']}</a></td>
          <td>{s['date']}</td>
          <td>{s['mode']}</td>
          <td>{s['ok']} / {s['total']}</td>
          <td>{s['rated']} / {s['total']} ({pct}%)</td>
          <td>
            <a class="btn" href="/tests/{s['id']}?key={key}">Ko'rish</a>
            <a class="btn sec" href="/tests/{s['id']}/report.html?key={key}">Tahlil</a>
          </td>
        </tr>"""

    return HTMLResponse(_page("Test Sessionlar", f"""
    <h1>Test Sessionlar</h1>
    <table>
      <thead><tr>
        <th>Session</th><th>Sana</th><th>Rejim</th>
        <th>Natija</th><th>Baholangan</th><th>Amal</th>
      </tr></thead>
      <tbody>{rows or '<tr><td colspan="6" style="text-align:center;color:#555">Hali session yo\'q</td></tr>'}</tbody>
    </table>
    """))


@router.get("/tests/{sid}", response_class=HTMLResponse)
async def session_review(sid: str, request: Request,
                         filter: str = "all", idx: int = 0):
    """Session review sahifasi."""
    from app import ADMIN_KEY
    if not _check_key(request, ADMIN_KEY):
        return HTMLResponse("<h2>403 — Admin key kerak</h2>", 403)

    meta  = _load_meta(sid)
    tests = meta.get("tests", [])
    key   = request.query_params.get("key", "")

    # Filter
    if filter == "unrated":
        view = [t for t in tests if not t.get("rating") and t["status"] == "ok"]
    elif filter == "good":
        view = [t for t in tests if t.get("rating") == "good"]
    elif filter == "mid":
        view = [t for t in tests if t.get("rating") == "mid"]
    elif filter == "bad":
        view = [t for t in tests if t.get("rating") == "bad"]
    elif filter == "error":
        view = [t for t in tests if t["status"] == "error"]
    else:
        view = [t for t in tests if t["status"] == "ok"]

    total_ok     = sum(1 for t in tests if t["status"] == "ok")
    total_rated  = sum(1 for t in tests if t.get("rating"))
    idx          = max(0, min(idx, len(view) - 1))
    current      = view[idx] if view else None

    # Progress bar
    pct = round(total_rated / total_ok * 100) if total_ok else 0

    nav_html = f"""
    <div class="nav-bar">
      <a class="btn{'  active' if filter=='all'     else ''}" href="?key={key}&filter=all">
        Hammasi ({total_ok})</a>
      <a class="btn{'  active' if filter=='unrated' else ''}" href="?key={key}&filter=unrated">
        Baholanmagan ({total_ok - total_rated})</a>
      <a class="btn{'  active' if filter=='good'    else ''}" href="?key={key}&filter=good">
        ✅ Yaxshi</a>
      <a class="btn{'  active' if filter=='mid'     else ''}" href="?key={key}&filter=mid">
        ⚠️ O'rta</a>
      <a class="btn{'  active' if filter=='bad'     else ''}" href="?key={key}&filter=bad">
        ❌ Yomon</a>
      <a class="btn{'  active' if filter=='error'   else ''}" href="?key={key}&filter=error">
        Xato ({sum(1 for t in tests if t['status']=='error')})</a>
      <span class="sep"></span>
      <a class="btn sec" href="/tests/{sid}/report.html?key={key}">📊 Tahlil</a>
    </div>
    <div class="progress-bar">
      <div class="progress-fill" style="width:{pct}%"></div>
      <span class="progress-label">Baholangan: {total_rated}/{total_ok} ({pct}%)</span>
    </div>
    """

    if not current:
        return HTMLResponse(_page(f"Session {sid}", nav_html + """
        <div style="text-align:center;padding:80px;color:#555;font-size:1.1rem">
          Bu filtrdagi natijalar yo'q
        </div>"""))

    # Issue checkboxes
    issue_html = "".join(
        f"""<label class="issue-tag{'  sel' if k in (current.get('issues') or []) else ''}">
          <input type="checkbox" name="issues" value="{k}"
            {'checked' if k in (current.get('issues') or []) else ''}> {v}
        </label>"""
        for k, v in ISSUE_LABELS.items()
    )

    prev_url = f"?key={key}&filter={filter}&idx={idx-1}" if idx > 0 else "#"
    next_url = f"?key={key}&filter={filter}&idx={idx+1}" if idx < len(view)-1 else "#"

    # Rating buttons
    def rb(val, label):
        active = "active" if current.get("rating") == val else ""
        return f'<button class="rate-btn {val} {active}" data-rating="{val}">{label}</button>'

    panels_html = ""
    if current.get("result_path"):
        res_url  = f"/tests/{sid}/image/{Path(current['result_path']).name}?key={key}"
        pers_url = "/" + current["person_path"].replace("\\", "/")
        garm_url = "/" + current["garment_path"].replace("\\", "/")
        panels_html = f"""
        <div class="panels">
          <div class="panel"><p class="panel-label">👤 Model</p>
            <img src="{pers_url}"></div>
          <div class="panel"><p class="panel-label">👕 Kiyim</p>
            <img src="{garm_url}"></div>
          <div class="panel main-panel"><p class="panel-label">✨ Natija</p>
            <img src="{res_url}"></div>
        </div>"""
    else:
        panels_html = f"""<div class="error-box">
          ❌ Xato: {current.get('error','?')}</div>"""

    review_html = f"""
    {nav_html}
    <div class="review-wrap">
      <div class="test-meta">
        <span class="badge">{current['gender']}</span>
        <span class="badge">{current['category']}</span>
        <span class="badge sec">{current.get('elapsed_s','?')}s</span>
        <span style="color:#555;font-size:.75rem;margin-left:auto">
          {idx+1} / {len(view)}
        </span>
      </div>

      {panels_html}

      <div class="rate-section" id="rateForm">
        <div class="rate-row">
          {rb('good','✅ Yaxshi')}
          {rb('mid', '⚠️ O\'rta')}
          {rb('bad', '❌ Yomon')}
        </div>
        <div class="issues-row">{issue_html}</div>
        <textarea id="noteBox" placeholder="Izoh (ixtiyoriy)..."
          rows="2">{current.get('note','')}</textarea>
        <div class="save-row">
          <button class="btn-save" id="saveBtn" onclick="saveRating()">💾 Saqlash</button>
          <span id="saveStatus"></span>
        </div>
      </div>

      <div class="nav-arrows">
        <a class="arrow{'  disabled' if idx==0 else ''}" href="{prev_url}"
          id="prevBtn">← Oldingi</a>
        <a class="arrow{'  disabled' if idx==len(view)-1 else ''}" href="{next_url}"
          id="nextBtn">Keyingi →</a>
      </div>
    </div>

    <script>
    const SID = "{sid}", TID = "{current['id']}", KEY = "{key}";
    const FILTER = "{filter}", IDX = {idx};

    let currentRating = "{current.get('rating') or ''}";

    document.querySelectorAll('.rate-btn').forEach(btn => {{
      btn.addEventListener('click', () => {{
        document.querySelectorAll('.rate-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        currentRating = btn.dataset.rating;
      }});
    }});

    async function saveRating() {{
      const issues = [...document.querySelectorAll('input[name=issues]:checked')]
                      .map(x => x.value);
      const note   = document.getElementById('noteBox').value;
      const st     = document.getElementById('saveStatus');
      st.textContent = 'Saqlanmoqda...';
      try {{
        const r = await fetch(`/tests/${{SID}}/rate?key=${{KEY}}`, {{
          method: 'POST',
          headers: {{'Content-Type': 'application/json'}},
          body: JSON.stringify({{
            test_id: TID, rating: currentRating, issues, note
          }})
        }});
        const d = await r.json();
        if (d.ok) {{
          st.textContent = '✓ Saqlandi';
          st.style.color = '#4ade80';
          setTimeout(() => {{
            const nxt = document.getElementById('nextBtn');
            if (!nxt.classList.contains('disabled')) nxt.click();
          }}, 600);
        }} else {{
          st.textContent = '✗ Xato: ' + d.detail;
          st.style.color = '#f87171';
        }}
      }} catch(e) {{
        st.textContent = '✗ ' + e; st.style.color = '#f87171';
      }}
    }}

    // Keyboard shortcuts
    document.addEventListener('keydown', e => {{
      if (e.target.tagName === 'TEXTAREA') return;
      if (e.key === '1') document.querySelector('.rate-btn.good')?.click();
      if (e.key === '2') document.querySelector('.rate-btn.mid')?.click();
      if (e.key === '3') document.querySelector('.rate-btn.bad')?.click();
      if (e.key === 'Enter') saveRating();
      if (e.key === 'ArrowLeft') {{
        const p = document.getElementById('prevBtn');
        if (!p.classList.contains('disabled')) p.click();
      }}
      if (e.key === 'ArrowRight') {{
        const n = document.getElementById('nextBtn');
        if (!n.classList.contains('disabled')) n.click();
      }}
    }});
    </script>
    """

    return HTMLResponse(_page(f"Review — {sid}", review_html))


@router.get("/tests/{sid}/image/{name}")
async def serve_result_image(sid: str, name: str, request: Request):
    from app import ADMIN_KEY
    if not _check_key(request, ADMIN_KEY):
        raise HTTPException(403)
    path = TEST_DIR / sid / name
    if not path.exists():
        raise HTTPException(404)
    return FileResponse(path)


class RatePayload(BaseModel):
    test_id: str
    rating:  str         # good | mid | bad | ""
    issues:  list[str] = []
    note:    str = ""


@router.post("/tests/{sid}/rate")
async def save_rating(sid: str, payload: RatePayload, request: Request):
    from app import ADMIN_KEY
    if not _check_key(request, ADMIN_KEY):
        raise HTTPException(403)

    if payload.rating and payload.rating not in ("good", "mid", "bad"):
        raise HTTPException(422, "rating: good | mid | bad | ''")

    meta  = _load_meta(sid)
    found = False
    for t in meta["tests"]:
        if t["id"] == payload.test_id:
            t["rating"] = payload.rating or None
            t["issues"] = payload.issues
            t["note"]   = payload.note
            found = True
            break

    if not found:
        raise HTTPException(404, f"test_id '{payload.test_id}' topilmadi")

    _save_meta(sid, meta)
    return {"ok": True}


@router.get("/tests/{sid}/report")
async def report_json(sid: str, request: Request):
    from app import ADMIN_KEY
    if not _check_key(request, ADMIN_KEY):
        raise HTTPException(403)
    return JSONResponse(_build_report(_load_meta(sid)))


@router.get("/tests/{sid}/report.html", response_class=HTMLResponse)
async def report_html(sid: str, request: Request):
    from app import ADMIN_KEY
    if not _check_key(request, ADMIN_KEY):
        return HTMLResponse("<h2>403</h2>", 403)

    r   = _build_report(_load_meta(sid))
    key = request.query_params.get("key", "")

    def pct(a, b):
        return f"{round(a/b*100)}%" if b else "—"

    def stat_card(label, val, color=""):
        return f'<div class="stat"><div class="lbl">{label}</div><div class="val {color}">{val}</div></div>'

    # Per category rows
    cat_rows = ""
    for cat, s in r["by_category"].items():
        total = s["total"]
        good  = s.get("good", 0)
        mid   = s.get("mid",  0)
        bad   = s.get("bad",  0)
        cat_rows += f"""<tr>
          <td>{cat}</td><td>{total}</td>
          <td>{good} ({pct(good,total)})</td>
          <td>{mid}  ({pct(mid,total)})</td>
          <td>{bad}  ({pct(bad,total)})</td>
        </tr>"""

    gender_rows = ""
    for gen, s in r["by_gender"].items():
        total = s["total"]
        good  = s.get("good", 0)
        gender_rows += f"""<tr>
          <td>{gen}</td><td>{total}</td>
          <td>{good} ({pct(good,total)})</td>
        </tr>"""

    issue_rows = ""
    for issue, cnt in sorted(r["issue_counts"].items(), key=lambda x: -x[1]):
        label = ISSUE_LABELS.get(issue, issue)
        issue_rows += f"<tr><td>{label}</td><td>{cnt}</td></tr>"

    body = f"""
    <div style="display:flex;align-items:center;gap:16px;margin-bottom:32px">
      <h1>📊 Tahlil — {sid}</h1>
      <a class="btn sec" href="/tests/{sid}?key={key}">← Review ga qaytish</a>
    </div>

    <h2>Umumiy ko'rsatkichlar</h2>
    <div class="grid">
      {stat_card("Jami testlar",   r['total'])}
      {stat_card("Bajarildi",      r['ok'],    'g')}
      {stat_card("Xato",           r['error'], 'r' if r['error'] else '')}
      {stat_card("Baholangan",     r['rated'])}
      {stat_card("✅ Yaxshi",      r['good'],  'g')}
      {stat_card("⚠️ O'rta",       r['mid'],   'y')}
      {stat_card("❌ Yomon",       r['bad'],   'r')}
      {stat_card("O'rtacha vaqt",  f"{r['avg_elapsed_s']}s")}
    </div>

    <h2>Kategoriya bo'yicha</h2>
    <table>
      <thead><tr><th>Kategoriya</th><th>Jami</th>
        <th>✅ Yaxshi</th><th>⚠️ O'rta</th><th>❌ Yomon</th></tr></thead>
      <tbody>{cat_rows or '<tr><td colspan="5" style="color:#555">Ma\'lumot yo\'q</td></tr>'}</tbody>
    </table>

    <h2>Jins bo'yicha</h2>
    <table>
      <thead><tr><th>Jins</th><th>Jami</th><th>✅ Yaxshi</th></tr></thead>
      <tbody>{gender_rows or '<tr><td colspan="3" style="color:#555">Ma\'lumot yo\'q</td></tr>'}</tbody>
    </table>

    <h2>Muammo turlari</h2>
    <table>
      <thead><tr><th>Muammo</th><th>Soni</th></tr></thead>
      <tbody>{issue_rows or '<tr><td colspan="2" style="color:#555">Hali muammo belgilanmagan</td></tr>'}</tbody>
    </table>
    """

    return HTMLResponse(_page(f"Tahlil — {sid}", body))


def _build_report(meta: dict) -> dict:
    tests = meta.get("tests", [])
    ok    = [t for t in tests if t["status"] == "ok"]
    rated = [t for t in ok    if t.get("rating")]

    by_cat    = {}
    by_gender = {}
    issues    = {}

    for t in rated:
        cat = t["category"]
        gen = t["gender"]
        rat = t["rating"]

        by_cat.setdefault(cat, {"total": 0})
        by_cat[cat]["total"] += 1
        by_cat[cat][rat] = by_cat[cat].get(rat, 0) + 1

        by_gender.setdefault(gen, {"total": 0})
        by_gender[gen]["total"] += 1
        by_gender[gen][rat] = by_gender[gen].get(rat, 0) + 1

        for iss in (t.get("issues") or []):
            issues[iss] = issues.get(iss, 0) + 1

    elapsed = [t["elapsed_s"] for t in ok if t.get("elapsed_s")]
    avg_el  = round(sum(elapsed) / len(elapsed), 1) if elapsed else 0

    return {
        "session_id":   meta["session_id"],
        "total":        meta.get("total", 0),
        "ok":           meta.get("ok", 0),
        "error":        meta.get("error", 0),
        "rated":        len(rated),
        "good":         sum(1 for t in rated if t["rating"] == "good"),
        "mid":          sum(1 for t in rated if t["rating"] == "mid"),
        "bad":          sum(1 for t in rated if t["rating"] == "bad"),
        "avg_elapsed_s": avg_el,
        "by_category":  by_cat,
        "by_gender":    by_gender,
        "issue_counts": issues,
    }


# ── HTML template ──────────────────────────────────────────────────────────

def _page(title: str, body: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="uz">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title} — Lookzi Test</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:#0a0a0a;color:#e0e0e0;
  font-family:'Inter','Segoe UI',sans-serif;
  padding:28px 20px;max-width:1100px;margin:0 auto}}
h1{{font-size:1.6rem;font-weight:800;margin-bottom:6px;
  background:linear-gradient(135deg,#fff 0%,#a0a0ff 100%);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
  background-clip:text}}
h2{{font-size:.72rem;color:#444;font-weight:700;text-transform:uppercase;
  letter-spacing:2px;margin:28px 0 12px}}
a{{color:inherit;text-decoration:none}}

/* Buttons */
.btn{{display:inline-block;padding:8px 16px;background:#1a1a1a;
  border:1px solid #2a2a2a;border-radius:8px;font-size:.8rem;
  cursor:pointer;transition:.15s;color:#aaa}}
.btn:hover{{background:#222;color:#fff}}
.btn.active{{background:#3333aa;border-color:#5555cc;color:#fff}}
.btn.sec{{background:#0f2020;border-color:#1a4040;color:#4ade80}}
.sep{{flex:1}}

/* Nav bar */
.nav-bar{{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:16px;align-items:center}}

/* Progress */
.progress-bar{{position:relative;height:6px;background:#1a1a1a;
  border-radius:3px;margin-bottom:24px;overflow:hidden}}
.progress-fill{{height:100%;background:linear-gradient(90deg,#3b82f6,#a855f7);
  border-radius:3px;transition:width .4s}}
.progress-label{{position:absolute;right:0;top:-18px;
  font-size:.7rem;color:#555}}

/* Review panels */
.review-wrap{{display:flex;flex-direction:column;gap:16px}}
.test-meta{{display:flex;align-items:center;gap:8px;flex-wrap:wrap}}
.badge{{padding:3px 12px;border-radius:100px;font-size:.72rem;font-weight:700;
  background:#1a1a1a;border:1px solid #2a2a2a;color:#aaa}}
.badge.sec{{color:#facc15}}
.panels{{display:grid;grid-template-columns:1fr 1fr 1.4fr;gap:12px}}
.panel{{background:#0f0f0f;border:1px solid #1e1e1e;border-radius:12px;
  padding:12px;display:flex;flex-direction:column;gap:8px}}
.panel.main-panel{{border-color:#3333aa}}
.panel-label{{font-size:.7rem;color:#555;text-transform:uppercase;letter-spacing:1px}}
.panel img{{width:100%;border-radius:8px;object-fit:contain;max-height:420px}}

/* Rating */
.rate-section{{background:#0f0f0f;border:1px solid #1e1e1e;
  border-radius:14px;padding:20px;display:flex;flex-direction:column;gap:14px}}
.rate-row{{display:flex;gap:10px}}
.rate-btn{{padding:12px 28px;border:2px solid #2a2a2a;border-radius:10px;
  background:#1a1a1a;color:#aaa;font-size:.95rem;font-weight:700;
  cursor:pointer;transition:.15s}}
.rate-btn:hover{{transform:translateY(-1px)}}
.rate-btn.good.active{{background:#052e16;border-color:#16a34a;color:#4ade80}}
.rate-btn.mid.active{{background:#2d2000;border-color:#ca8a04;color:#facc15}}
.rate-btn.bad.active{{background:#2d0000;border-color:#dc2626;color:#f87171}}
.issues-row{{display:flex;flex-wrap:wrap;gap:8px}}
.issue-tag{{padding:6px 12px;border:1px solid #2a2a2a;border-radius:20px;
  font-size:.75rem;cursor:pointer;background:#1a1a1a;color:#666;
  user-select:none;transition:.15s}}
.issue-tag input{{display:none}}
.issue-tag.sel,.issue-tag:has(input:checked){{background:#2a1a00;
  border-color:#d97706;color:#facc15}}
textarea{{width:100%;background:#111;border:1px solid #2a2a2a;border-radius:8px;
  color:#e0e0e0;padding:10px;font-size:.85rem;resize:vertical;
  font-family:inherit}}
.save-row{{display:flex;align-items:center;gap:12px}}
.btn-save{{padding:10px 28px;background:linear-gradient(135deg,#1d4ed8,#1e40af);
  border:none;border-radius:10px;color:#fff;font-size:.9rem;font-weight:700;
  cursor:pointer;transition:.15s}}
.btn-save:hover{{opacity:.85}}
#saveStatus{{font-size:.85rem;font-weight:600}}

/* Nav arrows */
.nav-arrows{{display:flex;justify-content:space-between;margin-top:4px}}
.arrow{{padding:10px 24px;background:#1a1a1a;border:1px solid #2a2a2a;
  border-radius:10px;font-size:.85rem;color:#aaa;transition:.15s}}
.arrow:hover:not(.disabled){{background:#222;color:#fff}}
.arrow.disabled{{opacity:.3;pointer-events:none}}

/* Table */
table{{width:100%;border-collapse:collapse;margin-top:8px}}
th,td{{padding:10px 14px;border-bottom:1px solid #1a1a1a;
  font-size:.83rem;text-align:left}}
th{{color:#555;font-size:.68rem;text-transform:uppercase;letter-spacing:1px}}
tr:hover td{{background:#0f0f0f}}

/* Stats grid */
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(130px,1fr));gap:10px}}
.stat{{background:#0f0f0f;border:1px solid #1e1e1e;border-radius:12px;padding:16px}}
.stat .lbl{{font-size:.68rem;color:#444;text-transform:uppercase;
  letter-spacing:1.5px;margin-bottom:8px}}
.stat .val{{font-size:1.1rem;font-weight:700;color:#fff}}
.val.g{{color:#4ade80}}.val.y{{color:#facc15}}.val.r{{color:#f87171}}

/* Error */
.error-box{{background:#1a0000;border:1px solid #7f1d1d;border-radius:10px;
  padding:20px;color:#f87171;font-size:.9rem}}

@media(max-width:700px){{
  .panels{{grid-template-columns:1fr}}
  .rate-row{{flex-direction:column}}
}}
</style>
</head>
<body>
{body}
</body>
</html>"""
