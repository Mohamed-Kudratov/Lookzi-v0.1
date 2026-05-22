"""
Lookzi Batch Test Runner
========================
Assets papkasidagi rasmlar asosida avtomatik test o'tkazadi.

Usage:
    python scripts/run_tests.py
    python scripts/run_tests.py --mode sample --n 5
    python scripts/run_tests.py --host http://localhost:7860
    python scripts/run_tests.py --session mysession

Modes:
    all     — har bir modelga bir kiyim (index bo'yicha) → 120 test
    sample  — har kategoriyadan N tasodifiy juft      → N×6 test
    full    — barcha kombinatsiyalar (uzoq!)           → 2400 test
"""

import argparse
import json
import os
import random
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

import requests
from tqdm import tqdm

# ── Papka strukturasi ──────────────────────────────────────────────────────
ASSETS_MAP = {
    "woman": {
        "models":  "Assets/Woman/models",
        "Upper":   "Assets/Woman/Upper",
        "Lower":   "Assets/Woman/lower",
        "Overall": "Assets/Woman/overall",
    },
    "man": {
        "models":  "Assets/man/models",
        "Upper":   "Assets/man/upper",
        "Lower":   "Assets/man/lower",
        "Overall": "Assets/man/overall",
    },
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

ROOT       = Path(__file__).parent.parent
TEST_DIR   = ROOT / "test_results"


def list_images(folder: str) -> list[Path]:
    p = ROOT / folder
    if not p.exists():
        return []
    return sorted(f for f in p.iterdir() if f.suffix.lower() in IMAGE_EXTS)


def build_pairs(mode: str, n: int) -> list[dict]:
    """Test juftlarini yaratish."""
    pairs = []

    for gender, paths in ASSETS_MAP.items():
        models = list_images(paths["models"])
        if not models:
            print(f"[!] Models topilmadi: {paths['models']}")
            continue

        for category in ("Upper", "Lower", "Overall"):
            garments = list_images(paths[category])
            if not garments:
                print(f"[!] Garments topilmadi: {paths[category]}")
                continue

            if mode == "all":
                # n-chi model ↔ n-chi kiyim (zip)
                for i, (m, g) in enumerate(zip(models, garments)):
                    pairs.append(_make_pair(gender, category, m, g, i))

            elif mode == "sample":
                sample_m = random.sample(models,   min(n, len(models)))
                sample_g = random.sample(garments, min(n, len(garments)))
                for i, (m, g) in enumerate(zip(sample_m, sample_g)):
                    pairs.append(_make_pair(gender, category, m, g, i))

            elif mode == "full":
                for i, m in enumerate(models):
                    for j, g in enumerate(garments):
                        pairs.append(_make_pair(gender, category, m, g, i * 100 + j))

    return pairs


def _make_pair(gender, category, model_path, garment_path, idx):
    return {
        "id":           f"{gender}_{category.lower()}_{idx:04d}",
        "gender":       gender,
        "category":     category,
        "person_path":  str(model_path.relative_to(ROOT)),
        "garment_path": str(garment_path.relative_to(ROOT)),
        "result_path":  None,
        "status":       "pending",
        "error":        None,
        "elapsed_s":    None,
        "rating":       None,
        "issues":       [],
        "note":         "",
    }


def run_test(pair: dict, session_dir: Path, host: str) -> dict:
    """Bitta testni API orqali bajarish."""
    person_file  = ROOT / pair["person_path"]
    garment_file = ROOT / pair["garment_path"]

    result_name = f"{pair['id']}.png"
    result_path = session_dir / result_name

    try:
        with open(person_file,  "rb") as pf, \
             open(garment_file, "rb") as gf:
            t0 = time.time()
            resp = requests.post(
                f"{host}/api/tryon",
                files={
                    "person_image":  ("person.jpg",  pf, "image/jpeg"),
                    "garment_image": ("garment.jpg", gf, "image/jpeg"),
                },
                data={
                    "category":           pair["category"],
                    "garment_photo_type": "model",
                    "return_base64":      "false",
                },
                timeout=120,
            )
            elapsed = round(time.time() - t0, 1)

        if resp.status_code == 200:
            result_path.write_bytes(resp.content)
            pair["status"]    = "ok"
            pair["elapsed_s"] = elapsed
            pair["result_path"] = f"test_results/{session_dir.name}/{result_name}"
        else:
            pair["status"] = "error"
            pair["error"]  = f"HTTP {resp.status_code}: {resp.text[:200]}"

    except Exception as e:
        pair["status"] = "error"
        pair["error"]  = str(e)

    return pair


def save_metadata(meta: dict, session_dir: Path):
    path = session_dir / "metadata.json"
    path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Lookzi Batch Test Runner")
    parser.add_argument("--mode",    default="all",
                        choices=["all", "sample", "full"],
                        help="Test rejimi (default: all)")
    parser.add_argument("--n",       type=int, default=5,
                        help="sample rejimida har kategoriyadan nechta juft (default: 5)")
    parser.add_argument("--host",    default="http://127.0.0.1:7860",
                        help="Server manzili")
    parser.add_argument("--session", default=None,
                        help="Session nomi (default: avtomatik UUID)")
    args = parser.parse_args()

    # ── Session papkasi ────────────────────────────────────────────────────
    session_id  = args.session or datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = TEST_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    # ── Juftlarni yaratish ─────────────────────────────────────────────────
    pairs = build_pairs(args.mode, args.n)
    if not pairs:
        print("Hech qanday test juft topilmadi. Assets papkasini tekshiring.")
        sys.exit(1)

    print(f"\n{'='*55}")
    print(f"  Lookzi Batch Test Runner")
    print(f"{'='*55}")
    print(f"  Session  : {session_id}")
    print(f"  Mode     : {args.mode}" + (f" (n={args.n})" if args.mode=="sample" else ""))
    print(f"  Tests    : {len(pairs)}")
    print(f"  Host     : {args.host}")
    print(f"  Output   : {session_dir}")
    print(f"{'='*55}\n")

    # ── Server tekshirish ──────────────────────────────────────────────────
    try:
        r = requests.get(f"{args.host}/api/health", timeout=10)
        info = r.json()
        if not info.get("model_loaded"):
            print("[!] Model yuklanmagan — testlar baribir boshlanadi (natijalar xato bo'lishi mumkin).")
        else:
            print(f"Server OK — {info.get('device','?')} | model loaded ✓\n")
    except Exception as e:
        print(f"[!] Health check xatosi: {e} — davom etilmoqda...")

    # ── Metadata saqlash (boshlang'ich) ────────────────────────────────────
    meta = {
        "session_id":  session_id,
        "created_at":  datetime.now().isoformat(),
        "mode":        args.mode,
        "host":        args.host,
        "total":       len(pairs),
        "ok":          0,
        "error":       0,
        "tests":       pairs,
    }
    save_metadata(meta, session_dir)

    # ── Test loop ──────────────────────────────────────────────────────────
    ok_count = err_count = 0

    with tqdm(pairs, unit="test", ncols=70) as bar:
        for pair in bar:
            bar.set_description(f"{pair['gender']:5s} {pair['category']:7s}")
            run_test(pair, session_dir, args.host)

            if pair["status"] == "ok":
                ok_count += 1
                bar.set_postfix(ok=ok_count, err=err_count)
            else:
                err_count += 1
                tqdm.write(f"  [ERR] {pair['id']}: {pair['error']}")

            meta["ok"]    = ok_count
            meta["error"] = err_count
            save_metadata(meta, session_dir)

    # ── Yakuniy hisobot ────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  Tugadi!  ✅ {ok_count}  ❌ {err_count}  / {len(pairs)}")
    print(f"  Natijalar: {session_dir}")
    print(f"\n  Ko'rish uchun:")
    print(f"  {args.host}/tests/{session_id}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
