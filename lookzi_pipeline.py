"""
Lookzi Quality Pipeline
========================
Faza 1 komponentlari:

1. GarmentPreprocessor — fon tozalash, normalize, best-image tanlash
2. PersonValidator     — input sifat tekshirish
3. ResultScorer        — natija sifat ballash (0-100)
4. PRESETS             — category bo'yicha optimal parametrlar
5. run_best()          — multi-generate + avtomatik best tanlash

Import:
    from lookzi_pipeline import preprocess_garment, validate_person, run_best, PRESETS
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image, ImageFilter, ImageStat

if TYPE_CHECKING:
    pass

logger = logging.getLogger("lookzi.pipeline")

# ── rembg (ixtiyoriy) ─────────────────────────────────────────────────────
try:
    from rembg import remove as _rembg_remove
    _REMBG_OK = True
    logger.info("rembg: OK — garment background removal enabled")
except ImportError:
    _REMBG_OK = False
    logger.warning("rembg not installed — background removal disabled. "
                   "Run: pip install rembg  to enable it.")

# ── OpenCV (ixtiyoriy — sharpness uchun) ─────────────────────────────────
try:
    import cv2 as _cv2
    _CV2_OK = True
except ImportError:
    _cv2 = None
    _CV2_OK = False


# ═══════════════════════════════════════════════════════════════════════════
# 1. PRESETS — Category va uslub bo'yicha optimal parametrlar
# ═══════════════════════════════════════════════════════════════════════════

PRESETS: dict[str, dict] = {
    # ── Upper kiyimlar ──────────────────────────────────────────────────
    "Upper": {
        "steps": 30, "guidance": 2.5, "seg_free": True,
        "description": "Ko'ylak, futbolka, bluza",
    },
    "Upper_tight": {
        "steps": 25, "guidance": 2.5, "seg_free": True,
        "description": "Yopishiq kiyim (tight fit)",
    },
    "Upper_loose": {
        "steps": 35, "guidance": 3.0, "seg_free": True,
        "description": "Keng kiyim (hoodie, jacket)",
    },
    "Upper_dark": {
        "steps": 35, "guidance": 3.5, "seg_free": True,
        "description": "Qorong'i rangli kiyim",
    },
    "Upper_pattern": {
        "steps": 40, "guidance": 3.5, "seg_free": True,
        "description": "Naqshli kiyim (Atlas, Adras)",
    },
    "Upper_white": {
        "steps": 30, "guidance": 2.0, "seg_free": True,
        "description": "Oq/och rangli kiyim",
    },

    # ── Lower kiyimlar ──────────────────────────────────────────────────
    "Lower": {
        "steps": 28, "guidance": 2.8, "seg_free": False,
        "description": "Shim, yubka",
    },
    "Lower_skirt": {
        "steps": 33, "guidance": 3.0, "seg_free": True,
        "description": "Yubka (keng, uzun)",
    },

    # ── Overall (to'liq kiyim) ──────────────────────────────────────────
    "Overall": {
        "steps": 35, "guidance": 3.5, "seg_free": True,
        "description": "Ko'ylak-dress, kombinezon",
    },
    "Overall_dress": {
        "steps": 38, "guidance": 3.8, "seg_free": True,
        "description": "Uzun ko'ylak, to'y kiyimi",
    },

    # ── Flat-lay (maneken yo'q, kiyim yotqizilgan) ──────────────────────
    "flat_lay": {
        "steps": 30, "guidance": 3.0, "seg_free": True,
        "photo_type": "flat-lay",
        "description": "Maneken yo'q, yotqizilgan kiyim",
    },
}


def get_preset(category: str, photo_type: str = "model") -> dict:
    """
    Kategoriya bo'yicha eng yaxshi preset parametrlarini qaytaradi.
    Agar maxsus preset topilmasa — asosiy kategoriya presetini beradi.
    """
    if photo_type == "flat-lay":
        return PRESETS["flat_lay"]
    base = PRESETS.get(category, PRESETS.get("Upper"))
    return base.copy()


# ═══════════════════════════════════════════════════════════════════════════
# 2. GARMENT PREPROCESSOR
# ═══════════════════════════════════════════════════════════════════════════

_GARMENT_SIZE = (768, 1024)   # W × H — FASHN VTON talab qilgan o'lcham


def preprocess_garment(img: Image.Image, remove_bg: bool = True) -> Image.Image:
    """
    Garment rasmini model uchun tayyorlaydi:
      1. Fon olib tashlash (rembg, agar o'rnatilgan bo'lsa)
      2. Avtomatik crop + markazlashtirish
      3. Oq fonda 768×1024 canvas ga joylashtirish
      4. Kontrast/yorug'likni normalize qilish

    Args:
        img: Xom garment rasmi (PIL.Image)
        remove_bg: Fon olib tashlashni yoqish (default True)

    Returns:
        Tozalangan, normalize qilingan PIL.Image
    """
    img = img.convert("RGBA")

    # 1. Fon olib tashlash
    if remove_bg and _REMBG_OK:
        try:
            img = _rembg_remove(img)
            logger.debug("rembg: background removed")
        except Exception as e:
            logger.warning("rembg xato: %s", e)
    elif remove_bg and not _REMBG_OK:
        logger.debug("rembg yo'q — fon tozalanmaydi")

    # 2. Alpha kanalga qarab auto-crop
    img = _autocrop_transparent(img)

    # 3. Oq fonda canvas ga joylashtirish
    img = _place_on_canvas(img, _GARMENT_SIZE, bg_color=(255, 255, 255))

    return img.convert("RGB")


def _autocrop_transparent(img: Image.Image) -> Image.Image:
    """Transparent/oq fon chegarasidan kiyimni kesib oladi."""
    if img.mode != "RGBA":
        return img
    alpha = np.array(img.split()[3])
    rows = np.any(alpha > 20, axis=1)
    cols = np.any(alpha > 20, axis=0)
    if not rows.any():
        return img
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    # Padding qo'shish
    pad = 20
    h, w = alpha.shape
    rmin = max(0, rmin - pad)
    rmax = min(h, rmax + pad)
    cmin = max(0, cmin - pad)
    cmax = min(w, cmax + pad)
    return img.crop((cmin, rmin, cmax, rmax))


def _place_on_canvas(img: Image.Image, size: tuple[int, int],
                     bg_color=(255, 255, 255)) -> Image.Image:
    """Rasmni aspect ratio saqlab canvas o'rtasiga joylashtiradi."""
    tw, th = size
    # Aspect ratio saqlagan holda kichraytirish
    img.thumbnail((tw, th), Image.LANCZOS)
    iw, ih = img.size
    canvas = Image.new("RGBA", (tw, th), (*bg_color, 255))
    x = (tw - iw) // 2
    y = (th - ih) // 2
    if img.mode == "RGBA":
        canvas.paste(img, (x, y), mask=img.split()[3])
    else:
        canvas.paste(img, (x, y))
    return canvas


def select_best_garment(images: list[Image.Image]) -> Image.Image:
    """
    Bir nechta garment rasmdan eng yaxshisini tanlaydi.
    Mezonlar: o'lcham, aniqlik, kiyim maydoni.
    """
    if not images:
        raise ValueError("Bo'sh ro'yxat")
    if len(images) == 1:
        return images[0]

    scores = []
    for img in images:
        s = 0.0
        # 1. O'lcham — katta = yaxshi
        s += (img.width * img.height) / (1024 * 1024) * 30

        # 2. Sharpness
        s += _sharpness_score(img) * 0.3

        # 3. Kiyim maydoni (qancha ko'p piksel kiyimga tegishli)
        s += _foreground_ratio(img) * 40

        scores.append(s)

    best_idx = int(np.argmax(scores))
    logger.debug("Best garment: idx=%d score=%.1f", best_idx, scores[best_idx])
    return images[best_idx]


# ═══════════════════════════════════════════════════════════════════════════
# 3. PERSON VALIDATOR
# ═══════════════════════════════════════════════════════════════════════════

class ValidationResult:
    def __init__(self, ok: bool, message: str = "", score: float = 100.0):
        self.ok      = ok
        self.message = message
        self.score   = score

    def __bool__(self):
        return self.ok


def validate_person(img: Image.Image) -> ValidationResult:
    """
    Odam rasmining sifatini tekshiradi.
    Returns: ValidationResult (ok=True yaxshi, ok=False — muammo bor)
    """
    w, h = img.size

    # 1. O'lcham
    if w < 256 or h < 256:
        return ValidationResult(False, f"Rasm juda kichik ({w}x{h}). Min: 384x512")
    if w < 384 or h < 512:
        return ValidationResult(False, f"⚠️ Rasm kichik ({w}×{h}). Yaxshi natija uchun min 384×512")

    # 2. Portrait (balandligi kengindan katta bo'lishi kerak)
    if w > h * 1.2:
        return ValidationResult(False, "⚠️ Rasm gorizontal. To'liq gavda portreti kerak.")

    # 3. Xiralik (sharpness)
    sharp = _sharpness_score(img)
    if sharp < 40:
        return ValidationResult(False, "⚠️ Rasm xiralashgan. Aniqroq surat talab qilinadi.")

    # 4. Juda qorong'i yoki oq
    stat = ImageStat.Stat(img.convert("RGB"))
    mean = sum(stat.mean) / 3
    if mean < 20:
        return ValidationResult(False, "⚠️ Rasm juda qorong'i.")
    if mean > 240:
        return ValidationResult(False, "⚠️ Rasm juda yorug'/overexposed.")

    # Sifat balli
    score = min(100.0, sharp * 0.5 + 50)
    return ValidationResult(True, "✅ Rasm qabul qilindi", score=score)


def validate_garment(img: Image.Image) -> ValidationResult:
    """Garment rasmining sifatini tekshiradi."""
    w, h = img.size

    if w < 128 or h < 128:
        return ValidationResult(False, f"⚠️ Kiyim rasmi juda kichik ({w}×{h})")

    stat = ImageStat.Stat(img.convert("RGB"))
    mean = sum(stat.mean) / 3
    std  = sum(stat.stddev) / 3

    if std < 5:
        return ValidationResult(False, "⚠️ Kiyim rasmi bir xil rang — kiyim ko'rinmayapti?")

    score = min(100.0, std * 1.5 + 30)
    return ValidationResult(True, "✅ Kiyim qabul qilindi", score=score)


# ═══════════════════════════════════════════════════════════════════════════
# 4. RESULT SCORER
# ═══════════════════════════════════════════════════════════════════════════

def score_result(result: Image.Image,
                 garment: Image.Image,
                 person: Image.Image | None = None) -> float:
    """
    Try-on natijasini 0-100 oralig'ida baholaydi.

    Mezonlar:
      - Yorug'lik va kontrast normal bo'lishi
      - Sharpness yetarli bo'lishi
      - Garment rangi saqlanishi
      - Artefakt (qora/oq bloklar) yo'qligi
    """
    score = 100.0
    res_rgb = result.convert("RGB")

    # 1. Umumiy yorug'lik (juda qora yoki oq → artifact)
    stat = ImageStat.Stat(res_rgb)
    mean = sum(stat.mean) / 3
    std  = sum(stat.stddev) / 3

    if mean < 12 or mean > 243:
        score -= 35
    elif mean < 25 or mean > 235:
        score -= 15

    # 2. Kontrast (juda past std → bir xil rangli, yomon)
    if std < 8:
        score -= 30
    elif std < 15:
        score -= 15

    # 3. Sharpness
    sharp = _sharpness_score(res_rgb)
    if sharp < 30:
        score -= 25
    elif sharp < 60:
        score -= 10

    # 4. Garment rang moslik
    color_match = _color_similarity(garment.convert("RGB"), res_rgb)
    if color_match < 0.3:
        score -= 20
    elif color_match < 0.5:
        score -= 10

    # 5. Artefakt bloklar (bir burchakda qora/oq patch)
    if _has_corner_artifacts(res_rgb):
        score -= 15

    return max(0.0, min(100.0, score))


def auto_rate(score: float) -> str:
    """Ball asosida good/mid/bad belgilash."""
    if score >= 72:
        return "good"
    elif score >= 45:
        return "mid"
    else:
        return "bad"


# ═══════════════════════════════════════════════════════════════════════════
# 5. MULTI-GENERATE + BEST SELECTION
# ═══════════════════════════════════════════════════════════════════════════

def run_best(
    run_tryon_fn,                    # app.py dagi run_tryon() funksiyasi
    person_image:  Image.Image,
    garment_image: Image.Image,
    category:      str,
    photo_type:    str   = "model",
    n_samples:     int   = 3,        # nechta variant generate qilish
    use_preset:    bool  = True,     # preset parametrlarini ishlatish
    # Manual override (ixtiyoriy)
    steps:         int   | None = None,
    guidance:      float | None = None,
    seg_free:      bool  | None = None,
) -> tuple[Image.Image | None, str, float]:
    """
    Garment ni preprocessing qilib, n_samples variant generate qiladi,
    eng yaxshi natijani qaytaradi.

    Returns:
        (best_image, status_message, quality_score)
    """
    # 1. Garment preprocessing
    logger.info("Preprocessing garment...")
    try:
        clean_garment = preprocess_garment(garment_image, remove_bg=True)
    except Exception as e:
        logger.warning("Preprocessing xato: %s — xom rasm ishlatiladi", e)
        clean_garment = garment_image

    # 2. Preset parametrlarini olish
    if use_preset:
        preset = get_preset(category, photo_type)
        _steps    = steps    if steps    is not None else preset.get("steps",    30)
        _guidance = guidance if guidance is not None else preset.get("guidance", 2.5)
        _seg_free = seg_free if seg_free is not None else preset.get("seg_free", True)
        _ptype    = preset.get("photo_type", photo_type)
    else:
        _steps    = steps    or 30
        _guidance = guidance or 2.5
        _seg_free = seg_free if seg_free is not None else True
        _ptype    = photo_type

    # 3. N ta variant generate qilish
    candidates: list[tuple[Image.Image, float]] = []
    seeds = [random.randint(1, 2**31) for _ in range(n_samples)]

    logger.info("Generating %d variants (category=%s, steps=%d, guidance=%.1f)",
                n_samples, category, _steps, _guidance)

    for i, seed in enumerate(seeds):
        try:
            result, msg = run_tryon_fn(
                person_image, clean_garment, category, _ptype,
                _steps, _guidance, seed, _seg_free,
            )
            if result is not None:
                s = score_result(result, clean_garment, person_image)
                candidates.append((result, s))
                logger.debug("  Variant %d/%d — score=%.1f", i + 1, n_samples, s)
            else:
                logger.warning("  Variant %d/%d — run_tryon qaytarmadi: %s", i + 1, n_samples, msg)
        except Exception as e:
            logger.warning("  Variant %d/%d — xato: %s", i + 1, n_samples, e)

    if not candidates:
        return None, "❌ Hech qanday natija olinmadi", 0.0

    # 4. Eng yaxshi natijani tanlash
    best_img, best_score = max(candidates, key=lambda x: x[1])
    rating = auto_rate(best_score)
    n_ok   = len(candidates)

    status = (
        f"✅ Done — {n_ok}/{n_samples} variant | "
        f"Sifat: {best_score:.0f}/100 ({rating})"
    )
    logger.info("Best result: score=%.1f rating=%s", best_score, rating)
    return best_img, status, best_score


# ═══════════════════════════════════════════════════════════════════════════
# YORDAMCHI FUNKSIYALAR
# ═══════════════════════════════════════════════════════════════════════════

def _sharpness_score(img: Image.Image) -> float:
    """Laplacian orqali sharpness ballini hisoblaydi (0-100+)."""
    gray = img.convert("L")
    if _CV2_OK:
        try:
            arr = np.array(gray, dtype=np.uint8)
            lap = _cv2.Laplacian(arr, _cv2.CV_64F).var()
        except Exception:
            # Fallback to PIL if cv2 fails
            lap_img = gray.filter(ImageFilter.FIND_EDGES)
            lap = np.array(lap_img, dtype=np.float32).var()
    else:
        lap_img = gray.filter(ImageFilter.FIND_EDGES)
        lap = np.array(lap_img, dtype=np.float32).var()
    return float(min(100.0, lap ** 0.5))


def _foreground_ratio(img: Image.Image) -> float:
    """Rasmda oq bo'lmagan piksellar nisbati (kiyim maydoni taxminiy)."""
    rgb = img.convert("RGB")
    arr = np.array(rgb)
    # Oq fon: R, G, B > 240
    white_mask = np.all(arr > 240, axis=2)
    fg_ratio = 1.0 - (white_mask.sum() / white_mask.size)
    return float(fg_ratio)


def _color_similarity(img1: Image.Image, img2: Image.Image) -> float:
    """
    Ikkita rasmning dominant rangini solishtiradi.
    0.0 = butunlay boshqa, 1.0 = bir xil.
    """
    def dominant(im: Image.Image) -> np.ndarray:
        small = im.resize((32, 32))
        arr   = np.array(small, dtype=np.float32)
        return arr.mean(axis=(0, 1)) / 255.0

    d1 = dominant(img1)
    d2 = dominant(img2)
    diff = np.abs(d1 - d2).mean()
    return float(1.0 - diff)


def _has_corner_artifacts(img: Image.Image, patch_size: int = 30) -> bool:
    """Burchaklarda juda qorong'i/yorug' bloklar borligini tekshiradi."""
    w, h  = img.size
    p     = patch_size
    corners = [
        img.crop((0, 0, p, p)),
        img.crop((w - p, 0, w, p)),
        img.crop((0, h - p, p, h)),
        img.crop((w - p, h - p, w, h)),
    ]
    for c in corners:
        m = sum(ImageStat.Stat(c).mean) / 3
        if m < 8 or m > 247:
            return True
    return False
