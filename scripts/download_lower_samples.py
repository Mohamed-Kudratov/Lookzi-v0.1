#!/usr/bin/env python3
"""
Download sample bottom garment images (jeans/trousers/skirts) from Wikimedia Commons.
Images are CC-licensed / public domain. Run once to populate examples/data/garments/.

Usage:
    python scripts/download_lower_samples.py
"""
import os
import sys
from pathlib import Path

try:
    import httpx
except ImportError:
    sys.exit("httpx not found. Run: pip install httpx")

# Force UTF-8 output on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SAVE_DIR   = Path(__file__).parent.parent / "examples" / "data" / "garments"
API        = "https://commons.wikimedia.org/w/api.php"
TARGET     = 6          # how many images to download
START_IDX  = 1          # lower_01.jpg, lower_02.jpg, ...

HEADERS = {"User-Agent": "LookziVTON/1.0 (https://github.com/Mohamed-Kudratov/Lookzi-v0.1; dev) httpx/0.28"}

CATEGORIES = [
    "Product photographs of jeans",
    "Product photographs of trousers",
    "Product photographs of skirts",
    "Jeans",
    "Trousers",
]


def get_category_files(category: str, limit: int = 40) -> list[str]:
    r = httpx.get(
        API,
        params={
            "action": "query",
            "list": "categorymembers",
            "cmtitle": f"Category:{category}",
            "cmtype": "file",
            "cmlimit": limit,
            "format": "json",
        },
        headers=HEADERS,
        timeout=30,
    )
    r.raise_for_status()
    return [m["title"] for m in r.json()["query"]["categorymembers"]]


def get_image_url(title: str) -> str | None:
    r = httpx.get(
        API,
        params={
            "action": "query",
            "titles": title,
            "prop": "imageinfo",
            "iiprop": "url|mime|size",
            "format": "json",
        },
        headers=HEADERS,
        timeout=30,
    )
    r.raise_for_status()
    for page in r.json()["query"]["pages"].values():
        if "imageinfo" not in page:
            continue
        info = page["imageinfo"][0]
        mime = info.get("mime", "")
        size = info.get("size", 0)
        if mime in ("image/jpeg", "image/png") and size > 30_000:
            return info["url"]
    return None


def download(url: str, dest: Path) -> None:
    with httpx.stream("GET", url, headers=HEADERS, timeout=60, follow_redirects=True) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_bytes(8192):
                f.write(chunk)


def main() -> None:
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    count   = 0
    seen    = set()

    for cat in CATEGORIES:
        if count >= TARGET:
            break
        print(f"\nCategory: {cat}")
        try:
            files = get_category_files(cat)
        except Exception as e:
            print(f"  Failed to list: {e}")
            continue

        for title in files:
            if count >= TARGET:
                break
            if title in seen:
                continue
            seen.add(title)

            # Only jpeg/png by extension
            ext = Path(title).suffix.lower()
            if ext not in (".jpg", ".jpeg", ".png"):
                continue

            try:
                url = get_image_url(title)
                if not url:
                    continue
                idx  = START_IDX + count
                dest = SAVE_DIR / f"lower_{idx:02d}.jpg"
                print(f"  {title[:55]} -> lower_{idx:02d}.jpg")
                download(url, dest)
                kb = dest.stat().st_size // 1024
                print(f"    saved  {kb} KB")
                count += 1
            except Exception as e:
                print(f"  skip ({title[:40]}): {e}")

    print(f"\nDone — downloaded {count}/{TARGET} lower garment images to {SAVE_DIR}")
    if count < TARGET:
        print(
            f"\n  [!] Only {count} images found automatically."
            "\n  Add more manually: save pants/jeans/skirts JPGs as"
            f"\n  {SAVE_DIR}\\lower_{count+1:02d}.jpg, lower_{count+2:02d}.jpg ..."
        )


if __name__ == "__main__":
    main()
