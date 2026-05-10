"""
GraduationPictureService.py — alpha-masked aligned stitching

Uses the raw PNG alpha channel to mask compositing: transparent pixels
contribute weight=0, so background areas stay exactly black and are
cleanly removed by the final crop.

Usage:  python GraduationPictureService.py
Requires: pip install requests Pillow numpy
"""

import os, shutil, math, random, requests, numpy as np
from collections import deque
from PIL import Image
from io import BytesIO

# ── Config ──────────────────────────────────────────────────────────────────
BASE_URL  = "http://magnifier.flashphotography.com/"
RENDER    = "MagnifyRender.ashx"
PARAMS    = dict(O="27341211", R="10212", F="0305", A="71714")
REFERER   = BASE_URL + "Magnify.aspx?" + "&".join(f"{k}={v}" for k,v in PARAMS.items())

TILE_SIZE = 117
BOX_CX    = 94
BOX_CY    = 94
HALF      = 59
STEP      = 100
SEARCH    = 20
TILES_DIR = "tiles"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 Chrome/124.0 Safari/537.36",
    "Accept":     "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
    "Referer":    REFERER,
}

x_centers = [HALF + c * STEP for c in range(5)]   # 59,159,259,359,459
y_centers  = [HALF + r * STEP for r in range(7)]   # 59,…,659
n_cols, n_rows = len(x_centers), len(y_centers)
HALF1 = TILE_SIZE // 2      # 58
HALF2 = TILE_SIZE - HALF1   # 59


# ── Helpers ──────────────────────────────────────────────────────────────────

def make_url(x, y):
    q = "&".join(f"{k}={v}" for k,v in PARAMS.items())
    return f"{BASE_URL}{RENDER}?X={x}&Y={y}&{q}&rand={random.random()}"


def fetch_and_crop(x, y):
    """
    Download tile, return (rgb_crop, alpha_crop) both 117×117.
    Alpha channel is taken from the raw PNG BEFORE compositing so we know
    exactly which pixels are real photo content (alpha>0) vs transparent.
    The RGB is composited onto white so the saved preview looks clean.
    """
    r = requests.get(make_url(x, y), headers=HEADERS, timeout=30)
    r.raise_for_status()
    raw   = Image.open(BytesIO(r.content)).convert("RGBA")
    alpha = raw.split()[3]   # L-mode, 0=transparent, 255=opaque

    # Composite RGB onto white for a clean preview
    white_bg = Image.new("RGB", raw.size, (255, 255, 255))
    white_bg.paste(raw.convert("RGB"), mask=alpha)

    box        = (BOX_CX-HALF1, BOX_CY-HALF1, BOX_CX+HALF2, BOX_CY+HALF2)
    rgb_crop   = white_bg.crop(box)
    alpha_crop = alpha.crop(box)

    assert rgb_crop.size == (TILE_SIZE, TILE_SIZE)
    return rgb_crop, alpha_crop


def find_offset(data_a, data_b, nom_dx, nom_dy):
    """Cross-correlation on the RGB parts to find precise (dx,dy) offset."""
    tile_a, _ = data_a
    tile_b, _ = data_b
    a = np.array(tile_a.convert("L"), dtype=np.float32)
    b = np.array(tile_b.convert("L"), dtype=np.float32)
    H, W = a.shape

    best_mse = float("inf")
    best_dx, best_dy = nom_dx, nom_dy

    for dy in range(nom_dy - SEARCH, nom_dy + SEARCH + 1):
        for dx in range(nom_dx - SEARCH, nom_dx + SEARCH + 1):
            ox1 = max(0, dx);  ox2 = min(W, W + dx)
            oy1 = max(0, dy);  oy2 = min(H, H + dy)
            if ox2 <= ox1 or oy2 <= oy1 or (ox2-ox1)*(oy2-oy1) < 30:
                continue
            a_reg = a[oy1:oy2, ox1:ox2]
            b_reg = b[oy1-dy:oy2-dy, ox1-dx:ox2-dx]
            if a_reg.shape != b_reg.shape:
                continue
            mse = float(np.mean((a_reg - b_reg) ** 2))
            if mse < best_mse:
                best_mse = mse
                best_dx, best_dy = dx, dy

    return best_dx, best_dy, best_mse


def build_positions(h_off, v_off):
    pos   = {(0, 0): (0, 0)}
    queue = deque([(0, 0)])
    while queue:
        col, row = queue.popleft()
        bx, by   = pos[(col, row)]
        edges = [
            ((col+1, row),   h_off.get((col,   row)),    +1),
            ((col-1, row),   h_off.get((col-1, row)),    -1),
            ((col,   row+1), v_off.get((col,   row)),    +1),
            ((col,   row-1), v_off.get((col,   row-1)), -1),
        ]
        for (nc, nr), offset, sign in edges:
            if offset is None or (nc, nr) in pos:
                continue
            if 0 <= nc < n_cols and 0 <= nr < n_rows:
                dx, dy = offset
                pos[(nc, nr)] = (bx + sign*dx, by + sign*dy)
                queue.append((nc, nr))
    for row in range(n_rows):
        for col in range(n_cols):
            if (col, row) not in pos:
                pos[(col, row)] = (col * STEP, row * STEP)
                print(f"  ⚠ nominal fallback [{col},{row}]")
    return pos


def composite(tiles, positions):
    """
    Weighted composite using each tile's alpha channel as a mask.
    Transparent pixels (alpha=0) contribute weight=0, so areas outside
    the photo stay exactly black (0,0,0) in the result — easy to crop.
    """
    min_x = min(p[0] for p in positions.values())
    min_y = min(p[1] for p in positions.values())
    adj   = {k: (v[0]-min_x, v[1]-min_y) for k,v in positions.items()}

    cw = max(p[0] for p in adj.values()) + TILE_SIZE
    ch = max(p[1] for p in adj.values()) + TILE_SIZE

    # Centre-distance weight template
    cy_t, cx_t = TILE_SIZE / 2.0, TILE_SIZE / 2.0
    ys, xs     = np.mgrid[0:TILE_SIZE, 0:TILE_SIZE]
    cw_tmpl    = (np.clip(1.0 - np.abs(xs - cx_t) / cx_t, 0, 1) *
                  np.clip(1.0 - np.abs(ys - cy_t) / cy_t, 0, 1))

    color_buf  = np.zeros((ch, cw, 3), dtype=np.float64)
    weight_buf = np.zeros((ch, cw),    dtype=np.float64)

    for key, (px, py) in sorted(adj.items()):
        if key not in tiles:
            continue
        rgb_tile, alpha_tile = tiles[key]

        rgb   = np.array(rgb_tile,   dtype=np.float64)
        alpha = np.array(alpha_tile, dtype=np.float32) / 255.0  # 0..1

        # Weight = centre-distance × alpha (transparent pixels get 0)
        w = cw_tmpl * alpha

        dx1 = px;  dx2 = min(cw, px + TILE_SIZE)
        dy1 = py;  dy2 = min(ch, py + TILE_SIZE)
        sx2 = dx2 - dx1;  sy2 = dy2 - dy1

        color_buf [dy1:dy2, dx1:dx2] += rgb[:sy2, :sx2] * w[:sy2, :sx2, np.newaxis]
        weight_buf[dy1:dy2, dx1:dx2] += w[:sy2, :sx2]

    safe_w   = np.where(weight_buf > 0, weight_buf, 1.0)
    rgb      = np.where(weight_buf[:, :, np.newaxis] > 0,
                        color_buf / safe_w[:, :, np.newaxis],
                        0.0)
    rgb_u8   = np.clip(rgb, 0, 255).astype(np.uint8)
    # Alpha: 255 where real content, 0 where transparent background.
    # getbbox() on RGBA finds the exact non-transparent bounding box.
    alpha_u8 = (weight_buf > 0).astype(np.uint8) * 255
    rgba     = np.dstack([rgb_u8, alpha_u8])
    return Image.fromarray(rgba, "RGBA")


def crop_to_content(img, black_thresh=5):
    """
    Tight crop to bounding box of non-black pixels.
    After alpha-masked compositing, background = exactly black (0,0,0).
    Any pixel with any channel > black_thresh is real photo content.
    """
    arr        = np.array(img, dtype=np.uint8)
    is_content = ((arr[:, :, 0] > black_thresh) |
                  (arr[:, :, 1] > black_thresh) |
                  (arr[:, :, 2] > black_thresh))

    rows = np.where(is_content.any(axis=1))[0]
    cols = np.where(is_content.any(axis=0))[0]

    if len(rows) == 0 or len(cols) == 0:
        print("  Warning: no content found")
        return img

    t, b = int(rows[0]), int(rows[-1]) + 1
    l, r = int(cols[0]), int(cols[-1]) + 1
    print(f"  Content crop: ({l},{t}) → ({r},{b})  →  {r-l}×{b-t} px")
    return img.crop((l, t, r, b))


# ── All 19 images ────────────────────────────────────────────────────────────

import re

ORDERS_BASE = "https://orders.flashphotography.com/Orders/"

# Fallback list (used only if scraping fails)
IMAGES = [
    ("27341311", "00001", "0636", "Stage_Pose"),
    ("27341311", "00003", "1629", "Celebration"),
    ("27341311", "00005", "0063", "PRs"),
    ("27341311", "10201", "0305", "Alma_Mater_1"),
    ("27341311", "10202", "0305", "Alma_Mater_2"),
    ("27341311", "10203", "0305", "Alma_Mater_3"),
    ("27341311", "10204", "0305", "Alma_Mater_4"),
    ("27341311", "10205", "0305", "Azure"),
    ("27341311", "10206", "0305", "Flag"),
    ("27341311", "10207", "0305", "Goldenrod"),
    ("27341311", "10208", "0305", "Landmark_1"),
    ("27341311", "10209", "0305", "Landmark_2"),
    ("27341311", "10210", "0305", "Landmark_3"),
    ("27341311", "10211", "0305", "Onyx"),
    ("27341311", "10212", "0305", "Ring_1"),
    ("27341311", "10213", "0305", "Ring_2"),
    ("27341311", "10214", "0305", "Sanzio"),
    ("27341311", "10215", "0305", "Studio"),
    ("27341311", "10216", "0305", "Texas_Flag"),
    ("27341311", "10217", "0305", "Venue"),
]
OUTPUT_DIR = "output"


def scrape_images(page_url):
    """
    Log in to the Flash Photography orders site and scrape the Proofs page
    for all image parameters.  Returns (images_list, a_param).
    """
    page_hdrs = {
        "User-Agent": HEADERS["User-Agent"],
        "Accept": "text/html,application/xhtml+xml,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }
    session = requests.Session()

    # Always target Proofs.aspx — the page that lists ALL images
    proofs_url = ORDERS_BASE + "Proofs.aspx"
    print(f"  Fetching proofs page: {proofs_url}")
    resp = session.get(proofs_url, headers=page_hdrs, timeout=30)

    if "Default.aspx" in resp.url or "re-enter" in resp.text.lower():
        print("  Session expired — logging in…")

        # POST to the form's own action URL (may include ?Action=Expire)
        form_action = re.search(r'<form[^>]+action=["\'\']([^\"\'\']+)["\'\']', resp.text, re.I)
        login_url = ORDERS_BASE + (form_action.group(1).lstrip('./') if form_action else "Default.aspx?Action=Expire")
        login_resp = session.get(login_url, headers=page_hdrs, timeout=30)

        # Extract ALL hidden ASP.NET fields
        form = {}
        for m in re.finditer(r'<input[^>]+>', login_resp.text, re.I):
            t = re.search(r'type=["\'\']([^\"\'\']+)["\'\']', m.group(), re.I)
            n = re.search(r'name=["\'\']([^\"\'\']+)["\'\']', m.group(), re.I)
            v = re.search(r'value=["\'\']([^\"\'\']*)["\'\']', m.group(), re.I)
            if t and n and t.group(1).lower() == 'hidden':
                form[n.group(1)] = v.group(1) if v else ''

        # Find text inputs in order: first = ID/PIN, second = LastName
        text_inputs = re.findall(r'<input[^>]+type=["\'\']text["\'\'][^>]*>', login_resp.text, re.I)
        field_names = []
        for inp in text_inputs:
            nm = re.search(r'name=["\'\']([^\"\'\']+)["\'\']', inp, re.I)
            if nm:
                field_names.append(nm.group(1))
        id_field   = field_names[0] if len(field_names) > 0 else 'PIN'
        last_field = field_names[1] if len(field_names) > 1 else 'LastName'

        # Find submit button
        btn_m = re.search(r'<input[^>]+type=["\'\']submit["\'\'][^>]+name=["\'\']([^\"\'\']+)["\'\']',
                           login_resp.text, re.I)

        print(f"  Login fields: id={id_field!r}  last={last_field!r}  btn={btn_m.group(1) if btn_m else None!r}")
        order_id  = input("  Order ID:   ").strip()
        last_name = input("  Last name:  ").strip()

        form[id_field]   = order_id
        form[last_field] = last_name
        if btn_m:
            form[btn_m.group(1)] = "Submit"

        print(f"  POSTing to {login_url} …")
        post_resp = session.post(login_url, data=form, headers=page_hdrs, timeout=30)
        print(f"  POST landed on: {post_resp.url}")

        # Now fetch Proofs.aspx with the authenticated session
        resp = session.get(proofs_url, headers=page_hdrs, timeout=30)
        print(f"  Proofs page URL: {resp.url}")

    html = resp.text

    # Save the authenticated page for debugging
    with open("debug_proofs_page.html", "w", encoding="utf-8") as f:
        f.write(f"<!-- URL: {resp.url} -->\n")
        f.write(html)
    print(f"  Saved debug_proofs_page.html ({len(html):,} chars)")

    # Extract A parameter from JavaScript
    a_match = re.search(r"dimensionHash\['AccountNumber'\]\s*=\s*'(\d+)'", html)
    a_param = a_match.group(1) if a_match else "71714"
    print(f"  A = {a_param}")

    # Parse ORF spans and ViewName spans — most reliable selector in the page
    orfs  = re.findall(r'class="ORF">(\d+)-(\w+)-(\w+)<', html)
    names = re.findall(r'class="ViewName">([^<]+)<', html)

    images = []
    for (O, R, F), name in zip(orfs, names):
        clean = re.sub(r'\s+', '_', name.strip())
        images.append((O, R, F, clean))
        print(f"    {O}  R={R}  F={F}  → {clean}")

    print(f"  Found {len(images)} image(s).")
    return images, a_param


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    global PARAMS, REFERER, HEADERS

    print("Flash Photography — Bulk Downloader\n")
    page_url = input("Packages/Proofs page URL (or Enter for hardcoded list): ").strip()

    if page_url:
        images, a_param = scrape_images(page_url)
        if not images:
            print("\nERROR: Could not find any images in the scraped page.")
            print("Check debug_proofs_page.html to see what the server returned.")
            return
    else:
        images, a_param = IMAGES, "71714"

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\nProcessing {len(images)} images → '{OUTPUT_DIR}/'\n")

    for img_num, (O, R, F, name) in enumerate(images, 1):
        print(f"\n{'='*55}")
        print(f"[{img_num:02d}/{len(images)}] {name}  (O={O} R={R} F={F})")
        print(f"{'='*55}")

        PARAMS  = dict(O=O, R=R, F=F, A=a_param)
        REFERER = BASE_URL + "Magnify.aspx?" + "&".join(f"{k}={v}" for k,v in PARAMS.items())
        HEADERS["Referer"] = REFERER

        # Pre-fetch magnifier page to initialise server session
        try:
            requests.get(REFERER, headers={**HEADERS, "Accept": "text/html,*/*"}, timeout=30)
            print("  Magnifier page loaded OK")
        except Exception as e:
            print(f"  Warning: {e}")

        tile_dir = os.path.join(TILES_DIR, f"{img_num:02d}_{name}")
        shutil.rmtree(tile_dir, ignore_errors=True)
        os.makedirs(tile_dir)

        try:
            print("── Downloading ──")
            tiles = {}
            for row, y in enumerate(y_centers):
                for col, x in enumerate(x_centers):
                    rgb, alpha = fetch_and_crop(x, y)
                    rgb.save(os.path.join(tile_dir, f"crop_{col}_{row}.png"))
                    tiles[(col, row)] = (rgb, alpha)
                    # print(f"  [{col},{row}] X={x:4d} Y={y:4d}")

            print("\n── Horizontal offsets ──")
            h_off = {}
            for row in range(n_rows):
                for col in range(n_cols - 1):
                    dx, dy, mse = find_offset(tiles[(col,row)], tiles[(col+1,row)], STEP, 0)
                    h_off[(col, row)] = (dx, dy)
                    # print(f"  [{col},{row}]→[{col+1},{row}]:  dx={dx:4d}  dy={dy:+3d}  mse={mse:6.1f}"
                          # + ("  ← JITTER" if abs(dy) > 2 else ""))

            print("\n── Vertical offsets ──")
            v_off = {}
            for col in range(n_cols):
                for row in range(n_rows - 1):
                    dx, dy, mse = find_offset(tiles[(col,row)], tiles[(col,row+1)], 0, STEP)
                    v_off[(col, row)] = (dx, dy)
                    # print(f"  [{col},{row}]→[{col},{row+1}]:  dx={dx:+3d}  dy={dy:4d}  mse={mse:6.1f}"
                          # + ("  ← JITTER" if abs(dx) > 2 else ""))

            print("\n── Global positions ──")
            positions = build_positions(h_off, v_off)

            print("\n── Compositing ──")
            result = composite(tiles, positions)
            result = result.crop((8, 6, 488, 641)).convert("RGB")

            out = os.path.join(OUTPUT_DIR, f"{img_num:02d}_{name}.jpg")
            result.save(out, "JPEG", quality=95)
            print(f"\nSaved → {out}  ({result.size[0]}×{result.size[1]})")

        except Exception as e:
            print(f"\n  ERROR: {e}")
            import traceback; traceback.print_exc()

    print(f"\n{'='*55}")
    print(f"All done. Results in '{OUTPUT_DIR}/'")


if __name__ == "__main__":
    main()
