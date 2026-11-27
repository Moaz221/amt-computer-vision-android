#!/usr/bin/env python3
"""Generate the MOAZ Computer Vision launcher icon (original artwork, no external assets).

Calm deep-blue gradient with an eye + camera focus brackets: what the app does is
on-device computer vision. Writes legacy mipmap PNGs, an adaptive-icon
foreground + background, and previews.
"""
import math
import os

from PIL import Image, ImageDraw, ImageFilter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "app", "src", "main", "res")

S = 1024  # master canvas

TOP = (58, 96, 138)    # calm steel blue
BOTTOM = (13, 26, 43)  # deep navy
EYE_WHITE = (232, 243, 249)
IRIS = (86, 200, 232)
IRIS_DEEP = (38, 130, 170)
PUPIL = (10, 24, 40)
BRACKET = (143, 227, 255)


def gradient(size, top, bottom):
    g = Image.new("RGB", (1, size))
    for y in range(size):
        t = y / (size - 1)
        g.putpixel((0, y), tuple(int(top[i] + (bottom[i] - top[i]) * t) for i in range(3)))
    return g.resize((size, size))


def almond(cx, cy, w, h, steps=120):
    """Eye outline: pointed canthi, slightly asymmetric lids."""
    up, lo = [], []
    for i in range(steps + 1):
        t = i / steps
        x = cx - w / 2 + w * t
        arch = math.sin(math.pi * t)
        up.append((x, cy - h * 0.52 * arch))
        lo.append((x, cy + h * 0.44 * arch))
    return up + lo[::-1]


def draw_glyph(draw, box, glow_layer=None):
    """Draw the eye + focus brackets inside box=(x0, y0, x1, y1)."""
    x0, y0, x1, y1 = box
    w = x1 - x0
    h = y1 - y0
    cx, cy = x0 + w / 2, y0 + h / 2

    # soft glow behind the eye
    if glow_layer is not None:
        gd = ImageDraw.Draw(glow_layer)
        gr = 0.34 * w
        gd.ellipse([cx - gr, cy - gr, cx + gr, cy + gr], fill=(180, 220, 240, 70))

    # camera focus brackets
    m = 0.06 * w          # inset from box edges
    arm = 0.13 * w        # bracket arm length
    st = max(int(0.016 * w), 2)
    for sx in (0, 1):
        for sy in (0, 1):
            bx = x0 + m if sx == 0 else x1 - m
            by = y0 + m if sy == 0 else y1 - m
            dx = 1 if sx == 0 else -1
            dy = 1 if sy == 0 else -1
            draw.line([bx, by, bx + dx * arm, by], fill=BRACKET, width=st)
            draw.line([bx, by, bx, by + dy * arm], fill=BRACKET, width=st)

    # eye
    eye_w, eye_h = 0.62 * w, 0.34 * w
    draw.polygon(almond(cx, cy, eye_w, eye_h), fill=EYE_WHITE)

    # iris, deep ring, pupil, highlight
    ir = 0.135 * w
    draw.ellipse([cx - ir, cy - ir, cx + ir, cy + ir], fill=IRIS)
    rr = ir * 0.72
    draw.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], fill=IRIS_DEEP)
    pr = ir * 0.42
    draw.ellipse([cx - pr, cy - pr, cx + pr, cy + pr], fill=PUPIL)
    hr = ir * 0.16
    hx, hy = cx - ir * 0.35, cy - ir * 0.35
    draw.ellipse([hx - hr, hy - hr, hx + hr, hy + hr], fill=EYE_WHITE)


def rounded_mask(size, radius):
    m = Image.new("L", (size, size), 0)
    d = ImageDraw.Draw(m)
    d.rounded_rectangle([0, 0, size - 1, size - 1], radius=radius, fill=255)
    return m


def build_master():
    img = gradient(S, TOP, BOTTOM).convert("RGBA")
    glow = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    glyph = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(glyph)
    draw_glyph(d, (0, 0, S, S), glow_layer=glow)
    glow = glow.filter(ImageFilter.GaussianBlur(S * 0.03))
    img = Image.alpha_composite(img, glow)
    img = Image.alpha_composite(img, glyph)
    return img


def build_foreground():
    """Adaptive-icon foreground: transparent, glyph in the central safe zone."""
    fg = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    glow = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(fg)
    m = S * 0.17
    draw_glyph(d, (m, m, S - m, S - m), glow_layer=glow)
    glow = glow.filter(ImageFilter.GaussianBlur(S * 0.02))
    return Image.alpha_composite(glow, fg)


def main():
    master = build_master()
    fg = build_foreground()

    densities = {"mdpi": 48, "hdpi": 72, "xhdpi": 96, "xxhdpi": 144, "xxxhdpi": 192}
    for name, px in densities.items():
        out = os.path.join(RES, f"mipmap-{name}")
        os.makedirs(out, exist_ok=True)
        icon = master.resize((px, px), Image.LANCZOS)
        icon.putalpha(rounded_mask(px, int(px * 0.18)))
        icon.save(os.path.join(out, "ic_launcher.png"))
        icon.save(os.path.join(out, "ic_launcher_round.png"))
        fgs = fg.resize((px * 4, px * 4), Image.LANCZOS)  # foreground drawn at 4x density bucket
        fgs.save(os.path.join(out, "ic_launcher_foreground.png"))

    v26 = os.path.join(RES, "mipmap-anydpi-v26")
    os.makedirs(v26, exist_ok=True)
    adaptive = (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        '<adaptive-icon xmlns:android="http://schemas.android.com/apk/res/android">\n'
        '    <background android:drawable="@color/ic_launcher_background"/>\n'
        '    <foreground android:drawable="@mipmap/ic_launcher_foreground"/>\n'
        '</adaptive-icon>\n'
    )
    for fname in ("ic_launcher.xml", "ic_launcher_round.xml"):
        with open(os.path.join(v26, fname), "w") as f:
            f.write(adaptive)

    values = os.path.join(RES, "values")
    with open(os.path.join(values, "ic_launcher_background.xml"), "w") as f:
        f.write(
            '<?xml version="1.0" encoding="utf-8"?>\n'
            '<resources>\n'
            '    <color name="ic_launcher_background">#22374F</color>\n'
            '</resources>\n'
        )

    # previews for the report / repo
    prev = os.path.join(ROOT, "screenshots")
    os.makedirs(prev, exist_ok=True)
    master.save(os.path.join(prev, "icon_master.png"))
    canvas = Image.new("RGBA", (S * 2 + 60, S + 120), (10, 16, 26, 255))
    sq = master.copy()
    sq.putalpha(rounded_mask(S, int(S * 0.18)))
    canvas.alpha_composite(sq, (30, 60))
    circ = Image.new("L", (S, S), 0)
    ImageDraw.Draw(circ).ellipse([0, 0, S - 1, S - 1], fill=255)
    rc = master.copy()
    rc.putalpha(circ)
    canvas.alpha_composite(rc, (S + 30, 60))
    canvas.convert("RGB").save(os.path.join(prev, "icon_preview.png"))
    print("icon written to", RES)


if __name__ == "__main__":
    main()
