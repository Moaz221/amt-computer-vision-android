#!/usr/bin/env python3
"""Generate the AMT Computer Vision launcher icon (original artwork, no external assets).

Calm deep-blue gradient with a photo glyph (sun + mountains) and a small AI sparkle.
Writes legacy mipmap PNGs, an adaptive-icon foreground + background, and previews.
"""
import math
import os

from PIL import Image, ImageDraw, ImageFilter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "app", "src", "main", "res")

S = 1024  # master canvas

TOP = (58, 96, 138)    # calm steel blue
BOTTOM = (13, 26, 43)  # deep navy
SUN = (255, 222, 150)  # soft warm sun
BACK_MTN = (122, 168, 199)
FRONT_MTN = (232, 243, 249)
SPARKLE = (140, 226, 255)


def gradient(size, top, bottom):
    g = Image.new("RGB", (1, size))
    for y in range(size):
        t = y / (size - 1)
        g.putpixel((0, y), tuple(int(top[i] + (bottom[i] - top[i]) * t) for i in range(3)))
    return g.resize((size, size))


def draw_glyph(draw, box, glow_layer=None):
    """Draw the photo glyph inside box=(x0, y0, x1, y1) on draw."""
    x0, y0, x1, y1 = box
    w = x1 - x0
    h = y1 - y0

    def px(fx, fy):
        return (x0 + fx * w, y0 + fy * h)

    # sun with soft glow
    sc, sr = px(0.40, 0.34), 0.10 * w
    if glow_layer is not None:
        gd = ImageDraw.Draw(glow_layer)
        gd.ellipse([sc[0] - sr * 1.9, sc[1] - sr * 1.9, sc[0] + sr * 1.9, sc[1] + sr * 1.9],
                   fill=SUN + (90,))
    draw.ellipse([sc[0] - sr, sc[1] - sr, sc[0] + sr, sc[1] + sr], fill=SUN)

    # mountains (back then front)
    base = px(0, 0.78)[1]
    draw.polygon([px(0.14, 0.78), px(0.38, 0.50), px(0.60, 0.78)], fill=BACK_MTN)
    draw.polygon([px(0.42, 0.78), px(0.64, 0.56), px(0.84, 0.78)], fill=FRONT_MTN)
    # ground line
    draw.rectangle([px(0.14, 0.78)[0], base, px(0.84, 0.78)[0], base + 0.012 * h], fill=FRONT_MTN)

    # AI sparkle (4-point star)
    cx, cy = px(0.72, 0.28)
    r = 0.075 * w
    pts = []
    for i in range(8):
        ang = math.pi / 4 * i - math.pi / 2
        rr = r if i % 2 == 0 else r * 0.28
        pts.append((cx + rr * math.cos(ang), cy + rr * math.sin(ang)))
    draw.polygon(pts, fill=SPARKLE)


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
    glow = glow.filter(ImageFilter.GaussianBlur(S * 0.02))
    img = Image.alpha_composite(img, glow)
    img = Image.alpha_composite(img, glyph)
    return img


def build_foreground():
    """Adaptive-icon foreground: transparent, glyph in the central safe zone."""
    fg = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    glow = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(fg)
    # central 66% safe zone
    m = S * 0.17
    draw_glyph(d, (m, m, S - m, S - m), glow_layer=glow)
    glow = glow.filter(ImageFilter.GaussianBlur(S * 0.015))
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
    # launcher-style preview: circular mask on a dark backdrop, next to the square
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
