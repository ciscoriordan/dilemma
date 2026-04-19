#!/usr/bin/env python3
"""Generate social.jpg (1280x640) from dilemma_highres.png by scaling the hero
to fill the card, adding a gradient scrim at the top, and setting the wordmark
and tagline over it.
"""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).parent.parent
HERO = ROOT / "dilemma_highres.png"
OUT = ROOT / "social.jpg"

CARD_W, CARD_H = 1280, 640
WORDMARK = "Dilemma"
TAGLINE = "Ancient, Medieval, and Modern Greek Lemmatizer"

FONT_PATH = "/Library/Fonts/SF-Pro.ttf"
WORDMARK_VARIATION = b"Bold"
TAGLINE_VARIATION = b"Regular"
WORDMARK_SIZE = 128
TAGLINE_SIZE = 38

PAD_X = 56
PAD_TOP = 48
WORDMARK_TAGLINE_GAP = 28
SCRIM_HEIGHT = 360
SCRIM_TOP_ALPHA = 220
SCRIM_BOTTOM_ALPHA = 0


def cover(img: Image.Image, w: int, h: int) -> Image.Image:
    src_ratio = img.width / img.height
    dst_ratio = w / h
    if src_ratio > dst_ratio:
        new_h = h
        new_w = round(h * src_ratio)
    else:
        new_w = w
        new_h = round(w / src_ratio)
    resized = img.resize((new_w, new_h), Image.LANCZOS)
    left = (new_w - w) // 2
    top = (new_h - h) // 2
    return resized.crop((left, top, left + w, top + h))


def vertical_gradient(w: int, h: int, top_alpha: int, bottom_alpha: int) -> Image.Image:
    gradient = Image.new("L", (1, h))
    for y in range(h):
        t = y / (h - 1)
        gradient.putpixel((0, y), round(top_alpha + (bottom_alpha - top_alpha) * t))
    alpha = gradient.resize((w, h))
    scrim = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    scrim.putalpha(alpha)
    return scrim


def main() -> None:
    hero = Image.open(HERO).convert("RGBA")
    canvas = cover(hero, CARD_W, CARD_H)

    scrim = vertical_gradient(CARD_W, SCRIM_HEIGHT, SCRIM_TOP_ALPHA, SCRIM_BOTTOM_ALPHA)
    canvas.alpha_composite(scrim, (0, 0))

    draw = ImageDraw.Draw(canvas)
    wordmark_font = ImageFont.truetype(FONT_PATH, WORDMARK_SIZE)
    wordmark_font.set_variation_by_name(WORDMARK_VARIATION)
    tagline_font = ImageFont.truetype(FONT_PATH, TAGLINE_SIZE)
    tagline_font.set_variation_by_name(TAGLINE_VARIATION)

    word_bbox = draw.textbbox((0, 0), WORDMARK, font=wordmark_font)
    word_h = word_bbox[3] - word_bbox[1]
    word_y = PAD_TOP

    tag_y = word_y + word_h + WORDMARK_TAGLINE_GAP

    draw.text((PAD_X, word_y), WORDMARK, font=wordmark_font, fill=(255, 255, 255, 255))
    draw.text((PAD_X, tag_y), TAGLINE, font=tagline_font, fill=(230, 230, 230, 255))

    canvas.convert("RGB").save(OUT, "JPEG", quality=85, optimize=True, progressive=True)
    print(f"wrote {OUT} ({CARD_W}x{CARD_H}, {OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
