"""
White_Img.py
-----------------------
Renders radiology findings text onto white canvas images,
then saves them as JPEGs. Reads image paths and text from a CSV.

Usage:
    python White_Img.py --csv /path/to/data.csv --out_dir ./output --limit 100
"""

import os
import argparse
import pandas as pd
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont


# ---------------------------------------------------------------------------
# Core renderer
# ---------------------------------------------------------------------------

def render_text_on_white(
    text,
    out_image_path="out.jpg",
    image_size=(1024, 512),
    font_path=None,
    font_size=60,
    padding=40,
    scale=1,
    repeat=False,
    spacing=40,
):
    """
    Renders ``text`` onto a white canvas and saves it as a JPEG.

    Parameters
    ----------
    text : str
        The text to render.
    out_image_path : str
        Destination path for the output JPEG.
    image_size : tuple[int, int]
        (width, height) of the output image in pixels.
    font_path : str or None
        Path to a .ttf font file. Falls back to DejaVuSans, then PIL default.
    font_size : int
        Base font size (scaled internally by ``scale``).
    padding : int
        Horizontal padding in pixels.
    scale : int
        Internal super-sampling scale for sharper text (downsampled before save).
    repeat : bool
        If True, tile the text block vertically across the canvas.
    spacing : int
        Extra vertical gap between repeated text blocks.

    Returns
    -------
    str
        Path to the saved JPEG.
    """
    W, H = image_size
    W_s, H_s = W * scale, H * scale
    pad_s = padding * scale

    # --- font loading ---
    def _load_font(path, size):
        try:
            return ImageFont.truetype(path, size, layout_engine=getattr(ImageFont, "LAYOUT_BASIC", None))
        except TypeError:
            return ImageFont.truetype(path, size)

    if font_path is None:
        try:
            font = _load_font("DejaVuSans.ttf", font_size * scale)
        except OSError:
            font = ImageFont.load_default()
    else:
        font = _load_font(font_path, font_size * scale)

    # --- canvas ---
    canvas = Image.new("RGBA", (W_s, H_s), (255, 255, 255, 0))
    draw = ImageDraw.Draw(canvas)

    # --- pixel-accurate word wrap ---
    words = text.split()
    lines, current_line = [], []
    for word in words:
        trial = " ".join(current_line + [word])
        if draw.textlength(trial, font=font) <= (W_s - 2 * pad_s):
            current_line.append(word)
        else:
            if current_line:
                lines.append(" ".join(current_line))
            current_line = [word]
    if current_line:
        lines.append(" ".join(current_line))

    wrapped = "\n".join(lines)

    # --- measure text block ---
    bbox = draw.multiline_textbbox((0, 0), wrapped, font=font, spacing=10 * scale)
    block_w = bbox[2] - bbox[0]
    block_h = bbox[3] - bbox[1]

    # --- draw ---
    draw_kwargs = dict(
        font=font,
        fill=(0, 0, 0, 255),
        spacing=10 * scale,
        stroke_width=int(scale * 2),
        stroke_fill=(255, 255, 255, 255),
        align="center",
    )

    if repeat:
        y = pad_s
        while y < H_s - block_h:
            x = (W_s - block_w) // 2
            draw.multiline_text((x, y), wrapped, **draw_kwargs)
            y += block_h + spacing * scale
    else:
        x = (W_s - block_w) // 2
        y = (H_s - block_h) // 2
        draw.multiline_text((x, y), wrapped, **draw_kwargs)

    # --- downsample + flatten to RGB ---
    canvas_small = canvas.resize((W, H), resample=Image.Resampling.LANCZOS)
    canvas_rgb = Image.new("RGB", (W, H), (255, 255, 255))
    alpha = canvas_small.split()[-1].point(lambda p: 255 if p > 30 else 0)
    canvas_rgb.paste(canvas_small, mask=alpha)

    # --- ensure .jpg extension ---
    if not out_image_path.lower().endswith(".jpg"):
        out_image_path = out_image_path.rsplit(".", 1)[0] + ".jpg"

    canvas_rgb.save(out_image_path, "JPEG", quality=95)
    return out_image_path


# ---------------------------------------------------------------------------
# CSV driver
# ---------------------------------------------------------------------------

def generate_images_from_csv(csv_path, out_dir=None, limit=100):
    """
    Reads a CSV with columns ``images`` and ``Findings_5``,
    renders each finding as a text image, and saves to disk.

    Parameters
    ----------
    csv_path : str
        Path to the input CSV file.
    out_dir : str or None
        If provided, all images are saved here (original filenames kept).
        If None, images are saved relative to the paths in the ``images`` column.
    limit : int
        Maximum number of rows to process.
    """
    df = pd.read_csv(csv_path).head(limit)

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Generating images"):
        img_path = str(row["images"])
        text = str(row["Findings_5"])

        if not img_path.lower().endswith(".jpg"):
            img_path = img_path.rsplit(".", 1)[0] + ".jpg"

        if out_dir:
            img_path = os.path.join(out_dir, os.path.basename(img_path))

        os.makedirs(os.path.dirname(img_path) or ".", exist_ok=True)

        render_text_on_white(
            text,
            out_image_path=img_path,
            image_size=(1200, 800),
            font_size=40,
            repeat=True,
            spacing=30,
        )

    print("Done! All images saved as JPG.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(description="Render CSV findings text as JPEG images.")
    parser.add_argument("--csv",     required=True,       help="Path to input CSV file")
    parser.add_argument("--out_dir", default=None,        help="Output directory")
    parser.add_argument("--limit",   type=int, default=100, help="Max rows to process (default: 100)")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    generate_images_from_csv(args.csv, out_dir=args.out_dir, limit=args.limit)
