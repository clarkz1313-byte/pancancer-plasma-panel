#!/usr/bin/env python3
"""Create the concise Figure 2 UMAP panel without recomputing its embedding.

The approved UMAP render is retained pixel-for-pixel below its headings.  This
script only replaces verbose raster headings with short labels.  It exists
because the environment used for the final figure package does not include
``umap-learn``; substituting another embedding would alter the research result.
"""
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from scipy import ndimage


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "figurev5" / "output" / "fig2f_umap.png"
OUT = ROOT / "figurev6" / "output" / "fig2f_umap.png"
FONT = Path("C:/Windows/Fonts/arialbd.ttf")

CODES = ["AML", "BRC", "CLL", "CRC", "CVX", "ENDC",
         "GLIOM", "LUNGC", "DLBCL", "MYEL", "OVC", "PRC"]
COLORS = ["#b22222", "#c97b63", "#7a3e9d", "#d68600", "#c13d86", "#8e5d2c",
          "#1a8db8", "#2e8b57", "#3856a6", "#8c564b", "#d1495b", "#008b8b"]

TILE_COLUMNS = [(2442, 3263), (3356, 4178), (4270, 5092), (5185, 6006)]
TILE_ROWS = [(267, 1069), (1129, 1931), (1990, 2792)]


def centered(draw, xy, text, font, fill):
    box = draw.textbbox((0, 0), text, font=font)
    draw.text((xy[0] - (box[2] - box[0]) / 2, xy[1]), text,
              font=font, fill=fill)


def _rgb(hex_color):
    return tuple(bytes.fromhex(hex_color.lstrip("#")))


def enlarge_focal_dots(img, box, color):
    """Thicken only the focal-colour marks inside one locked UMAP tile."""
    x0, y0, x1, y1 = box
    crop = img.crop((x0 + 12, y0 + 12, x1 - 12, y1 - 12)).convert("RGB")
    target = _rgb(color)
    pixels = crop.load()
    mask = Image.new("L", crop.size, 0)
    mask_px = mask.load()
    for yy in range(crop.height):
        for xx in range(crop.width):
            px = pixels[xx, yy]
            if sum((int(px[k]) - target[k]) ** 2 for k in range(3)) < 75 ** 2:
                mask_px[xx, yy] = 255
    expanded = mask.filter(ImageFilter.MaxFilter(9))
    colour_layer = Image.new("RGB", crop.size, target)
    crop.paste(colour_layer, mask=expanded)
    img.paste(crop, (x0 + 12, y0 + 12))


def zoom_tile_content(img, box, zoom=1.08):
    """Uniformly enlarge one satellite without altering relative positions."""
    x0, y0, x1, y1 = box
    inset = 10
    dst = (x0 + inset, y0 + inset, x1 - inset, y1 - inset)
    width, height = dst[2] - dst[0], dst[3] - dst[1]
    crop_w, crop_h = int(width / zoom), int(height / zoom)
    cx, cy = (dst[0] + dst[2]) // 2, (dst[1] + dst[3]) // 2
    src = (cx - crop_w // 2, cy - crop_h // 2,
           cx + crop_w // 2, cy + crop_h // 2)
    enlarged = img.crop(src).resize((width, height), Image.Resampling.LANCZOS)
    img.paste(enlarged, dst[:2])


def remove_tile_frame(img, box):
    """Erase the decorative cancer-coloured rectangle around a satellite."""
    x0, y0, x1, y1 = box
    draw = ImageDraw.Draw(img)
    background = "#f5f5f5"
    width = 14
    draw.rectangle((x0, y0, x1, y0 + width), fill=background)
    draw.rectangle((x0, y1 - width, x1, y1), fill=background)
    draw.rectangle((x0, y0, x0 + width, y1), fill=background)
    draw.rectangle((x1 - width, y0, x1, y1), fill=background)


def erase_baked_main_labels(img):
    """Remove only the white-filled legacy codes from the main UMAP.

    The original plot face is #fafafa, so its text's pure-white fill is a
    reliable mask. A local blur restores the underlying smooth raster before
    the larger labels are drawn; embedding points and coordinates are not
    recomputed.
    """
    box = (175, 267, 2260, 2792)
    crop = np.asarray(img.crop(box).convert("RGB")).copy()
    white = np.all(crop >= 253, axis=2)
    yy, xx = np.ogrid[:crop.shape[0], :crop.shape[1]]
    central = (xx > 430) & (xx < 1500) & (yy > 750) & (yy < 1700)
    mask = white & central
    mask = ndimage.maximum_filter(mask, size=29)
    smooth = ndimage.gaussian_filter(crop.astype(np.float32),
                                    sigma=(12, 12, 0)).clip(0, 255).astype(np.uint8)
    crop[mask] = smooth[mask]
    img.paste(Image.fromarray(crop), box[:2])


def enlarge_main_dots(img):
    """Modestly enlarge small saturated components in the main UMAP only."""
    box = (175, 267, 2260, 2792)
    crop = np.asarray(img.crop(box).convert("RGB")).copy()
    rgb = crop.astype(np.float32) / 255.0
    high = rgb.max(axis=2)
    low = rgb.min(axis=2)
    saturation = np.divide(high - low, high, out=np.zeros_like(high), where=high > 0)
    coloured = (saturation > 0.18) & (high < 0.98)
    labels, _ = ndimage.label(coloured)
    counts = np.bincount(labels.ravel())
    small = (labels > 0) & (counts[labels] >= 3) & (counts[labels] <= 90)
    expanded = ndimage.maximum_filter(small, size=5)
    if small.any():
        _, nearest = ndimage.distance_transform_edt(~small, return_indices=True)
        propagated = crop[nearest[0], nearest[1]]
        crop[expanded] = propagated[expanded]
    img.paste(Image.fromarray(crop), box[:2])


def mute_main_hulls(img):
    """Subdue transparent confidence-hull fills without touching dot colour.

    The locked source raster contains twelve overlapping translucent ellipses.
    They form large muddy colour patches when reduced.  Enlarged, saturated
    points are retained; only pale, low-saturation fill pixels are blended
    back toward the plot face.
    """
    box = (175, 267, 2260, 2792)
    crop = np.asarray(img.crop(box).convert("RGB")).copy()
    rgb = crop.astype(np.float32) / 255.0
    high, low = rgb.max(axis=2), rgb.min(axis=2)
    sat = np.divide(high - low, high, out=np.zeros_like(high), where=high > 0)
    # A dozen fill layers also produce near-neutral grey where hues overlap.
    # Protect locally saturated dot marks first, then fade every remaining
    # mid/light fill pixel (including grey overlapping-hull regions).
    dot_core = (sat > 0.14) & (high < 0.95)
    components, _ = ndimage.label(dot_core)
    component_size = np.bincount(components.ravel())
    # Individual point clouds stay local; confidence outlines form much
    # larger connected components.  Preserve only local coloured marks.
    local_marks = ((components > 0) & (component_size[components] <= 2200))
    protected_dots = ndimage.maximum_filter(local_marks, size=7)
    pale_hull = ((sat < 0.30) & (high > 0.50) & (low > 0.30) &
                 ~protected_dots)
    face = np.array([250.0, 250.0, 250.0], dtype=np.float32)
    crop[pale_hull] = (0.16 * crop[pale_hull] + 0.84 * face).astype(np.uint8)
    img.paste(Image.fromarray(crop), box[:2])


def right_label(draw, box, text_value, font, color):
    """Place a large satellite code with a halo, but no enclosing box."""
    _, y0, x1, _ = box
    draw.text((x1 - 18, y0 + 18), text_value, font=font, fill=color,
              anchor="ra", stroke_width=5, stroke_fill="white")


def draw_cancer_key(img, font):
    """Use the cleared header for a compact, readable colour key.

    Direct labels inside the dense overview UMAP obscure points and become
    illegible after manuscript scaling.  The key makes colour encoding
    explicit without adding text over the embedding.
    """
    draw = ImageDraw.Draw(img)
    x0, dx = 135, 350
    for idx, (code, color) in enumerate(zip(CODES, COLORS)):
        row, col = divmod(idx, 6)
        x = x0 + col * dx
        y = 20 + row * 118
        draw.ellipse((x, y + 13, x + 61, y + 61), fill=color,
                     outline="white", width=2)
        draw.text((x + 76, y), code, font=font, fill="#222222")


def main():
    img = Image.open(SRC).convert("RGB")
    if img.size != (6033, 2958):
        raise ValueError(f"unexpected locked UMAP dimensions: {img.size}")
    draw = ImageDraw.Draw(img)
    white = "#ffffff"
    code_font = ImageFont.truetype(str(FONT), 108)
    key_font = ImageFont.truetype(str(FONT), 58)
    axis_font = ImageFont.truetype(str(FONT), 126)
    tick_font = ImageFont.truetype(str(FONT), 72)

    # Global and main-panel headings.  These white masks do not touch plot
    # borders or points; coordinates are locked to the audited source size.
    draw.rectangle((0, 0, img.width, 112), fill=white)
    draw.rectangle((0, 112, 2260, 262), fill=white)
    erase_baked_main_labels(img)
    enlarge_main_dots(img)
    # Preserve the original translucent confidence-hull blending.  The colour
    # fields carry the cohort-overlap structure; only direct text over points
    # is removed and replaced by the external key above.
    draw_cancer_key(img, key_font)
    draw = ImageDraw.Draw(img)

    # Remove the old headings, enlarge the focal points, then place every
    # short code in a white top-right inset inside its own frame.
    masks = [(2285, 180, 6032, 266),
             (2285, 1044, 6032, 1130),
             (2285, 1908, 6032, 1994)]
    for box in masks:
        draw.rectangle(box, fill=white)
    for i, (code, color) in enumerate(zip(CODES, COLORS)):
        col, row = i % 4, i // 4
        x0, x1 = TILE_COLUMNS[col]
        y0, y1 = TILE_ROWS[row]
        tile = (x0, y0, x1, y1)
        zoom_tile_content(img, tile)
        enlarge_focal_dots(img, tile, color)
        remove_tile_frame(img, tile)
        draw = ImageDraw.Draw(img)
        right_label(draw, tile, code, code_font, color)

    # Replace the small baked-in axis titles without touching the embedding.
    # The locked raster's main-panel tick labels were too small after the
    # manuscript-scale mosaic resize.  Redraw them with a white halo in the
    # narrow plot margins; this keeps them clear of the enlarged axis titles.
    draw.rectangle((820, 2805, 1550, 2957), fill=white)
    centered(draw, (1185, 2815), "UMAP-1", axis_font, "#111111")
    draw.rectangle((0, 1190, 150, 1870), fill=white)
    vertical = Image.new("RGBA", (500, 140), (255, 255, 255, 0))
    vdraw = ImageDraw.Draw(vertical)
    vdraw.text((250, 70), "UMAP-2", font=axis_font, fill="#111111",
               anchor="mm")
    vertical = vertical.rotate(90, expand=True, resample=Image.Resampling.BICUBIC)
    img.paste(vertical, (5, 1280), vertical)

    # Main UMAP plotting bounds in the audited source raster.  Tick labels are
    # placed just outside the axes and never in the UMAP-1/UMAP-2 title bands.
    x0, x1 = 175, 2260
    y0, y1 = 2792, 267
    # Start at 2 here: the origin tick is redrawn separately after the axes
    # are restored so it is not hidden by the shared axis corner.
    for value in range(2, 15, 2):
        x = x0 + (x1 - x0) * value / 15.0
        # Remove the original small glyphs in the outside margin only.
        draw.rectangle((x - 66, 2795, x + 66, 2880), fill=white)
        draw.text((x, 2705), str(value), font=tick_font, fill="#111111",
                  anchor="mt", stroke_width=6, stroke_fill="white")
    for value in range(2, 15, 2):
        y = y0 - (y0 - y1) * value / 15.0
        draw.rectangle((84, y - 48, 174, y + 48), fill=white)
        draw.text((166, y), str(value), font=tick_font, fill="#111111",
                  anchor="rm", stroke_width=6, stroke_fill="white")
    # Remove the original y=0 glyph so the origin is represented once by the
    # x-axis tick at the shared corner.
    draw.rectangle((84, y0 - 48, 174, y0 + 48), fill=white)
    # Also clear the baked-in x=0 glyph before restoring the axis.  It sits at
    # the shared corner in the source raster and would otherwise show through
    # beneath the relocated, readable origin label.
    draw.rectangle((x0 - 66, 2795, x0 + 66, 2880), fill=white)
    # The margin masks around the zero tick touch the plot border; restore
    # those two audited axis strokes after the redraw.
    draw.line((x0, y0, x1, y0), fill="#111111", width=7)
    draw.line((x0, y1, x0, y0), fill="#111111", width=7)
    # The broad x-margin masks also cover the prior central title; restore it
    # after the old tick glyphs have been removed.
    draw.rectangle((820, 2805, 1550, 2957), fill=white)
    centered(draw, (1185, 2815), "UMAP-1", axis_font, "#111111")
    # Put the single origin label just inside the old x-tick position, clear
    # of both axis strokes.  The y=0 label remains intentionally suppressed
    # so that the origin is represented once, not twice.
    draw.text((x0 + 54, 2705), "0", font=tick_font, fill="#111111",
              anchor="mt", stroke_width=6, stroke_fill="white")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    img.save(OUT, dpi=(400, 400), optimize=True)
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
