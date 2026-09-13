#!/usr/bin/env python3
"""
Assemble the finished PPI figures into the two actual deck slides.

New 2026-08-24, after the user settled the final placement:
  Slide 10: fig10 (locked-25, raw) + fig11 (12-panel mosaic, raw)
            -> output fig10e (naming continues the fig10/fig10b family)
  Slide 11: fig11c (pooled, coloured by panel) + fig11d (pooled, official
            STRING render) -> output fig11f (naming continues the
            fig11/fig11b/fig11c/fig11d family)
fig10b/fig11b (extended/first_shell_max5) are deliberately excluded --
confirmed backup/appendix material only, not part of the live deck.

WHITE-SPACE PASS 2026-08-24: fig10/fig11c/fig11d were independently
tightened (image height, stats-column width, and inter-element gaps all
reduced -- see each script's own changelog entry) before this composite
was rebuilt, so the per-figure white space reduction the user asked for
is upstream of this script, not something this compositing step does on
its own.

Pure image compositing (Pillow), not matplotlib -- the four source PNGs
are already finished, fully composed figures (own titles, captions,
legends baked in). This script only scales each pair to a common height
and places them side by side on a white canvas; it does not touch any
figure's own content, so nothing here can introduce a data or labelling
error that wasn't already caught when that source figure was reviewed.

SCALING PASS 2026-08-24: the first version matched both images to the
SHORTER source height, which for slide 10 meant shrinking fig11 (already
a dense 12-tile mosaic, small per-tile text) down to fig10's much shorter
native height -- made fig11 noticeably harder to read, and the resulting
canvas came out at a 2.8:1 aspect ratio, far wider than a 16:9 slide.
Switched to matching the TALLER source height instead: fig10 has a much
sparser single network plus a wide legend column, so scaling IT up to
match fig11's height doesn't cost anything readability-wise (still well
above native screen resolution), while fig11 keeps every pixel of its
own dense content at full size. Reserve judgement pair-by-pair rather
than assuming "shrink to the smaller one" is always the safe default --
it depends on which of the two images is actually content-dense.

Source: figurev5/output/{fig10_locked25_ppi_network,
    fig11_disease_panels_ppi_mosaic, fig11c_pooled12panels_network,
    fig11d_pooled12panels_official}.png
Output: figurev5/output/slide10_ppi.png, figurev5/output/slide11_ppi.png
Run:    python build_ppi_slides.py
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image

ROOT = Path("e:/Proteomics")
OUT = ROOT / "figurev5/output"

GAP_PX = 20
MARGIN_PX = 15
BG = (255, 255, 255, 255)

# Canva/slide upload target. The source figures render at DPI=400 (print
# quality) purely so the per-figure PNG/PDF stay archival; a composite of
# two of those came out 14729px wide / 8.7MB, which is pointless for a
# slide and slow to upload. A 16:9 Canva slide renders at 1920px wide, so
# 3600px is ~2x that -- text stays crisp if someone zooms or exports at
# 2x, at ~1.3-1.7MB per slide instead of 4-9MB.
#
# PNG, not JPEG, despite JPEG being ~40% smaller here: these figures are
# fine bold text on white plus thin single-pixel evidence-colour edges,
# exactly the content JPEG rings around. The size win is not worth
# artefacts on the lines the figure exists to show.
MAX_W_PX = 3600


def side_by_side(left_path: Path, right_path: Path, out_path: Path) -> None:
    left = Image.open(left_path).convert("RGBA")
    right = Image.open(right_path).convert("RGBA")

    target_h = max(left.height, right.height)
    left = left.resize((round(left.width * target_h / left.height), target_h), Image.LANCZOS)
    right = right.resize((round(right.width * target_h / right.height), target_h), Image.LANCZOS)

    canvas_w = left.width + GAP_PX + right.width + 2 * MARGIN_PX
    canvas_h = target_h + 2 * MARGIN_PX
    canvas = Image.new("RGBA", (canvas_w, canvas_h), BG)
    canvas.paste(left, (MARGIN_PX, MARGIN_PX), left)
    canvas.paste(right, (MARGIN_PX + left.width + GAP_PX, MARGIN_PX), right)

    canvas = canvas.convert("RGB")
    if canvas.width > MAX_W_PX:
        out_h = round(canvas.height * MAX_W_PX / canvas.width)
        canvas = canvas.resize((MAX_W_PX, out_h), Image.LANCZOS)

    canvas.save(out_path, optimize=True)
    mb = out_path.stat().st_size / 1048576
    print(f"  saved -> {out_path}  ({canvas.width}x{canvas.height}px, {mb:.2f} MB)")


def main() -> None:
    side_by_side(
        OUT / "fig11_disease_panels_ppi_mosaic.png",
        OUT / "fig10_locked25_ppi_network.png",
        OUT / "fig10e_slide10_ppi.png",
    )
    side_by_side(
        OUT / "fig11c_pooled12panels_network.png",
        OUT / "fig11d_pooled12panels_official.png",
        OUT / "fig11f_slide11_ppi.png",
    )


if __name__ == "__main__":
    print("Building PPI slide composites ...")
    main()
    print("Done.")
