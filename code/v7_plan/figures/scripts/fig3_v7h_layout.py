"""v7 Figure 3, layout e (17 Oct: panels from fig3_v7e_derivation.py). Layout d (author, 17 Oct): A-E (v6.x derivation mosaic, rebuilt without its small column letters and
without the "26/180 selected" box) on the left; F above G on the right with their plot areas the same width and left-
aligned; H (12 sweeps and heatmaps, 2 rows of 6) across the full width below; all letters the same size.
Run fig3_v7d_derivation.py first (F, G, H). Output: figures/output/fig3_v7d_layout.png
"""
import inspect
import json

import matplotlib.image as mpimg

import v7_style as S
from fig3_v7_derivation import inputs

plt = S.plt
import sys
STACK = True
B3D = S.V7 / ("figures/build/fig3h")
LETTER = S.FS["letter"] - 2


def top_block():
    volcano, summary, mc = inputs()
    import fig3_panels as F
    import fig3_slide3_mosaic as M
    for mod in (F, M):
        mod.VOLCANO_SRC, mod.SUPP_S1, mod.OUT = volcano, summary, B3D
        if hasattr(mod, "LOCKED_25"):
            mod.LOCKED_25 = mc
    F.PART_A_SINGLE = S.ROOT / "revise_plan/v5_seed52_common_split/part_a_single"
    src = inspect.getsource(F.draw_partb_rank_profile)
    o = "    ax.text(ax.get_xlim()[1] * 0.98, n_all - 0.5,"
    assert src.count(o) == 1
    exec(compile(src.replace(o, "    if False: ax.text(ax.get_xlim()[1] * 0.98, n_all - 0.5,"), "rank (v7d)", "exec"), F.__dict__)
    src = inspect.getsource(M.build_slide3_mosaic)
    o = "                 key.upper(),"
    assert src.count(o) == 1
    M.LETTER_POS = []
    M.WSPACE_OUTER = M.WSPACE_OUTER * 1.45          # a little more air between A-E
    exec(compile(src.replace(o, '                 (LETTER_POS.append((key.upper(), pos.x0)) or ""),'), "mosaic (v7d)", "exec"),
         M.__dict__)
    M.build_slide3_mosaic(stem="fig3_v7h_top")
    return mpimg.imread(B3D / "fig3_v7h_top.png"), list(M.LETTER_POS)


def main():
    top, letters = top_block()
    f = mpimg.imread(B3D / "f_sweep.png"); g = mpimg.imread(B3D / "fig3_6b_partb_expression_candidate.png")
    h = mpimg.imread(B3D / "fig4_7c_pairs_wide.png")
    bf = json.loads((B3D / "f_sweep.axis.json").read_text())["box"]
    bg = json.loads((B3D / "fig3_6b_partb_expression_candidate.axis.json").read_text())["box"]
    asp = lambda im: im.shape[1] / im.shape[0]
    wt = asp(top)                                     # top block: height 1
    gap = 0.03
    # F and G stacked in height 1 with equal plot-area widths: Wf * (bf1 - bf0) = Wg * (bg1 - bg0)
    k = (bg[1] - bg[0]) / (bf[1] - bf[0])             # Wf = k * Wg
    Wg = (1 - gap) / (k / asp(f) + 1 / asp(g))
    Wf = k * Wg
    xcol = wt + 0.05
    left = xcol + max(bg[0] * Wg, bf[0] * Wf)         # common left edge of the two plot areas
    xg, xf = left - bg[0] * Wg, left - bf[0] * Wf
    W = max(xg + Wg, xf + Wf)
    hh = W / asp(h)
    hh_cap = 0.40 / 0.60 * (0.035 + 1 + 0.015)        # H at most 40% of the figure height
    wh = W
    if hh > hh_cap * 1.08:                            # safety only; H is rendered flat enough upstream
        hh, wh = hh_cap, hh_cap * asp(h)
    T = 0.035                                          # letter band above A-E, F
    Ht = T + 1 + 0.015 + hh
    fig = plt.figure(figsize=(36.0, 36.0 * Ht / W))

    def place(img, x, y, w, hgt):
        ax = fig.add_axes([x / W, 1 - (y + hgt) / Ht, w / W, hgt / Ht])
        ax.imshow(img, interpolation="lanczos"); ax.set_axis_off()

    def letter(s, x, y):
        fig.text(x / W, 1 - y / Ht, s, fontsize=LETTER, fontweight="bold", ha="left", va="top")

    place(top, 0, T, wt, 1)
    for s, fx in letters:
        letter(s, fx * wt - 0.03, 0.0)
    hf, hg = Wf / asp(f), Wg / asp(g)
    place(f, xf, T, Wf, hf); letter("F", xcol - 0.03, 0.0)
    place(g, xg, T + hf + gap, Wg, hg); letter("G", xcol - 0.03, T + hf + gap - 0.02)
    place(h, (W - wh) / 2, T + 1.015, wh, hh); letter("H", (W - wh) / 2 - 0.005, T + 1.0)
    S.save(fig, "fig3_v7h_layout.png")


if __name__ == "__main__":
    main()
