"""v7 Figure 5, revision g (18 Oct): sensitivity row taller, square tiles unframed, single 0 at H calibration
origin. Earlier: revision e (17 Oct): violins in A, right band margin 10% smaller than left, H letter level with
A, sensitivity value and n moved off the squares. Earlier: revision c (17 Oct): larger panel-H band covering its labels, H unit labels for D and E.
Revision b (author, 17 Oct): repeated labels only on the AML column (other cancer columns show
numbers only; decision-curve axis keeps 0.5); the 26-protein column is panel H with its own axis labels.
Revision a: single-cancer tissue tests as one 13-column mosaic.

Columns 1-12 are the disease-specific lists (one per cancer, fixed threshold 0.80). Column 13, set apart by a gap
and a faint purple band, summarises the 26-protein multiclass list over all 12 tissue sets (fixed threshold 0.50):
pooled scores, macro-averaged ROC, pooled calls, macro-mean specificity and sensitivity, mean net benefit and pooled
calibration. Rows: A scores, B ROC, C calls, D specificity (ring), E sensitivity (100-square array), F decision
curves, G calibration. The likelihood-ratio row of the earlier mosaics is dropped; the sensitivity row has no target line
(author, 17 Oct).

Tiles are drawn by the unchanged figurev6 renderers (fig89_common); the grid, the 13th column and the legend are
built here, so figurev6/scripts/fig89_stacked_mosaic.py is not modified. Inputs are the B6 predictions read by
fig8_v7_external_tissue.preds(). Output: figures/output/fig5_v7g_tissue_mosaic13.png (+ build/fig5t/*.pdf)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch
from sklearn.metrics import auc, roc_curve

import v7_style as S
import inspect as _insp
import fig89_stacked_mosaic as M   # same text scaling and spacing constants as the v6 mosaics
from fig8_v7_external_tissue import preds
import fig89_common as FCM
_src = _insp.getsource(FCM._ring_gauge_ax).replace('    N_BANDS = 6', '    N_BANDS = 1').replace('shade_frac = 0.45 + 0.55 * (i / (N_BANDS - 1))', 'shade_frac = 1.0')
exec(compile(_src, 'ring (solid)', 'exec'), FCM.__dict__)
_src = _insp.getsource(FCM._icon_array_ax).replace('sp.set_color(_SP["color"]); sp.set_linewidth(_SP["lw"])', 'sp.set_visible(False)').replace('grid_h + 0.78, f"{val:.2f}"', 'grid_h + 1.45, f"{val:.2f}"').replace('ax.text(grid_w / 2, -0.76, f"n={n}"', 'ax.text(grid_w / 2, -1.30, f"n={n}"').replace('ax.set_ylim(-1.34, grid_h + (1.95 if show_label else 1.82))', 'ax.set_ylim(-3.1, grid_h + 3.6)')
exec(compile(_src, 'icons (spaced)', 'exec'), FCM.__dict__)
_PATCHED = True
from fig89_common import (CANCER_ORDER, PLOT_COLOR, load_preds, _wilson_ci, _dca_curve, _calib_curve, _box_ax,
                          _roc_ax, _cm_ax, _cm_macro_ax, _icon_array_ax, _ring_gauge_ax, _dca_ax, _calib_tile_ax,
                          _SP, FALSE_ALARM_COLOR)

BUILD = S.V7 / "figures/build/fig5t"
STEM = "fig5_v7g_tissue_mosaic13"
THR_DS, THR_MC, TARGET = 0.80, 0.50, 0.90
MC = S.PURPLE                     # 26-protein list, as in Fig 3 and Fig 4
BAND = S.tint(S.PURPLE, 0.12)     # faint purple band behind column 13
ROWS = ["box", "roc", "cm", "spec", "sens", "dca", "calib"]
ROW_H = dict(box=2.75, roc=1.90, cm=2.00, spec=2.15, sens=3.0, dca=1.90, calib=1.90)
GAP = 0.85                        # spacer column between the 12 cancers and column 13 (in tile widths)
MC_LABEL = "26-protein"


def counts(pdata, thr):
    y = pdata["true_label"].values
    yp = (pdata["prob"].values >= thr).astype(int)
    return dict(pdata=pdata, prev=float(y.mean()),
                tp=int(((y == 1) & (yp == 1)).sum()), fn=int(((y == 1) & (yp == 0)).sum()),
                tn=int(((y == 0) & (yp == 0)).sum()), fp=int(((y == 0) & (yp == 1)).sum()))


def no_target_line(ax, key):
    """Sensitivity row (author, 17 Oct): no 0.90 target line; the icon array draws it as a dashed axhline."""
    if key == "sens":
        for ln in list(ax.lines):
            if ln.get_linestyle() == "--":
                ln.remove()


def slim_and_label(fig, ROWS, last):
    """Columns 2-12: numbers only (labels stay on AML). Column 13 (panel H, the 26-protein list): its own axis labels."""
    key = {i: k for i, k in enumerate(ROWS)}
    for ax in fig.axes:
        ss = ax.get_subplotspec()
        if ss is None:
            continue
        r, c = ss.rowspan.start, ss.colspan.start
        k = key[r]
        if c == 0 and k == "calib":
            for t in ax.texts:
                t.set_fontsize(15)
        if 0 < c < 12:
            if k == "box":
                ax.set_xticklabels([]); ax.set_yticklabels([])
            elif k == "cm":
                ax.set_xticklabels(["", ""])
            elif k == "dca":
                ax.set_xticks([0.5])
            for t in ax.texts:
                txt = t.get_text()
                for pre in ("AUC: ", "n=", "ECE=", "Brier="):
                    if txt.startswith(pre):
                        t.set_text(txt[len(pre):])
        elif c == last:
            fs = 13
            if k == "box":
                ax.set_yticklabels(["0", "0.5", "1"]); ax.set_ylabel("P(cancer)", fontsize=fs, fontweight="bold")
            elif k == "roc":
                ax.set_xticklabels(["0", "1"]); ax.set_yticklabels(["0", "0.5", "1"])
                ax.set_ylabel("Sensitivity", fontsize=fs, fontweight="bold"); ax.set_xlabel("1−Spec", fontsize=fs, fontweight="bold")
            elif k == "cm":
                ax.set_yticks([0.5, 1.5]); ax.set_yticklabels(["Control", "Cancer"], fontsize=14, fontweight="bold")
            elif k == "dca":
                yt = [v for v in ax.get_yticks() if ax.get_ylim()[0] <= v <= ax.get_ylim()[1]]
                ax.set_yticks(yt); ax.set_yticklabels([f"{v:g}" for v in yt], fontsize=13)
                ax.set_ylabel("Net benefit", fontsize=fs, fontweight="bold"); ax.set_xlabel("Threshold", fontsize=fs, fontweight="bold")
            elif k == "calib":
                for t in ax.texts:
                    t.set_fontsize(15)
                ax.set_xticks([0, 0.5, 1]); ax.set_xticklabels(["", "0.5", "1"]); ax.set_yticklabels(["0", "0.5", "1"])
                ax.set_ylabel("Observed rate", fontsize=fs, fontweight="bold"); ax.set_xlabel("Predicted", fontsize=fs, fontweight="bold")


def _violin_ax(ax, pdata, cancer, col, first, show_label=True, compact=True):
    """Row A as violins: cases (cancer colour) and controls (light), median as a white bar, 0.5 cutoff dashed."""
    cases = pdata[pdata["true_label"] == 1]["prob"].values
    ctrls = pdata[pdata["true_label"] == 0]["prob"].values
    parts = ax.violinplot([cases, ctrls], positions=[0, 1], widths=0.82, showextrema=False, bw_method=0.25)
    for body, fc in zip(parts["bodies"], (col, FCM._blend(col, 0.30))):
        body.set_facecolor(fc); body.set_edgecolor(col); body.set_linewidth(1.4); body.set_alpha(1.0)
    for x, v in ((0, cases), (1, ctrls)):
        m = float(np.median(v)); q1, q3 = np.percentile(v, [25, 75])
        ax.plot([x, x], [q1, q3], color="#222222", lw=3.2, zorder=4, solid_capstyle="butt")
        ax.scatter([x], [m], s=46, color="white", edgecolor="#222222", lw=1.4, zorder=5)
    ax.axhline(0.5, color="#555555", lw=2.2, ls="--", zorder=1)
    ax.set_xlim(-0.6, 1.6); ax.set_ylim(-0.04, 1.08)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Case", "Ctrl"], fontsize=12, fontweight="bold")
    ax.set_yticks([0, 0.5, 1]); ax.set_yticklabels(["0", "0.5", "1"], fontsize=13, fontweight="bold")
    ax.tick_params(length=3, pad=2)
    if first:
        ax.set_ylabel("P(cancer)", fontsize=13, fontweight="bold")
    for sp in ax.spines.values():
        sp.set_color(_SP["color"]); sp.set_linewidth(_SP["lw"])
    if show_label:
        ax.set_title(FCM.TILE_LABEL.get(cancer, cancer), color=col, fontsize=18, fontweight="bold", pad=4)


def relabel_n(ax, text):
    for t in ax.texts:
        if t.get_text().startswith("n="):
            t.set_text(text)


def main():
    BUILD.mkdir(parents=True, exist_ok=True)
    np.random.seed(20261016)   # box-plot jitter
    ds = {c: counts(load_preds(p), THR_DS) for c, p in preds("disease_specific").items()}
    mc = {c: counts(load_preds(p), THR_MC) for c, p in preds("multiclass_26").items()}

    heights = [ROW_H[k] for k in ROWS]
    n_cols = 12 + GAP + 1
    RPAD = 1.25                      # room for a symmetric band around H
    fig_w = M.LEFT_PAD + n_cols * M.COL_W + RPAD
    fig_h = M.TOP_PAD + sum(heights) + M.HSPACE * np.mean(heights) * (len(ROWS) - 1) + M.BOTTOM_PAD
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(len(ROWS), 14, width_ratios=[1] * 12 + [GAP, 1], height_ratios=heights,
                          left=M.LEFT_PAD / fig_w, right=1 - RPAD / fig_w,
                          top=1 - M.TOP_PAD / fig_h, bottom=M.BOTTOM_PAD / fig_h, wspace=M.WSPACE, hspace=M.HSPACE)
    R = {k: i for i, k in enumerate(ROWS)}
    last = 13

    # faint purple band behind column 13, drawn first so the tiles sit on it
    top = gs[0, last].get_position(fig); bot = gs[-1, last].get_position(fig)
    pad_l, pad_r, pad_top, pad_bot = 0.036, 0.0292, 0.030, 0.034     # band covers H's own axis labels
    pad_x = pad_l
    fig.patches.append(FancyBboxPatch((top.x0 - pad_l, bot.y0 - pad_bot), top.x1 - top.x0 + pad_l + pad_r,
                                      top.y1 - bot.y0 + pad_top + pad_bot, boxstyle="round,pad=0,rounding_size=0.006",
                                      transform=fig.transFigure, facecolor=BAND, edgecolor=S.PURPLE_LIGHT,
                                      linewidth=1.4, zorder=-5))

    pooled_mc = pd.concat([mc[c]["pdata"] for c in CANCER_ORDER])

    # A — scores
    for i, c in enumerate(CANCER_ORDER):
        _violin_ax(fig.add_subplot(gs[R["box"], i]), ds[c]["pdata"], c, PLOT_COLOR[c], first=(i == 0),
                show_label=True, compact=True)
    ax = fig.add_subplot(gs[R["box"], last])
    _violin_ax(ax, pooled_mc, MC_LABEL, MC, first=False, show_label=True, compact=True)
    ax.axhline(THR_MC, color=MC, lw=1.6, ls=":", zorder=1)

    # B — ROC
    for i, c in enumerate(CANCER_ORDER):
        ax = fig.add_subplot(gs[R["roc"], i])
        _roc_ax(ax, ds[c]["pdata"], c, PLOT_COLOR[c], first=(i == 0), show_label=False)
        if i > 0:
            ax.set_xlabel(""); ax.set_xticklabels([])
    ax = fig.add_subplot(gs[R["roc"], last])
    grid = np.linspace(0, 1, 200)
    tprs = []
    for c in CANCER_ORDER:
        fpr, tpr, _ = roc_curve(mc[c]["pdata"]["true_label"], mc[c]["pdata"]["prob"])
        tprs.append(np.interp(grid, fpr, tpr))
    mtpr = np.mean(tprs, axis=0)
    ax.fill_between(grid, mtpr, alpha=0.15, color=MC)
    ax.plot(grid, mtpr, color=MC, lw=3.2)
    ax.plot([0, 1], [0, 1], color="#bbbbbb", lw=1.1, ls="--")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_xticks([0, 1]); ax.set_xticklabels([]); ax.set_yticks([0, 0.5, 1]); ax.set_yticklabels([])
    ax.tick_params(labelsize=14, length=3, pad=2)
    ax.text(0.97, 0.05, f"AUC: {auc(grid, mtpr):.2f}", ha="right", va="bottom", transform=ax.transAxes,
            fontsize=18, fontweight="bold", color="black",
            bbox=dict(boxstyle="round,pad=0.38", facecolor="white", edgecolor="black", linewidth=1.5, alpha=0.92))
    for sp in ax.spines.values():
        sp.set_color(_SP["color"]); sp.set_linewidth(_SP["lw"])

    # C — calls at the fixed threshold
    for i, c in enumerate(CANCER_ORDER):
        _cm_ax(fig.add_subplot(gs[R["cm"], i]), ds[c]["pdata"], c, PLOT_COLOR[c], THR_DS, first=(i == 0),
               show_label=False)
    ax = fig.add_subplot(gs[R["cm"], last])
    _cm_macro_ax(ax, *(sum(mc[c][k] for c in CANCER_ORDER) for k in ("tp", "fn", "tn", "fp")))
    ax.set_title("")
    for t in ax.texts:   # pooled counts run to four digits; shrink them to stay inside the cells
        if t.get_text().isdigit():
            t.set_fontsize(t.get_fontsize() * 0.74)

    # D — specificity (ring), E — sensitivity (100 squares); column 13 = macro mean over the 12 sets
    for key, row in (("spec", R["spec"]), ("sens", R["sens"])):
        vals = []
        for i, c in enumerate(CANCER_ORDER):
            d = ds[c]
            x, n = (d["tn"], d["tn"] + d["fp"]) if key == "spec" else (d["tp"], d["tp"] + d["fn"])
            v, lo, hi = _wilson_ci(x, n)
            draw = _ring_gauge_ax if key == "spec" else _icon_array_ax
            gax = fig.add_subplot(gs[row, i])
            draw(gax, c, PLOT_COLOR[c], v, lo, hi, n, TARGET if key == "spec" else 0.0, show_label=False)
            no_target_line(gax, key)
        for c in CANCER_ORDER:
            d = mc[c]
            x, n = (d["tn"], d["tn"] + d["fp"]) if key == "spec" else (d["tp"], d["tp"] + d["fn"])
            vals.append(x / n)
        mv = float(np.mean(vals))
        ax = fig.add_subplot(gs[row, last])
        (_ring_gauge_ax if key == "spec" else _icon_array_ax)(ax, MC_LABEL, MC, mv, mv, mv, 12,
                                                             TARGET if key == "spec" else 0.0, show_label=False)
        no_target_line(ax, key)
        relabel_n(ax, "12 sets")

    # F — decision curves (one shared y-scale across all 13 tiles)
    pt = np.linspace(0.05, 0.85, 150)
    nb_ds = {c: _dca_curve(ds[c]["pdata"], pt) for c in CANCER_ORDER}
    nb_mc = np.mean([_dca_curve(mc[c]["pdata"], pt) for c in CANCER_ORDER], axis=0)
    ylim = (-0.3, max(max(nb.max() for nb in nb_ds.values()), nb_mc.max()) * 1.15)
    for i, c in enumerate(CANCER_ORDER):
        ax = fig.add_subplot(gs[R["dca"], i])
        _dca_ax(ax, c, PLOT_COLOR[c], pt, nb_ds[c], ds[c]["prev"], "area", ylim, first=(i == 0), show_label=False)
        if i > 0:
            ax.set_xlabel("")
    ax = fig.add_subplot(gs[R["dca"], last])
    _dca_ax(ax, "Macro", MC, pt, nb_mc, float(np.mean([mc[c]["prev"] for c in CANCER_ORDER])), "area", ylim,
            first=False, show_label=False)
    ax.set_xlabel("")

    # G — calibration
    for i, c in enumerate(CANCER_ORDER):
        ax = fig.add_subplot(gs[R["calib"], i])
        grp, ece, brier = _calib_curve(ds[c]["pdata"])
        _calib_tile_ax(ax, c, PLOT_COLOR[c], grp, ece, brier, first=(i == 0), show_label=False)
        if i > 0:
            ax.set_xlabel(""); ax.set_xticklabels([])
        else:
            ax.set_xticks([0.5, 1.0]); ax.set_xticklabels(["0.5", "1.0"])
    ax = fig.add_subplot(gs[R["calib"], last])
    grp, ece, brier = _calib_curve(pooled_mc, n_bins=8)
    _calib_tile_ax(ax, "Macro", MC, grp, ece, brier, first=False, show_label=False)
    ax.set_xlabel(""); ax.set_xticklabels([])

    slim_and_label(fig, ROWS, last)
    fig.text(top.x0 - pad_x - 0.004, gs[0, 0].get_position(fig).y1, "H", fontsize=21, fontweight="bold",
             color="#222222", ha="right", va="top")      # same height as A
    for k, unit in (("spec", "specificity"), ("sens", "sensitivity")):   # H's own unit labels for rows D and E
        pos = gs[ROWS.index(k), last].get_position(fig)
        fig.text(pos.x0 - 0.012, (pos.y0 + pos.y1) / 2, unit, rotation=90, ha="center", va="center", fontsize=14,
                 fontweight="bold", color="#222222")
    # row letters, with the unit for the two gauge rows
    for r, k in enumerate(ROWS):
        pos = gs[r, 0].get_position(fig)
        unit = {"spec": "specificity", "sens": "sensitivity"}.get(k)
        M._row_label(fig, pos.y1, "ABCDEFG"[r], unit=unit, y_bottom=pos.y0)

    # one legend
    H = [plt.Rectangle((0, 0), 1, 1, fc=PLOT_COLOR["CRC"], ec="white", label="disease-specific list (cutoff 0.80)"),
         plt.Rectangle((0, 0), 1, 1, fc=MC, ec="white", label="26-protein list, 12 sets combined (cutoff 0.50)"),
         plt.Rectangle((0, 0), 1, 1, fc=FALSE_ALARM_COLOR, ec="white", label="false alarm / missed case"),
         plt.Line2D([0], [0], color="black", lw=2.6, label="95% CI"),
         plt.Line2D([0], [0], color="#ff006e", lw=2.6, ls="--", label=f"{TARGET:.2f} specificity target"),
         plt.Line2D([0], [0], color="#999999", lw=1.4, ls=":", label="treat all"),
         plt.Line2D([0], [0], color="#555555", lw=1.4, ls="--", label="treat none")]
    fig.legend(handles=H, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.004), fontsize=12, framealpha=0.94,
               columnspacing=1.6, handletextpad=0.6)

    fig.savefig(BUILD / f"{STEM}.pdf", facecolor="white")
    fig.savefig(S.OUT / f"{STEM}.png", dpi=300, facecolor="white")
    plt.close(fig)
    print("wrote", S.OUT / f"{STEM}.png", f"({fig_w:.1f} x {fig_h:.1f} in)")
    for c in CANCER_ORDER:
        d = ds[c]
        print(f"  {S.NAME[c]:6s} spec {d['tn']/(d['tn']+d['fp']):.2f}  sens {d['tp']/(d['tp']+d['fn']):.2f}")


if __name__ == "__main__":
    main()
