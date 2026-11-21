"""v7 Figure 4 (revision d, author review 16 Oct): internal test of the 413 patients.
Row 1  A multiclass radar, B mean probability matrix (v6 fig5_panels images), C decision curves (v6 code, square canvas,
       faint purple background, own legend).
Row 2  comparison with the published multiclass models (Bueno Alvez et al. 2023, Supplementary Data 6), aligned under A-C:
       D radar overlay of macro AUC, macro F1, balanced accuracy (macro recall) and minimum class recall, ours (purple fill)
         against the published 36-, 83- and 1,463-protein models;
       E per-cancer F1 matrix, rows = models (12, 26 = this study, 36, 83, 1,463 proteins), columns = cancers;
       F size-performance frontier: macro AUC and macro F1 against proteins in the model (log), ours with 95% bootstrap
         intervals (1,000 resamples).
Row 3  G the 12 disease-specific models: confusion matrix at 0.50 above the AUC/Sp/Se radar (95% intervals: bootstrap AUC,
       500 resamples, seed 52; Wilson for Se/Sp); H each panel against the published per-cancer size series (same-sex
       scoring for BRC, CVX, ENDC, OVC, PRC). G and H share the 12 cancer columns.
Output: figures/output/fig4_v7d_internal.png
"""
import importlib
import inspect

import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, NullLocator
from PIL import Image
from sklearn.metrics import roc_auc_score

import v7_style as S
plt = S.plt
Image.MAX_IMAGE_PIXELS = None
B = S.V7 / "figures/build"
BUILD = B / "fig4d"
BUILD.mkdir(parents=True, exist_ok=True)
GREY_PUB = "#555555"
SIZES = [3, 10, 50, 200, 1463]
SAME_SEX = {"BRC", "CVX", "ENDC", "OVC", "PRC"}
PUB_NAME = {"AML": "AML", "CLL": "CLL", "DLBCL": "LYMPH", "Myeloma": "MYEL", "Colorectal": "CRC", "Lung": "LUNGC",
            "Glioma": "GLIOM", "Breast": "BRC", "Cervical": "CVX", "Endometrial": "ENDC", "Ovarian": "OVC", "Prostate": "PRC"}
SOFT = mcolors.LinearSegmentedColormap.from_list("soft", ["#f6f4fa", "#ddd0ee", "#b79ad8", "#8a63bd"], N=256)
FSZ = dict(tick=15, label=17, title=18, annot=14)


def blend(col, frac):
    rgb = np.array(mcolors.to_rgba(col))[:3]
    return tuple((rgb * frac + (1 - frac)).clip(0, 1))


def wilson(x, n, z=1.959964):
    p = x / n
    d = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / d
    return p, max(0, c - h), min(1, c + h)


def boot_auc(y, s, n_boot=500, seed=52):
    rng = np.random.default_rng(seed)
    out = [roc_auc_score(y[i], s[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(n_boot)) if 0 < y[i].sum() < len(y)]
    return np.percentile(out, 2.5), np.percentile(out, 97.5)


def published():
    s6 = pd.read_excel(S.ROOT / "references/alves_supplementary_data/supp6.xlsx")
    s6["code"] = s6.Class.map(PUB_NAME)
    assert s6.code.notna().all()
    macro = s6.groupby("Num_proteins").agg(auc=("AUC", "mean"), f1=("F1", "mean"), rec=("Recall", "mean"), minrec=("Recall", "min"))
    per = s6.pivot(index="Num_proteins", columns="code", values="F1")
    return macro, per


def ours():
    ci = pd.read_csv(B / "fig4/bootstrap_ci_summary.csv").set_index("metric")
    g = lambda k: ci.loc[k, ["point_estimate", "ci_lower_2_5", "ci_upper_97_5"]].astype(float).to_numpy()
    pc = pd.read_csv(S.RES / "B1_multiclass/B1_per_cancer.csv").set_index("cancer")
    f1 = 2 * pc.precision * pc.recall / (pc.precision + pc.recall)
    return {"auc": g("macro_ovr_auc"), "f1": g("macro_f1"), "rec": g("balanced_accuracy"), "minrec": g("min_class_recall")}, f1


def save(fig, name):
    p = BUILD / name
    fig.savefig(p, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return p


def panel_d_radar(macro, us):
    fig = plt.figure(figsize=(8.6, 8.0))
    ax = fig.add_subplot(projection="polar")
    keys = [("auc", "Macro\nAUC"), ("f1", "Macro F$_1$"), ("rec", "Balanced\naccuracy"), ("minrec", "Min class\nrecall")]
    ang = np.linspace(0, 2 * np.pi, len(keys), endpoint=False).tolist()
    close = lambda v: v + v[:1]
    ax.set_theta_offset(np.pi / 2); ax.set_theta_direction(-1); ax.set_ylim(0, 1.0)
    ax.set_xticks(ang); ax.set_xticklabels([k[1] for k in keys], fontsize=FSZ["label"])
    ax.set_yticks([0.25, 0.5, 0.75, 1.0]); ax.set_yticklabels(["", "0.5", "", "1.0"], fontsize=FSZ["tick"] - 2, color=S.GREY)
    ax.grid(color="#dddddd"); ax.spines["polar"].set_color("#cccccc")
    styles = {36: ("--", "#9a9a9a", 2.2), 83: ("-", "#555555", 2.4), 1463: (":", "#222222", 2.6)}
    for n, (ls, c, lw) in styles.items():
        v = [macro.loc[n, k] for k, _ in keys]
        ax.plot(close(ang), close(v), ls=ls, color=c, lw=lw, zorder=3)
    v = [us[k][0] for k, _ in keys]
    ax.fill(close(ang), close(v), color=S.PURPLE, alpha=0.28, zorder=2)
    ax.plot(close(ang), close(v), color=S.PURPLE, lw=3.2, zorder=4)
    ax.scatter(ang, v, s=70, color=S.PURPLE, ec="white", lw=1.2, zorder=6)
    for a, (k, _) in zip(ang, keys):
        _, lo, hi = us[k]
        ax.plot([a, a], [lo, hi], color="black", lw=2.6, zorder=7)
    leg = [Patch(fc=S.PURPLE, alpha=0.5, ec=S.PURPLE, label="This study, 26"),
           Line2D([], [], ls="--", color="#9a9a9a", lw=2.2, label="Published, 36"),
           Line2D([], [], ls="-", color="#555555", lw=2.4, label="Published, 83"),
           Line2D([], [], ls=":", color="#222222", lw=2.6, label="Published, 1,463")]
    fig.legend(handles=leg, loc="lower center", ncol=2, frameon=False, fontsize=FSZ["annot"], bbox_to_anchor=(0.5, -0.03))
    fig.subplots_adjust(bottom=0.16)
    return save(fig, "d_radar.png")


def panel_e_matrix(per, f1_ours):
    order = ["AML", "BRC", "CLL", "CRC", "CVX", "ENDC", "GLIOM", "LUNGC", "LYMPH", "MYEL", "OVC", "PRC"]
    rows = [("12", per.loc[12, order]), ("26  This study", f1_ours.loc[order]), ("36", per.loc[36, order]),
            ("83", per.loc[83, order]), ("1,463", per.loc[1463, order])]
    M = np.array([r[1].to_numpy(float) for r in rows])
    fig, ax = plt.subplots(figsize=(9.2, 7.2))
    ax.imshow(M, cmap=SOFT, vmin=0.3, vmax=1.0, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=11.5,
                    color="white" if M[i, j] > 0.86 else S.INK, fontweight="bold" if i == 1 else "normal")
    ax.add_patch(plt.Rectangle((-0.5, 0.5), 12, 1, fill=False, ec=S.PURPLE_DARK, lw=3))
    ax.set_xticks(range(12)); ax.set_xticklabels([S.NAME[c] for c in order], rotation=45, ha="right", fontsize=FSZ["tick"])
    for t, c in zip(ax.get_xticklabels(), order):
        t.set_color(S.COL[c])
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows], fontsize=FSZ["tick"])
    ax.get_yticklabels()[1].set_color(S.PURPLE_DARK)
    ax.set_ylabel("Proteins in the multiclass model", fontsize=FSZ["label"])
    ax.set_title("Per-cancer F$_1$", fontsize=FSZ["title"], color=S.PURPLE, loc="left", pad=10)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    cax = ax.inset_axes([1.02, 0.1, 0.03, 0.8])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=mcolors.Normalize(0.3, 1.0), cmap=SOFT), cax=cax)
    cb.ax.tick_params(labelsize=FSZ["tick"] - 3); cb.outline.set_visible(False)
    return save(fig, "e_matrix.png")


def panel_f_frontier(macro, us):
    fig, ax = plt.subplots(figsize=(9.2, 8.0))
    S.frame(ax, fs=FSZ["tick"])
    ax.set_facecolor(S.tint("#6a3d9a", 0.07))
    n = macro.index.to_numpy()
    for k, mk, lab in (("auc", "o", "Macro AUC"), ("f1", "s", "Macro F$_1$")):
        ax.plot(n, macro[k], "-", marker=mk, color=GREY_PUB, lw=2.6, ms=10, mfc="white", mew=2.2, zorder=3)
        e, lo, hi = us[k]
        ax.errorbar([26], [e], yerr=[[e - lo], [hi - e]], fmt="D" if k == "auc" else "s", color=S.PURPLE, ms=15,
                    mec="white", mew=1.4, elinewidth=3.2, capsize=0, zorder=5)
        ax.text(1463 * 1.12, macro.loc[1463, k], lab, va="center", ha="left", fontsize=FSZ["annot"], color=S.INK)
    ax.set_xscale("log"); ax.xaxis.set_major_locator(FixedLocator([12, 26, 36, 83, 1463])); ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels(["12", "26", "36", "83", "1,463"]); ax.set_xlim(9, 5200); ax.set_ylim(0.5, 1.0)
    ax.get_xticklabels()[1].set_color(S.PURPLE_DARK)
    ax.set_xlabel("Proteins in the multiclass model (log scale)", fontsize=FSZ["label"])
    ax.set_ylabel("Internal-test value", fontsize=FSZ["label"])
    ax.grid(color="#e6e0ef"); ax.set_axisbelow(True)
    leg = [Line2D([], [], color=GREY_PUB, lw=2.6, marker="o", ms=9, mfc="white", mew=2, label="Published models"),
           Line2D([], [], ls="none", marker="D", ms=12, color=S.PURPLE, label="This study, 95% interval")]
    ax.legend(handles=leg, loc="lower right", frameon=False, fontsize=FSZ["annot"])
    return save(fig, "f_frontier.png")


def lower_gh():
    pan = pd.read_csv(S.V7 / "panels/panel_memberships_v7.csv")
    comp = pd.read_csv(S.RES / "B3_published_comparison/same_sex/table1_same_cohort_comparison.csv").set_index("cancer")
    fig = plt.figure(figsize=(32, 11.5))
    gs = fig.add_gridspec(3, 12, height_ratios=[1, 1, 0.95], hspace=0.30, wspace=0.22, left=0.045, right=0.99, top=0.92, bottom=0.14)
    angles = np.linspace(0, 2 * np.pi, 3, endpoint=False).tolist()
    first = {}
    for j, c in enumerate(S.ORDER):
        name, col = S.NAME[c], S.COL[c]
        n_prot = int((pan.panel == f"single_{name}").sum())
        d = pd.read_csv(B / f"fig5/predictions/{c.lower()}/tables/{c.lower()}_test_predictions.csv")
        y, s_, yp = d.y_true.to_numpy(), d.y_score.to_numpy(), d.y_pred.to_numpy()
        tp, fn = int(((y == 1) & (yp == 1)).sum()), int(((y == 1) & (yp == 0)).sum())
        tn, fp = int(((y == 0) & (yp == 0)).sum()), int(((y == 0) & (yp == 1)).sum())
        se, se_lo, se_hi = wilson(tp, tp + fn); sp, sp_lo, sp_hi = wilson(tn, tn + fp)
        auc = roc_auc_score(y, s_); a_lo, a_hi = boot_auc(y, s_)
        am = fig.add_subplot(gs[0, j]); am.set_box_aspect(1)
        fills = [[blend(col, 1.0), blend(col, 0.15)], [blend(col, 0.09), blend(col, 0.62)]]
        for r, row in enumerate([[tp, fn], [fp, tn]]):
            for k, v in enumerate(row):
                am.add_patch(plt.Rectangle((k, 1 - r), 1, 1, fc=fills[r][k], ec="white", lw=2.5))
                am.text(k + 0.5, 1.5 - r, str(v), ha="center", va="center", fontsize=19, fontweight="bold",
                        color="white" if (r, k) == (0, 0) else "black")
        am.set_xlim(0, 2); am.set_ylim(0, 2); am.xaxis.tick_top()
        am.set_xticks([0.5, 1.5]); am.set_xticklabels(["Pred +", "Pred −"], fontsize=12)
        am.set_yticks([0.5, 1.5]); am.set_yticklabels(["Ctrl", "Case"] if j == 0 else ["", ""], fontsize=13)
        am.tick_params(length=0, pad=2)
        for sp_ in am.spines.values():
            sp_.set_visible(False)
        am.set_title(name, color=col, fontsize=S.FS["title"] - 2, pad=30)
        ar = fig.add_subplot(gs[1, j], projection="polar")
        vals, los, his = [auc, sp, se], [a_lo, sp_lo, se_lo], [a_hi, sp_hi, se_hi]
        ar.set_theta_offset(np.pi / 2); ar.set_theta_direction(-1); ar.set_ylim(0, 1.2)
        ar.set_xticks(angles); ar.set_xticklabels([]); ar.set_yticks([0.5, 1.0]); ar.set_yticklabels([])
        ar.grid(color="#dddddd", lw=0.6); ar.spines["polar"].set_color("#cccccc")
        ar.fill(angles + angles[:1], vals + vals[:1], color=col, alpha=0.35)
        ar.plot(angles + angles[:1], vals + vals[:1], color=col, lw=2)
        for ang, lo, hi in zip(angles, los, his):
            ar.plot([ang, ang], [lo, hi], color="black", lw=2.4)
        for m, ang, v, (dx, dy), ha in zip(("AUC", "Sp", "Se"), angles, vals, ((0, 6), (8, -4), (-8, -4)), ("center", "left", "right")):
            ar.annotate(f"{m}\n{v:.2f}", xy=(ang, min(v + 0.05, 1.08)), xytext=(dx, dy), textcoords="offset points",
                        ha=ha, va="bottom" if m == "AUC" else "center", fontsize=11.5, linespacing=0.95,
                        bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=0.3))
        af = fig.add_subplot(gs[2, j]); S.frame(af, fs=12)
        r = comp.loc[name]
        pubv = [r[f"alvez_{k}p"] for k in SIZES]
        af.fill_between(SIZES, pubv, 0.55, color=S.tint(c, 0.22), zorder=1)
        af.plot(SIZES, pubv, "-o", color=GREY_PUB, lw=2.4, ms=6.5, mfc="white", mew=1.8, zorder=3)
        af.errorbar([n_prot], [r.auc], yerr=[[r.auc - r.auc_lo], [r.auc_hi - r.auc]], fmt="D", color=col, ms=11,
                    mec="white", mew=1.2, elinewidth=3, capsize=0, zorder=5)
        af.set_xscale("log"); af.xaxis.set_major_locator(FixedLocator(SIZES)); af.xaxis.set_minor_locator(NullLocator())
        af.set_xticklabels(["3", "10", "50", "200", "1.5k"], fontsize=11)
        af.set_xlim(1.8, 2600); af.set_ylim(0.55, 1.03); af.set_yticks([0.6, 0.8, 1.0])
        af.spines[["top", "right"]].set_visible(False)
        af.text(0.97, 0.06, f"{n_prot}P" + (" · same sex" if name in SAME_SEX else ""), transform=af.transAxes,
                ha="right", fontsize=11, color=col, fontweight="bold")
        if j:
            af.set_yticklabels([])
        else:
            af.set_ylabel("Test-set AUC", fontsize=S.FS["label"] - 2)
            af.set_xlabel("Proteins (log scale)", fontsize=S.FS["label"] - 3)
            first = {"G": am, "H": af}
    S.letter(fig, first["G"], "G", dx=-0.012, dy=0.045)
    S.letter(fig, first["H"], "H", dx=-0.012, dy=0.02)
    leg = [Line2D([], [], color=GREY_PUB, lw=2.6, marker="o", ms=9, mfc="white", mew=2, label="Published models of the same cohort"),
           Line2D([], [], ls="none", marker="D", ms=12, color=S.INK, label="This study, with 95% interval")]
    fig.legend(handles=leg, loc="lower center", ncol=2, frameon=False, fontsize=S.FS["legend"], bbox_to_anchor=(0.5, 0.0))
    return save(fig, "lower_GH.png")


def dca_panel():
    F = importlib.import_module("fig5_panels")      # enables v6 text scaling for figures saved afterwards
    F.TABLES = F.OUT = B / "fig4"
    src = inspect.getsource(F.panel_decision_curve)
    reps = {
        "    fig, ax = plt.subplots(figsize=(11.5, 6.2))\n    fig.subplots_adjust(left=0.14, right=0.87, top=0.94, bottom=0.22)":
            "    fig, ax = plt.subplots(figsize=(8.8, 8.2))\n    fig.subplots_adjust(left=0.17, right=0.97, top=0.96, bottom=0.36)\n"
            "    ax.set_facecolor(V7_TINT)",
        '    # One legend is drawn below the complete Figure 5 mosaic.':
            '    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, fontsize=11.5, handlelength=1.5,\n'
            '               columnspacing=0.8, bbox_to_anchor=(0.55, 0.0))  #',
        'p = OUT / f"fig5e_decision_curve.{ext}"': 'p = OUT / f"v7d_decision_curve.{ext}"',
    }
    for o, n in reps.items():
        assert src.count(o) == 1, o
        src = src.replace(o, n)
    F.V7_TINT = S.tint("#6a3d9a", 0.07)
    exec(compile(src, "panel_decision_curve (v7d)", "exec"), F.__dict__)
    F.panel_decision_curve()
    return B / "fig4/v7d_decision_curve.png"


def main():
    macro, per = published()
    us, f1_ours = ours()
    pd_, pe_, pf_ = panel_d_radar(macro, us), panel_e_matrix(per, f1_ours), panel_f_frontier(macro, us)
    gh = lower_gh()
    dca = dca_panel()
    import build_slide_mosaics as bsm
    bsm.OUT = BUILD
    spec = dict(title="", grid=(2, 3), label_band=104, col_weights=[1.0, 1.0, 1.0],
                source_paths={"radar": B / "fig4/fig5f_macro_radar.png", "matrix": B / "fig4/fig5d_score_matrix.png", "dca": dca,
                              "d_radar": pd_, "e_matrix": pe_, "f_frontier": pf_},
                panels=[("radar", 0, 0, 1, 1), ("matrix", 0, 1, 1, 1), ("dca", 0, 2, 1, 1),
                        ("d_radar", 1, 0, 1, 1), ("e_matrix", 1, 1, 1, 1), ("f_frontier", 1, 2, 1, 1)])
    bsm.build("top_AF", spec)
    top = Image.open(BUILD / "top_AF.png").convert("RGB")
    bot = Image.open(gh).convert("RGB")
    w = max(top.width, bot.width)
    top = top.resize((w, round(top.height * w / top.width)), Image.LANCZOS)
    bot = bot.resize((w, round(bot.height * w / bot.width)), Image.LANCZOS)
    gap = int(0.012 * w)
    out = Image.new("RGB", (w, top.height + gap + bot.height), "white")
    out.paste(top, (0, 0)); out.paste(bot, (0, top.height + gap))
    p = S.OUT / "fig4_v7d_internal.png"
    out.save(p, optimize=True)
    print("wrote", p, out.size)


if __name__ == "__main__":
    main()
