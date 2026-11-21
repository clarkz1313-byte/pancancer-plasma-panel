"""v7 Supplementary Figure, revision d (author, 17 Oct): revision b (the author's backup) with only two changes: panel F
without the full-vs-assay strip, and black category labels in panel C. Revision b: protein
interactions and functional annotation, in two sections. Disease-specific panels (white): (A) STRING network of each
panel, (B) the 81 proteins pooled, node colour = panel(s), dark red = edges between panels, (C) functional categories
(UniProt). 26-protein list (faint purple band, the list's colour): (D) STRING network, (E) chord of the list's
proteins and the annotation terms they share (top five terms per database by over-representation P against the
measured proteins, at least two list proteins, identical protein sets merged; annotation, not enrichment; protein
colour = functional category as in C, term and chord colour = database), (F) over-representation against the measured
proteins (top) and, for the same terms, the full annotation universe (bottom).
Query-only STRING networks. Panels A-D and F are the intermediates of figS_v7_ppi.py and figS_v7_pathway.py (run
those first); E is drawn here from results/B9_biology/ora/ora_terms_with_panel_overlap.csv.
Output: figures/output/figS_v7d_biology_annotation.png
"""
import re
import textwrap

import matplotlib.image as mpimg
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch, PathPatch
from matplotlib.path import Path as MPath

import v7_style as S
from figS_v7_pathway import V7_FUNC, v5_func

plt = S.plt
B = S.V7 / "figures/build"
BUILD = B / "figS_bioann"
P = {"A": B / "figS_ppi/fig11_disease_panels_ppi_mosaic_lean.png",
     "B": B / "figS_ppi/fig11c_pooled12panels_network_lean.png",
     "C": B / "figS_pathway_c/figS_v7_parta_sankey.png",
     "D": B / "figS_ppi/fig10_locked25_ppi_network_lean.png",
     "E": BUILD / "chord_26.png",
     "F": B / "figS_pathway_c/fig6_pathway_enrichment.png"}
STRING_EV = [("#e8384f", "fusion"), ("#4a9c3f", "neighbourhood"), ("#4f5fd0", "co-occurrence"),   # as v6 fig10_5_ppi_combined
             ("#a12ec4", "experiments"), ("#d9a520", "text mining"), ("#3fc4e8", "databases"), ("#111111", "co-expression")]
DB = {"GO_BP_2023": ("GO", S.GO_COLOR), "KEGG_2021_Human": ("KEGG", "#c4761c"), "Reactome_2022": ("Reactome", "#2e7d5b")}


# 26-list proteins absent from the v6 Sankey category map (assigned from UniProt function, as V7_FUNC)
MC_FUNC = {"CNTN1": "Neural/Struct.", "CCDC80": "ECM/Adhesion", "ADAMTS15": "Enzyme/Protease"}


def _block(name):
    src = (S.ROOT / "figurev6/scripts/fig6_parta_sankey_bubble.py").read_text(encoding="utf-8")
    return src[src.index(name + " = {"):src.index("}", src.index(name + " = {")) + 1]


def func_colours():
    return dict(re.findall(r'"([^"]+)":\s*"(#[0-9a-fA-F]{6})"', _block("FUNC_COL")))


def func_map():
    return {**dict(re.findall(r'"([A-Z0-9]+)":\s*"([^"]+)"', _block("FUNC"))), **v5_func(), **V7_FUNC, **MC_FUNC}


def short(term):
    t = re.sub(r"\s*\(GO:\d+\)|\s*R-HSA-\d+", "", term).strip()
    return textwrap.fill(t, 20)


def chord():
    """Proteins on the upper arc, terms on the lower arc, one curve per protein-term membership."""
    d = pd.read_csv(S.RES / "B9_biology/ora/ora_terms_with_panel_overlap.csv")
    d = d[(d.panel == "multiclass_26") & (d.overlap_count >= 2)].sort_values("p_value")
    d = d.drop_duplicates(["database", "overlap_genes"])
    terms = pd.concat([d[d.database == k].head(5) for k in DB]).reset_index(drop=True)
    func = func_map()
    fcol = func_colours()
    order = list(fcol)
    prots = sorted({p for g in terms.overlap_genes for p in g.split(";")}, key=lambda p: (order.index(func[p]), p))
    fig, ax = plt.subplots(figsize=(13, 13))
    ax.set_aspect("equal"); ax.set_xlim(-1.9, 1.9); ax.set_ylim(-2.25, 1.75); ax.axis("off")
    R0, Rd = 1.0, 1.07
    tp = np.radians(np.linspace(165, 15, len(prots)))
    tt = np.radians(np.linspace(192, 348, len(terms)))
    pos_p = dict(zip(prots, tp))
    for i, r in terms.iterrows():
        lab, c = DB[r.database]
        for p in r.overlap_genes.split(";"):
            a, b = pos_p[p], tt[i]
            path = MPath([(R0 * np.cos(a), R0 * np.sin(a)), (0, 0), (R0 * np.cos(b), R0 * np.sin(b))],
                         [MPath.MOVETO, MPath.CURVE3, MPath.CURVE3])
            ax.add_patch(PathPatch(path, fc="none", ec=c, lw=2.6, alpha=0.55, zorder=2))
    for p, a in pos_p.items():
        col = fcol[func[p]]
        ax.scatter(Rd * np.cos(a), Rd * np.sin(a), s=260, color=col, edgecolor="white", lw=1.5, zorder=4)
        deg = np.degrees(a)
        ax.text(1.16 * np.cos(a), 1.16 * np.sin(a), p, rotation=deg if np.cos(a) >= 0 else deg - 180,
                ha="left" if np.cos(a) >= 0 else "right", va="center", rotation_mode="anchor", fontsize=16, color=col)
    for i, r in terms.iterrows():
        lab, c = DB[r.database]
        a = tt[i]
        ax.scatter(Rd * np.cos(a), Rd * np.sin(a), s=260, marker="s", color=c, edgecolor="white", lw=1.5, zorder=4)
        deg = np.degrees(a)
        ax.text(1.16 * np.cos(a), 1.16 * np.sin(a), short(r.term), rotation=deg if np.cos(a) >= 0 else deg - 180,
                ha="left" if np.cos(a) >= 0 else "right", va="center", rotation_mode="anchor", fontsize=11.5,
                color=c, linespacing=0.92)
    h = [Line2D([], [], ls="none", marker="s", ms=13, color=c, label=l) for l, c in DB.values()]
    ax.legend(handles=h, loc="upper center", ncol=3, frameon=False, fontsize=16, bbox_to_anchor=(0.5, 0.0))
    BUILD.mkdir(parents=True, exist_ok=True)
    fig.savefig(P["E"], dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("chord:", len(prots), "proteins,", len(terms), "terms")


def show(fig, rect, key, note=None):
    ax = fig.add_axes(rect)
    ax.imshow(mpimg.imread(P[key]), interpolation="lanczos")
    ax.set_axis_off()
    S.letter(fig, ax, key, dx=-0.012, dy=0.004)
    if note:
        ax.text(0.5, -0.02, note, transform=ax.transAxes, ha="center", va="top", fontsize=S.FS["annot"] + 2, color=S.INK)
    return ax


def main():
    chord()
    st = pd.read_csv(S.RES / "B9_biology/string/ppi_reaudit_combined_results_v2.csv").set_index("panel")
    edges = {p: f"{int(st.at[p, 'official_number_of_edges'])} observed · "
                f"{st.at[p, 'official_expected_number_of_edges']:.0f} expected edges" for p in ("LOCKED25", "POOLED12")}
    asp = {k: (lambda im: im.shape[1] / im.shape[0])(mpimg.imread(v)) for k, v in P.items()}
    W, m, gap, head = 24.0, 0.035, 0.025, 0.03
    usable = 1 - 2 * m
    # heights in inches, then figure height
    hA = usable * W / asp["A"]
    h2 = (usable - gap) * W / (asp["B"] + asp["C"])
    h3 = (usable - gap) * W / (asp["D"] + asp["E"])
    hF = 0.86 * W / asp["F"]
    Hh = hA + h2 + h3 + hF + 7.0
    fig = plt.figure(figsize=(W, Hh))
    y = 1 - 1.1 / Hh
    fig.text(m, y + 0.4 / Hh, "Disease-specific panels", fontsize=S.FS["group"], color=S.INK, va="bottom")
    y -= hA / Hh
    show(fig, [m, y, usable, hA / Hh], "A")
    y -= 0.55 / Hh + h2 / Hh
    x = m
    for k, note in (("B", edges["POOLED12"]), ("C", None)):
        w = h2 * asp[k] / W
        show(fig, [x, y, w, h2 / Hh], k, note)
        x += w + gap
    # purple section
    y_band_top = y - 1.0 / Hh
    y = y_band_top - 1.5 / Hh - h3 / Hh
    x = m
    for k, note in (("D", edges["LOCKED25"]), ("E", None)):
        w = h3 * asp[k] / W
        show(fig, [x, y, w, h3 / Hh], k, note)
        x += w + gap
    y -= 0.7 / Hh + hF / Hh
    show(fig, [(1 - 0.86) / 2, y, 0.86, hF / Hh], "F")
    y_band_bot = y - 0.25 / Hh
    fig.patches.append(FancyBboxPatch((0.012, y_band_bot), 0.976, y_band_top - y_band_bot,
                                      boxstyle="round,pad=0,rounding_size=0.004", transform=fig.transFigure,
                                      facecolor=S.tint(S.PURPLE, 0.07), edgecolor=S.PURPLE_LIGHT, lw=1.6, zorder=-5))
    fig.text(m, y_band_top - 0.7 / Hh, "26-protein list", fontsize=S.FS["group"], color=S.PURPLE, va="bottom")
    h1 = [Line2D([], [], color=c, lw=4, label=l) for c, l in STRING_EV]
    h2_ = [Patch(fc=S.COL[c], label=S.NAME[c]) for c in S.ORDER]
    h3_ = [Line2D([], [], color="#8b1a1a", lw=3, label="between panels"), Line2D([], [], color="#9a9a9a", lw=3, label="within a panel")]
    kw = dict(frameon=False, fontsize=S.FS["legend"] - 2, title_fontsize=S.FS["legend"] - 2, handlelength=1.6, columnspacing=1.2)
    fig.legend(handles=h1, loc="lower left", bbox_to_anchor=(0.03, 0.003), ncol=4, title="STRING evidence (A, D)", **kw)
    fig.legend(handles=h2_ + h3_, loc="lower right", bbox_to_anchor=(0.985, 0.003), ncol=7, title="Panel and edge (B)", **kw)
    S.save(fig, "figS_v7d_biology_annotation.png")


if __name__ == "__main__":
    main()
