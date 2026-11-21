"""v7 Figure 8, revision j (18 Oct): chord E 25% taller than network D. Earlier: revision h (17 Oct): chord rotated (proteins left, terms right); D, E, F on the purple band
colour, D without its frame; D and E at the same height. Revision g (17 Oct): E wider than D, larger chord labels, no edge-count notes, vertical colour
bars in F. Revision f (author, 17 Oct): protein interactions and functional annotation, re-laid out for reading.
Disease-specific panels: (A) STRING network of each panel, 3 rows of 4 (tiles cut from the per-panel render);
(B) the 81 proteins pooled (node colour = panel; dark red = edges between panels); (C) functional categories as a
top-to-bottom flow, cancers above and categories below (category labels black).
26-protein list (purple band): (D) STRING network; (E) chord of list proteins and shared annotation terms (short term
names, database legend moved to the common legend); (F) over-representation, three databases side by side with short
term names. Query-only STRING networks; annotation, not enrichment, in C and E.
Inputs: figS_v7_ppi.py intermediates (A, B, D); figS_v7d_biology_annotation.chord (E, relabelled here);
figS_v7c_pathway.py source with short labels (F). Output: figures/output/fig8_v7f_biology_annotation.png
"""
import inspect
import re
import textwrap

import matplotlib.image as mpimg
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch

import v7_style as S
import figS_v7d_biology_annotation as A8

plt = S.plt
B = S.V7 / "figures/build"
BUILD = B / "fig8j"
BUILD.mkdir(parents=True, exist_ok=True)
DBCOL = {"GO": S.GO_COLOR, "KEGG": "#c4761c", "Reactome": "#2e7d5b"}
ABBR = [("Regulation Of ", "Reg. "), ("Regulation of ", "Reg. "), ("Differentiation", "diff."), ("differentiation", "diff."),
        ("Organization", "org."), ("organization", "org."), ("Metabolism Of ", "Metab. of "), ("metabolism", "metab."),
        ("Protein-Containing Complex Assembly", "complex assembly"), ("Cellular Component Biogenesis", "component biogenesis"),
        ("Programmed Cell Death", "cell death"), ("Signaling By ", ""), ("Signaling by ", ""), ("Amyotrophic lateral sclerosis", "ALS"),
        ("Epstein-Barr virus infection", "EBV infection"), ("Hematopoietic cell lineage", "Haematopoietic lineage"),
        ("Pathways in cancer", "Cancer pathways"), ("MAPK signaling pathway", "MAPK signalling"),
        ("Defective B3GALTL Causes PpS", "B3GALTL / PpS"), ("RAF/MAP Kinase Cascade", "RAF/MAP kinase"),
        ("Developmental Biology", "Development"), ("Lymphoid Progenitor Cell", "Lymphoid progenitor"),
        ("Intermediate Filament", "Intermediate filament")]


def abbreviate(t, width=16):
    t = re.sub(r"\s*\(GO:\d+\)|\s*R-HSA-\d+", "", str(t)).replace("\n", " ").strip()
    for a, b in ABBR:
        t = t.replace(a, b)
    return textwrap.fill(t[0].upper() + t[1:], width, break_long_words=False)


def trim(a, pad=8):
    ys, xs = np.nonzero((a[..., :3] < 0.96).any(axis=2))
    return a[max(ys.min() - pad, 0):ys.max() + pad, max(xs.min() - pad, 0):xs.max() + pad]


def runs(blank, min_len):
    out, start = [], None
    for i, b in enumerate(list(blank) + [False]):
        if b and start is None:
            start = i
        elif not b and start is not None:
            if i - start >= min_len:
                out.append((start, i))
            start = None
    return out


def split(im, axis, n, min_len):
    blank = ~(im[..., :3] < 0.96).any(axis=2).any(axis=1 - axis)
    gaps = [g for g in runs(blank, min_len) if g[0] > 0 and g[1] < len(blank)]
    cuts = sorted(sorted(gaps, key=lambda g: g[1] - g[0], reverse=True)[:n - 1])
    edges = [0] + [(a + b) // 2 for a, b in cuts] + [len(blank)]
    return [(edges[i], edges[i + 1]) for i in range(n)]


def panel_a_tiles():
    """Cut the 2 x 6 per-panel STRING render into 12 tiles and re-grid them 3 x 4."""
    im = mpimg.imread(B / "figS_ppi/fig11_disease_panels_ppi_mosaic_lean.png")[..., :3]
    tiles = []
    for r0, r1 in split(im, 0, 2, 15):
        band = im[r0:r1]
        for c0, c1 in split(band, 1, 6, 15):
            tiles.append(trim(band[:, c0:c1]))
    assert len(tiles) == 12
    h = max(t.shape[0] for t in tiles); w = max(t.shape[1] for t in tiles)
    g = 40
    canvas = np.ones((3 * h + 2 * g, 4 * w + 3 * g, 3))
    for i, t in enumerate(tiles):
        r, c = divmod(i, 4)
        y = r * (h + g) + (h - t.shape[0]) // 2; x = c * (w + g) + (w - t.shape[1]) // 2
        canvas[y:y + t.shape[0], x:x + t.shape[1]] = t
    return canvas


def sankey(path):
    """Cancers (top) to functional categories (bottom); ribbon width = panel proteins in that category."""
    pan = pd.read_csv(S.V7 / "panels/panel_memberships_v7.csv")
    func = A8.func_map(); fcol = A8.func_colours(); cats = list(fcol)
    flows = {}
    for c in S.ORDER:
        for p in pan[pan.panel == f"single_{S.NAME[c]}"].protein:
            flows[(c, func[p])] = flows.get((c, func[p]), 0) + 1
    tot_c = {c: sum(v for (a, _), v in flows.items() if a == c) for c in S.ORDER}
    tot_k = {k: sum(v for (_, b), v in flows.items() if b == k) for k in cats}
    cats = [k for k in cats if tot_k[k]]
    uniq = {k: len({p for c in S.ORDER for p in pan[pan.panel == f"single_{S.NAME[c]}"].protein if func[p] == k}) for k in cats}
    n = sum(tot_c.values()); gap = 1.6
    W = n + gap * (len(S.ORDER) - 1)
    xc, x = {}, 0.0
    for c in S.ORDER:
        xc[c] = x; x += tot_c[c] + gap
    sc = W / (n + gap * (len(cats) - 1))
    xk, x = {}, 0.0
    for k in cats:
        xk[k] = x; x += (tot_k[k] + gap) * sc
    fig, ax = plt.subplots(figsize=(24, 6.2))
    yt, yb, hb = 1.0, 0.0, 0.07
    oc = dict(xc); ok = dict(xk)
    for c in S.ORDER:
        for k in cats:
            v = flows.get((c, k), 0)
            if not v:
                continue
            a0, a1 = oc[c], oc[c] + v; b0, b1 = ok[k], ok[k] + v * sc
            ys = np.linspace(yb, yt, 60); t = (1 - np.cos(np.pi * (ys - yb) / (yt - yb))) / 2
            ax.fill_betweenx(ys, b0 + (a0 - b0) * t, b1 + (a1 - b1) * t, color=S.COL[c], alpha=0.55, lw=0)
            oc[c] += v; ok[k] += v * sc
    for c in S.ORDER:
        ax.add_patch(plt.Rectangle((xc[c], yt), tot_c[c], hb, color=S.COL[c], lw=0))
        ax.text(xc[c] + tot_c[c] / 2, yt + hb + 0.03, S.NAME[c], ha="center", va="bottom", fontsize=17, color=S.COL[c],
                fontweight="bold")
    for k in cats:
        ax.add_patch(plt.Rectangle((xk[k], yb - hb), tot_k[k] * sc, hb, color=fcol[k], lw=0))
        ax.text(xk[k] + tot_k[k] * sc / 2, yb - hb - 0.04, textwrap.fill(k.replace("/", "/ "), 11).replace("/ ", "/") + "\n" + f"n={uniq[k]}", ha="center", va="top", fontsize=16,
                color="#111111", fontweight="bold", linespacing=1.05)
    ax.set_xlim(-1, W + 1); ax.set_ylim(-0.42, 1.2); ax.axis("off")
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def chord(path):
    src = inspect.getsource(A8.chord)
    for o, n in {'    ax.legend(handles=h, loc="upper center", ncol=3, frameon=False, fontsize=16, bbox_to_anchor=(0.5, 0.0))\n': "",
                 "fig.savefig(P[\"E\"]": "fig.savefig(CHORD_OUT",
                 "short(r.term)": "abbreviate(r.term, 20)",
                 "fontsize=11.5,": "fontsize=17,",
                 "    tp = np.radians(np.linspace(165, 15, len(prots)))": "    tp = np.radians(np.linspace(100, 260, len(prots)))",
                 "    tt = np.radians(np.linspace(192, 348, len(terms)))": "    tt = np.radians(np.linspace(88, -88, len(terms)))",
                 'ax.set_xlim(-1.9, 1.9); ax.set_ylim(-2.25, 1.75)': 'ax.set_xlim(-2.0, 2.4); ax.set_ylim(-1.9, 1.9)',
                 'rotation_mode="anchor", fontsize=16, color=col)': 'rotation_mode="anchor", fontsize=21, color=col)'}.items():
        assert src.count(o) == 1, o
        src = src.replace(o, n)
    A8.__dict__.update(CHORD_OUT=path, abbreviate=abbreviate)
    exec(compile(src, "chord (fig8 v7f)", "exec"), A8.__dict__)
    A8.chord()


def pathway(outdir):
    s = (S.V7 / "figures/scripts/figS_v7c_pathway.py").read_text(encoding="utf-8")
    s = s.replace('BUILD = S.V7 / "figures/build/figS_pathway_c"', f'BUILD = Path(r"{outdir}")')
    s = s.replace("""                 '    if iso_path.exists():'""",
                  """                 '[wrap(clean_term(t), 19) for t in d.term]': '[V7_SHORT(clean_term(t)) for t in d.term]',
                 'cb = fig.colorbar(sc, ax=ax, orientation="horizontal", fraction=0.075,\\n                      pad=0.20, aspect=26)':
                     'cb = fig.colorbar(sc, ax=ax, orientation="vertical", fraction=0.06, pad=0.03, aspect=10)',
                 '    fig = plt.figure(figsize=(17.6, 5.4))': '    fig = plt.figure(figsize=(19.0, 4.4))',
                 '    if iso_path.exists():'""")
    s = s.replace('ns = {"__file__": str(SC / "fig6_pathway_enrichment_v2.py"), "__name__": "figS_ora"}',
                  'ns = {"__file__": str(SC / "fig6_pathway_enrichment_v2.py"), "__name__": "figS_ora", "V7_SHORT": V7_SHORT}')
    s = s.replace("import re\n", "import re\nfrom pathlib import Path\n", 1)
    ns = {"__name__": "fig8_pathway", "V7_SHORT": lambda t: abbreviate(t, 14)}
    exec(compile(s, "figS_v7c_pathway (fig8)", "exec"), ns)
    ns["main"]()


def inside_frame(a, pad=6):
    """Crop inside the purple frame of the STRING render."""
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    purple = (b > r) & (r > g + 0.12) & (g < 0.45)
    cols = np.where(purple.mean(axis=0) > 0.4)[0]; rows = np.where(purple.mean(axis=1) > 0.4)[0]
    if len(cols) < 2 or len(rows) < 2:
        return a
    return a[rows.min() + pad + 8:rows.max() - pad - 8, cols.min() + pad + 8:cols.max() - pad - 8]


def on_band(a, thr=0.965):
    """Replace the white background with the purple band colour."""
    a = a.copy()
    band = np.array(S.tint(S.PURPLE, 0.07))
    a[(a > thr).all(axis=2)] = band
    return a


def show(fig, rect, img, key, note=None):
    ax = fig.add_axes(rect)
    ax.imshow(img, interpolation="lanczos"); ax.set_axis_off()
    S.letter(fig, ax, key, dx=-0.012, dy=0.004)
    if note:
        ax.text(0.5, -0.01, note, transform=ax.transAxes, ha="center", va="top", fontsize=S.FS["annot"] + 2, color=S.INK)


def main():
    sankey(BUILD / "sankey_v.png")
    chord(BUILD / "chord.png")
    pathway(BUILD / "pathway")
    im = {"A": panel_a_tiles(), "B": trim(mpimg.imread(B / "figS_ppi/fig11c_pooled12panels_network_lean.png")[..., :3]),
          "C": trim(mpimg.imread(BUILD / "sankey_v.png")[..., :3]),
          "D": on_band(trim(inside_frame(trim(mpimg.imread(B / "figS_ppi/fig10_locked25_ppi_network_lean.png")[..., :3])))),
          "E": on_band(trim(mpimg.imread(BUILD / "chord.png")[..., :3])),
          "F": on_band(trim(mpimg.imread(BUILD / "pathway/fig6_pathway_enrichment.png")[..., :3]))}
    asp = {k: v.shape[1] / v.shape[0] for k, v in im.items()}
    st = pd.read_csv(S.RES / "B9_biology/string/ppi_reaudit_combined_results_v2.csv").set_index("panel")
    edges = {p: f"{int(st.at[p, 'official_number_of_edges'])} observed · {st.at[p, 'official_expected_number_of_edges']:.0f} expected edges"
             for p in ("LOCKED25", "POOLED12")}
    W, m, gap = 26.0, 0.03, 0.02
    usable = 1 - 2 * m
    h1 = (usable - gap) * W / (asp["A"] + asp["B"])
    hC = usable * W / asp["C"]
    KE = 1.25                                                       # chord 25% taller than the network
    hD = (usable - 2 * gap) * W / (asp["D"] + KE * asp["E"])
    h3 = KE * hD
    wD, wE = hD * asp["D"] / W, h3 * asp["E"] / W
    hF = usable * 0.92 * W / asp["F"]
    Hh = h1 + hC + h3 + hF + 7.4
    fig = plt.figure(figsize=(W, Hh))
    y = 1 - 1.1 / Hh
    fig.text(m, y + 0.4 / Hh, "Disease-specific panels", fontsize=S.FS["group"], color=S.INK, va="bottom")
    y -= h1 / Hh
    x = m
    for k, note in (("A", None), ("B", None)):
        w = h1 * asp[k] / W
        show(fig, [x, y, w, h1 / Hh], im[k], k, note)
        x += w + gap
    y -= 0.9 / Hh + hC / Hh
    show(fig, [m, y, usable, hC / Hh], im["C"], "C")
    y_band_top = y - 0.8 / Hh
    y = y_band_top - 1.3 / Hh - h3 / Hh
    x = m + (usable - wD - wE - gap) / 2
    for k, wk in (("D", wD), ("E", wE)):
        hk = wk * W / asp[k]
        show(fig, [x, y + (h3 - hk) / 2 / Hh, wk, hk / Hh], im[k], k)
        x += wk + gap
    y -= 0.9 / Hh + hF / Hh
    show(fig, [m + usable * 0.04, y, usable * 0.92, hF / Hh], im["F"], "F")
    y_band_bot = y - 0.35 / Hh
    fig.patches.append(FancyBboxPatch((0.012, y_band_bot), 0.976, y_band_top - y_band_bot,
                                      boxstyle="round,pad=0,rounding_size=0.004", transform=fig.transFigure,
                                      facecolor=S.tint(S.PURPLE, 0.07), edgecolor=S.PURPLE_LIGHT, lw=1.6, zorder=-5))
    fig.text(m, y_band_top - 0.7 / Hh, "26-protein list", fontsize=S.FS["group"], color=S.PURPLE, va="bottom")
    kw = dict(frameon=False, fontsize=S.FS["legend"] - 2, title_fontsize=S.FS["legend"] - 2, handlelength=1.6, columnspacing=1.2)
    l1 = [Line2D([], [], color=c, lw=4, label=l) for c, l in A8.STRING_EV]
    l2 = [Patch(fc=S.COL[c], label=S.NAME[c]) for c in S.ORDER] + \
         [Line2D([], [], color="#8b1a1a", lw=3, label="between panels"), Line2D([], [], color="#9a9a9a", lw=3, label="within a panel")]
    l3 = [Line2D([], [], ls="none", marker="s", ms=13, color=c, label=k) for k, c in DBCOL.items()]
    fig.legend(handles=l1, loc="lower left", bbox_to_anchor=(0.02, 0.003), ncol=4, title="STRING evidence (A, D)", **kw)
    fig.legend(handles=l3, loc="lower center", bbox_to_anchor=(0.47, 0.003), ncol=1, title="Database (E, F)", **kw)
    fig.legend(handles=l2, loc="lower right", bbox_to_anchor=(0.99, 0.003), ncol=7, title="Panel and edge (A–C)", **kw)
    S.save(fig, "fig8_v7j_biology_annotation.png")


if __name__ == "__main__":
    main()
