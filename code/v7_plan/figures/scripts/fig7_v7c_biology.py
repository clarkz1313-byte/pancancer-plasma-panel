"""v7 Figure 7, revision c (author review 17 Oct; forked from fig7_v7b_biology.py): B staircase of random-pool
nulls (B9e); C screen in the v5 layout (PP.H4 x, Wald y; neutral pass region; no P notes; no list legend);
D letter and GWAS labels (CRC/BRC in cancer colours). BRC/CVX colours swapped project-wide in v7_style.
Revision b notes: Changes: A one ring per panel
(wedge = panel protein, coloured = elevated in the cancer's own tissue or cell of origin, tick = share expected by
chance); B the 12 panels pooled as one bar of those proteins against random panels of the same sizes (mean, 95%
range of 10,000 draws); C screen in the first-draft style; E-F SuSiE credible sets as bars on a zoomed window.
Earlier description (D-G unchanged):
(A) Tissue/cell of origin: for each disease-specific panel, panel proteins elevated in its own cancer's origin (HPA
    25.1; filled, cancer colour) against the number expected from the 1,454 measured proteins (open grey); protein
    names on the right; * BH q < 0.05 (B9b).
(B) The 12 panels pooled: own-origin proteins in 10,000 random panels of the same sizes (grey) and in the panels (line).
(C) Genetic screen of 332 protein-cancer pairs: Wald-ratio P against colocalisation PP.H4; Bonferroni and 0.80 lines;
    purple = pairs of 26-protein-list proteins, grey = disease-specific proteins; stars pass both rules, diamonds the
    Wald rule only.
(D) Colorectal (up) and breast (down) cancer GWAS, GRCh38; windows locate E-G.
(E-G) Loci: pQTL and GWAS tracks (points coloured by distance from the probed variant, diamond = Wald instrument or
    probed variant) and, for E-F, SuSiE posterior inclusion probabilities with the 1000 Genomes GRCh38 EUR LD
    reference (B10g): filled circles = pQTL credible-set variants, open squares = GWAS credible-set variants.
    G is the BMP4 cis region on chr14 (control, no fine-mapping).
Inputs: B9b (tissue), results/B10_genetics_final (screen, B10g PIPs), figures/build/fig7 (fig7_v7_genetics_data.py),
revise_plan/smr_coloc source tables via figurev6/scripts/fig13_regional_evidence_2026-09-05.py (loaded, not modified).
Output: figures/output/fig7_v7c_biology.png
"""
import importlib
import json
import importlib.util
import zipfile

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch

import v7_style as S

plt = S.plt
R = importlib.import_module("fig13_regional_evidence_2026-09-05")
plt.rcParams.update({"font.family": "DejaVu Sans"})   # the v6 module switches to Arial on import
BUILD = S.V7 / "figures/build/fig7"
GEN = S.RES / "B10_genetics_final"
C_CRC, C_BRC, C_CTRL = S.COL["CRC"], S.COL["BRC"], "#6b6b6b"

# ---------------------------------------------------------------- tissue of origin (as fig10_v7_biology / B9b)
spec = importlib.util.spec_from_file_location("b9b", S.V7 / "scripts/B9b_tissue_of_origin_v7.py")
b9b = importlib.util.module_from_spec(spec); spec.loader.exec_module(b9b)
TO = S.RES / "B9_biology/tissue_origin"
hpa = pd.read_csv(zipfile.ZipFile(TO / "hpa/proteinatlas.tsv.zip").open("proteinatlas.tsv"), sep="\t",
                  usecols=["Gene", "Gene synonym", b9b.T, b9b.BL, b9b.SC]).drop_duplicates("Gene")
bg = [c for c in pd.read_csv(S.ROOT / "data/processed/filtered_pancancer_data.csv", nrows=1).columns
      if c not in ("Sample_ID", "Cancer", "protein_count", "Label")]
syn = {}
for _, r in hpa.iterrows():
    for s_ in str(r["Gene synonym"]).split(","):
        syn.setdefault(s_.strip(), r.Gene)
genes = set(hpa.Gene)
to_hpa = {p: (p if p in genes else syn.get(p)) for p in bg}
bg = [p for p in bg if to_hpa[p]]
H = hpa.set_index("Gene")
elev = {c: {p for p in bg if b9b.labels(H.loc[to_hpa[p], col]) & set(t)} for c, (col, t) in b9b.MAP.items()}
pan = pd.read_csv(S.V7 / "panels/panel_memberships_v7.csv")
CODES = [S.NAME[c] for c in S.ORDER]
ds = {c: [p for p in pan[pan.panel == f"single_{c}"].protein if to_hpa.get(p)] for c in CODES}
own = pd.read_csv(TO / "B9b_per_cancer.csv").set_index("cancer")
assert int(own.elevated_in_panel.sum()) == 21


def ccol(c):
    return S.COL["LYMPH" if c == "DLBCL" else c]


def bump(ax, k=1.2):
    for t in ax.texts:
        t.set_fontsize(t.get_fontsize() * k)
    for t in ax.get_xticklabels() + ax.get_yticklabels() + [ax.xaxis.label, ax.yaxis.label]:
        t.set_fontsize(t.get_fontsize() * k)


def donut(ax, c):
    """One ring per panel: one wedge per panel protein; wedges in the cancer colour are elevated in the cancer's own
    tissue or cell of origin, grey wedges are not. Black tick = the share expected by chance."""
    n, o = int(own.loc[c, "panel_size"]), int(own.loc[c, "elevated_in_panel"])
    e, q = own.loc[c, "expected"], own.loc[c, "q_bh"]
    cols = [ccol(c)] * o + ["#dedede"] * (n - o)
    ax.pie([1] * n, colors=cols, startangle=90, counterclock=False, radius=1.0,
           wedgeprops=dict(width=0.34, edgecolor="white", linewidth=2.2))
    th = np.radians(90 - 360 * e / n)
    ax.plot([0.60 * np.cos(th), 1.08 * np.cos(th)], [0.60 * np.sin(th), 1.08 * np.sin(th)], color=S.INK, lw=2.6,
            solid_capstyle="butt", zorder=5)
    ax.text(0, 0.02, f"{o}/{n}", ha="center", va="center", fontsize=S.FS["annot"] + 4, color=S.INK)
    ax.set_title(c + (" *" if q < 0.05 else ""), color=ccol(c), fontsize=S.FS["tick"] + 1, pad=2)
    prots = own.loc[c, "proteins"] if isinstance(own.loc[c, "proteins"], str) else ""
    ax.text(0, -1.22, "\n".join(prots.split(";")), ha="center", va="top", fontsize=S.FS["annot"] - 3, color=ccol(c),
            linespacing=1.05)
    ax.set_xlim(-1.15, 1.15); ax.set_ylim(-1.9, 1.15); ax.set_aspect("equal"); ax.axis("off")


def panel_a(fig, cell):
    sub = cell.subgridspec(2, 6, hspace=0.30, wspace=0.08)
    axes = [fig.add_subplot(sub[i // 6, i % 6]) for i in range(12)]
    for ax, c in zip(axes, CODES):
        donut(ax, c)
    return axes[0]


def null_draws():
    rng = np.random.default_rng(20261015)   # same draws as B9b
    bga = np.array(bg)
    return np.array([sum(len(set(rng.choice(bga, len(P), replace=False)) & elev[c]) for c, P in ds.items())
                     for _ in range(10000)])


def panel_b(ax):
    """Own-origin proteins in the 12 panels (bar built from each panel's proteins) against random panels of the same
    sizes drawn from three pools that follow the selection steps: all measured proteins (B9b), proteins that passed
    that cancer's screen, and that cancer's LASSO-ranked proteins (B9e). Grey bar = mean, whisker = 95% range."""
    js = json.load(open(TO / "B9e_summary.json"))
    S.frame(ax)
    rows = [("all", "All"), ("screen", "Screened"), ("lasso", "Ranked")]
    for y, (k, lab) in zip((3, 2, 1), rows):
        m, (lo, hi) = js[k]["mean"], js[k]["q2.5_q97.5"]
        ax.barh(y, m, height=0.56, color="#c4c4c4", edgecolor="white", zorder=3)
        ax.plot([lo, hi], [y, y], color=S.INK, lw=2.2, zorder=4)
        for x_ in (lo, hi):
            ax.plot([x_, x_], [y - 0.11, y + 0.11], color=S.INK, lw=2.2, zorder=4)
        ax.text(hi + 0.5, y, f"{m:.1f}", ha="left", va="center", fontsize=S.FS["annot"] + 4, color=S.GREY)
    left, y = 0, 0
    for c in CODES:
        o = int(own.loc[c, "elevated_in_panel"])
        if o:
            ax.barh(y, o, left=left, height=0.56, color=ccol(c), edgecolor="white", lw=1.4, zorder=3)
            left += o
    ax.text(left + 0.5, y, f"{left}", ha="left", va="center", fontsize=S.FS["annot"] + 6, color=S.INK)
    ax.set_yticks([3, 2, 1, 0]); ax.set_yticklabels([r[1] for r in rows] + ["Panels"], fontsize=S.FS["tick"] + 1)
    ax.set_ylim(-0.6, 3.6); ax.set_xlim(0, 25)
    ax.set_xlabel("Own-origin proteins", fontsize=S.FS["label"])
    ax.grid(axis="x", color="#ececec", lw=1, zorder=0)


def panel_c(ax):
    """Genetic screen in the layout of the v5 regional figure (fig11_v50, panel A): colocalisation PP.H4 (x) against
    Wald-ratio P (y); stars pass both rules, filled circles the Wald rule only; neutral shading = both rules."""
    g = pd.read_csv(GEN / "tier1_combined_v7_final.csv")
    S.frame(ax)
    x = g["PP.H4"]; y = -np.log10(g.p_wald_pkg.clip(lower=1e-40))
    bonf = -np.log10(g.bonferroni.iloc[0])
    ax.add_patch(plt.Rectangle((0.8, bonf), 0.25, 40, color="#ececec", zorder=0))
    ax.axhline(bonf, color=S.GREY, ls="--", lw=1.4, zorder=1); ax.axvline(0.8, color=S.GREY, ls="--", lw=1.4, zorder=1)
    sig = g.p_wald_pkg < g.bonferroni.iloc[0]
    ax.scatter(x[~sig], y[~sig], s=40, color="#c8c8c8", lw=0, zorder=2)
    at = {"BMP4": (0.60, 31.0), "LEP": (0.55, 24.5), "MSMB": (0.08, 19.4), "CD74": (0.08, 11.3), "IL17A": (0.08, 7.4)}
    for _, h in g[sig].iterrows():
        xx, yy = h["PP.H4"], -np.log10(max(h.p_wald_pkg, 1e-40))
        cn = "DLBCL" if h.cancer == "LYMPH" else h.cancer
        both = xx > 0.8
        if both:
            ax.scatter(xx, yy, s=520, marker="*", color=ccol(cn), edgecolor=S.INK, lw=1.0, zorder=5)
        else:
            ax.scatter(xx, yy, s=110, color=S.INK, lw=0, zorder=5)
        tx, ty = at[h.protein]
        ax.annotate(f"{h.protein} × {cn}", (xx, yy), xytext=(tx, ty), ha="left", va="center",
                    fontsize=S.FS["annot"] + (1 if both else 0), color=ccol(cn) if both else S.INK,
                    arrowprops=dict(arrowstyle="-", color=S.GREY, lw=1) if both else None)
    ax.set_xlim(-0.04, 1.05); ax.set_ylim(-0.8, 34)
    ax.set_xlabel("Colocalisation PP.H4", fontsize=S.FS["label"]); ax.set_ylabel("Wald ratio −log$_{10}$ P", fontsize=S.FS["label"])


def panel_d(ax):
    crc = pd.read_csv(BUILD / "manhattan_CRC.csv"); brc = pd.read_csv(BUILD / "manhattan_BRC.csv")
    ends = pd.concat([crc, brc]).groupby("chr").bp.max()
    offs = (ends * 1.02).cumsum().shift(fill_value=0)
    cap = 30
    for d, sign in ((crc, 1), (brc, -1)):
        lp = -np.log10(d.p.clip(lower=1e-300)); xc = d.bp + d.chr.map(offs)
        for par, col in ((0, "#8796a3"), (1, "#c3ccd4")):
            m = (d.chr % 2 == par)
            ax.scatter(xc[m], sign * lp[m].clip(upper=cap), s=4, c=col, linewidths=0, rasterized=True, zorder=2)
        over = lp > cap
        ax.scatter(xc[over], np.full(over.sum(), sign * cap), s=26, marker="^" if sign > 0 else "v", c="#8796a3",
                   linewidths=0, zorder=2)
        ax.axhline(sign * -np.log10(5e-8), color="#c0392b", lw=0.9, ls="--", zorder=3)
    ax.axhline(0, color=S.SPINE, lw=1.2)
    for chrom, pos, col, tag, sign in ((11, R.CHR11_INSTRUMENT, C_CRC, "E", 1), (14, 53_953_000, C_CTRL, "G", 1),
                                       (16, R.LEP_INSTRUMENT, C_BRC, "F", -1)):
        xc = pos + offs[chrom]
        ax.axvspan(xc - 8e6, xc + 8e6, color=col, alpha=0.30, zorder=1)
        ax.text(xc, sign * (cap + 2.2), tag, ha="center", va="bottom" if sign > 0 else "top",
                fontsize=S.FS["group"], color=col)
    ax.set_xlim(0, offs.iloc[-1] + ends.iloc[-1]); ax.set_ylim(-cap - 6, cap + 6)
    ax.set_xticks([offs[c] + ends[c] / 2 for c in ends.index])
    ax.set_xticklabels([str(c) if c <= 18 or c % 2 == 0 else "" for c in ends.index], fontsize=S.FS["tick"] - 3)
    ax.set_yticks([-30, -15, 0, 15, 30]); ax.set_yticklabels(["30", "15", "0", "15", "30"], fontsize=S.FS["tick"] - 2)
    ax.set_ylabel("−log$_{10}$ P", fontsize=S.FS["label"])
    ax.text(0.005, 0.98, "CRC GWAS", transform=ax.transAxes, ha="left", va="top", fontsize=S.FS["annot"] + 2, color=C_CRC)
    ax.text(0.005, 0.02, "BRC GWAS", transform=ax.transAxes, ha="left", va="bottom", fontsize=S.FS["annot"] + 2, color=C_BRC)
    ax.set_xlabel("Chromosome", fontsize=S.FS["label"])
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


ZOOM = {"BMP4": (74.32, 74.74), "LEP": (53.755, 53.845)}


def susie_track(ax, above, pip, prot, chrom, probe, col, chk):
    """Credible sets as bars on a zoomed window (marked on the GWAS track above): one row per set, pQTL sets above the
    line, GWAS sets below; dot = the set's top variant, size = its posterior inclusion probability."""
    z0, z1 = ZOOM[prot]
    above.axvspan(z0, z1, color="#000000", alpha=0.07, zorder=1)
    for xa, xb in ((z0, 0), (z1, 1)):
        ax.figure.add_artist(ConnectionPatch((xa, 0), (xb, 1), coordsA="data", coordsB="axes fraction", axesA=above,
                                             axesB=ax, color="#9a9a9a", lw=1.0, ls="--"))
    ax.set_facecolor("#f7f7f7")
    rows = []
    for k, sign, c_ in (("pqtl", 1, R.C_PQTL), ("gwas", -1, R.C_GWAS)):
        cs = pip[pip[f"cs_{k}"].notna()]
        for i, (j, g) in enumerate(sorted(cs.groupby(f"cs_{k}"), key=lambda t: t[1].pos.min())):
            yy = sign * (i + 1)
            top = g.loc[g[f"pip_{k}"].idxmax()]
            ax.plot([g.pos.min() / 1e6, g.pos.max() / 1e6], [yy, yy], color=c_, lw=7, solid_capstyle="round",
                    alpha=0.55, zorder=3)
            ax.scatter(top.pos / 1e6, yy, s=60 + 260 * top[f"pip_{k}"], color=c_, edgecolor="white", lw=1.2, zorder=4)
            rows.append((k, yy, g.pos.min(), g.pos.max()))
    nq = sum(r[0] == "pqtl" for r in rows); ng = sum(r[0] == "gwas" for r in rows)
    ax.axhline(0, color=S.SPINE, lw=1)
    ax.axvline(probe / 1e6, color=col, lw=1.2, ls=":", zorder=2)
    ax.text(0.012, 0.97, "pQTL credible sets", transform=ax.transAxes, ha="left", va="top", fontsize=12, color=R.C_PQTL)
    c = chk.loc[prot]
    none = "" if np.isfinite(c["susie_max_PP.H4"]) else ": none"
    ax.text(0.012, 0.03, "GWAS credible sets" + none, transform=ax.transAxes, ha="left", va="bottom", fontsize=12,
            color=R.C_GWAS)
    if np.isfinite(c["susie_max_PP.H4"]):
        # shade where the pQTL set holding the instrument overlaps GWAS sets (the colocalising pair)
        q = [r for r in rows if r[0] == "pqtl" and r[2] <= probe <= r[3]][0]
        for gs_ in [r for r in rows if r[0] == "gwas" and min(q[3], r[3]) > max(q[2], r[2])]:
            ax.axvspan(max(q[2], gs_[2]) / 1e6, min(q[3], gs_[3]) / 1e6, color=col, alpha=0.18, zorder=1)
        ax.text(0.98, 0.75, f"shared  PP.H4 {c['susie_max_PP.H4']:.2f}", transform=ax.transAxes, ha="right",
                va="center", fontsize=13, color=col)
    ax.set_ylim(-max(ng, 1) - 0.8, nq + 1.7); ax.set_yticks([])
    ax.set_xlim(z0, z1)
    ax.set_ylabel("SuSiE", fontsize=14, color=S.INK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", labelsize=12, colors=col)
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax.set_xlabel(f"chr{chrom} position (Mb), zoomed", fontsize=14, color=col)


def diamond(ax, pos, s, col, ytop):
    if len(s):
        row = s.loc[(s["pos"] - pos).abs().idxmin()]
        ax.scatter([pos / 1e6], [min(row.neglog10p, ytop * 0.94)], s=170, marker="D", facecolor=col,
                   edgecolor=S.INK, linewidth=1.6, zorder=12)


def loci(fig, cells):
    man, trans, merged, cis, cis_sum, land, brow = R.load()
    lq, lg = R.load_lep()
    pips = pd.read_csv(GEN / "B10g_susie_pips_1kg38.csv")
    chk = pd.read_csv(GEN / "B10g_check.csv").set_index("protein")
    tq, tg = trans[trans.source == "pQTL"], trans[trans.source == "GWAS"]
    cq, cg = cis[cis.source == "pQTL"], cis[cis.source == "GWAS"]
    cis_probe = int(cq.loc[cq.neglog10p.idxmax(), "pos"])
    spec_ = [("E", tq, tg, R.CHR11_INSTRUMENT, C_CRC, 11, R.LABELS_BMP4, {"CHRDL2": R.CHRDL2_SPAN_MB, "POLD3": R.POLD3_SPAN_MB},
              ("HEIDI p  .004 / .075", "PP.H4  .99→.66"), "BMP4"),
             ("F", lq, lg, R.LEP_INSTRUMENT, C_BRC, 16, R.LABELS_LEP, {"FTO": R.FTO_SPAN_MB},
              ("HEIDI p  .011 / .055", "PP.H4  .99→.43"), "LEP"),
             ("G", cq, cg, cis_probe, C_CTRL, 14, R.LABELS_BMP4, {"BMP4": R.BMP4_SPAN_MB}, ("No overlap", ""), None)]
    first = slot = None
    for cell, (tag, q, g, probe, col, chrom, labs, spans, verdict, prot) in zip(cells, spec_):
        sub = cell.subgridspec(3, 1, hspace=0.34, height_ratios=[1, 1, 0.80])
        a1 = fig.add_subplot(sub[0]); a2 = fig.add_subplot(sub[1], sharex=a1)
        R._track(a1, q, "pQTL", probe, col, False, chrom, 50.0, labels=labs)
        R._track(a2, g, "GWAS", probe, col, prot is None, chrom, 30.0, labels=labs)
        diamond(a1, probe, q, col, 50.0); diamond(a2, probe, g, col, 30.0)
        for a in (a1, a2):
            R._genes(a, spans)
        R._verdict(a1, *verdict, col)
        axes = [a1, a2]
        if prot is None:
            a2.xaxis.set_major_locator(plt.MaxNLocator(5))
            slot = sub[2]
        if prot:
            a3 = fig.add_subplot(sub[2])
            a2.set_xlabel(f"chr{chrom} position (Mb)", fontsize=14, color=col)
            a2.tick_params(axis="x", labelbottom=True); plt.setp(a2.get_xticklabels(), visible=True)
            susie_track(a3, a2, pips[pips.protein == prot], prot, chrom, probe, col, chk)
            axes.append(a3)
        for a in axes:
            bump(a, 1.18)
        S.letter(fig, a1, tag, dx=-0.03, dy=0.006)
        first = first or a1
    return slot


def legend(fig, slot):
    ax = fig.add_subplot(slot); ax.set_axis_off()
    h = [Line2D([], [], ls="none", marker="o", ms=11, color=R.DIST_COL[b], label=R.BIN_LABEL[b]) for b in R.BINS]
    h += [Line2D([], [], ls="none", marker="D", ms=11, mfc="white", mec=S.INK, mew=1.6, label="instrument / probe"),
          Line2D([], [], color=R.C_PQTL, lw=6, alpha=0.6, label="pQTL credible set"),
          Line2D([], [], color=R.C_GWAS, lw=6, alpha=0.6, label="GWAS credible set"),
          Line2D([], [], color="#c0392b", ls="--", lw=1.2, label="P = 5×10$^{-8}$")]
    kw = dict(frameon=False, fontsize=S.FS["legend"] - 4, title_fontsize=S.FS["legend"] - 4, handletextpad=0.4,
              columnspacing=1.0, alignment="left")
    ax.legend(handles=h, loc="upper left", bbox_to_anchor=(0.0, 0.62), ncol=2, title="E–G", **kw)


def main():
    fig = plt.figure(figsize=(26, 31))
    gs = fig.add_gridspec(3, 6, height_ratios=[0.86, 0.78, 1.55], hspace=0.30, wspace=0.55,
                          left=0.07, right=0.985, top=0.975, bottom=0.035)
    axA = panel_a(fig, gs[0, 0:4])
    axB = fig.add_subplot(gs[0, 4:6]); panel_b(axB)
    axC = fig.add_subplot(gs[1, 0:2]); panel_c(axC)
    axD = fig.add_subplot(gs[1, 2:6]); panel_d(axD)
    for ax, s_ in ((axA, "A"), (axB, "B"), (axC, "C"), (axD, "D")):
        S.letter(fig, ax, s_, dx={"A": -0.02, "B": -0.075, "C": -0.045, "D": -0.035}[s_], dy=0.008)
    legend(fig, loci(fig, [gs[2, 0:2], gs[2, 2:4], gs[2, 4:6]]))
    S.save(fig, "fig7_v7c_biology.png")


if __name__ == "__main__":
    main()
