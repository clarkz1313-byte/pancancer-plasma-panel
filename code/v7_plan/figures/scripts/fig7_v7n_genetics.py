"""v7 Figure 7, revision n (18 Oct): headless pointers from the B labels, no panel-letter prefix, "Chromosome N".
Earlier: revision l (17 Oct): "top cis variant" label middle-left in E. Revision k (17 Oct): tracks not x-shared (pQTL keeps all position labels); "pQTL for BMP4-trans".
Revision j (17 Oct): rsID pointer on pQTL, GWAS and credible-set tracks; gene names on both
tracks; centred one-line position labels except under the zoom lines. Revision h (17 Oct): position axis on both locus tracks, chromosome numbers on both Manhattans
with the zoomed chromosomes highlighted, light column guides between C-E. Revision g (author review 17 Oct): black "for" and "×" in every label; B column labels
"C · BMP4-trans × CRC" with pointers; gene boundaries as black dotted lines in both tracks; bolder zoom lines;
credible-set strips with the same pQTL/GWAS split in C and D, packed rows, instrument diamond, kb axis. Revision f: genetic evidence, colours back on the v5 template
(fig11_v50_regional_evidence): protein (pQTL) tracks in the pQTL teal, GWAS tracks in an olive green (instead of the v5
rose), each with a faint tint; the protein or cancer named in a track or label is written in its cancer colour, the
data type in its theme colour ("pQTL for BMP4", "GWAS for CRC"), the same in A-E.
(A) Screen of 332 pairs: colocalisation PP.H4 against Wald-ratio P; the five Wald-significant pairs labelled in their
    cancer colour with pointers; pairs passing both rules filled in the cancer colour, Wald-only pairs grey with a dark
    border; grey region = both rules.
(B) CRC and BRC GWAS in grey scale; zoomed columns highlighted and named, instruments called trans (variant far from
    the protein's own gene) or cis (variant at the protein's own gene).
(C-E) Loci (v5 grammar: points coloured by distance from the diamond; pointer = instrument variant; gene names
    black). Under C and D, SuSiE credible sets on a zoomed window (dashed lines from the GWAS track): one bar per set,
    pQTL sets above the line and GWAS sets below; the shaded column marks where the pQTL and GWAS sets overlap.
Output: figures/output/fig7_v7g_genetics.png
"""
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch
from matplotlib.ticker import FuncFormatter

import v7_style as S
import fig7_v7c_biology as B

plt = S.plt
R = B.R
plt.rcParams.update({"font.family": "DejaVu Sans"})
C_CRC, C_BRC = S.COL["CRC"], S.COL["BRC"]
PQ, GW_C = "#1f6f8b", "#5b7a1f"                 # pQTL teal (v5) and GWAS olive green
BG = {"pQTL": "#f2f8fb", "GWAS": "#f5f8ee"}
GW = -np.log10(5e-8)
INK_K = "#111111"
LOCI = {"C": (11, R.CHR11_INSTRUMENT, "BMP4", "trans", C_CRC, "CRC"),
        "D": (16, R.LEP_INSTRUMENT, "LEP", "trans", C_BRC, "BRC"),
        "E": (14, 53_953_000, "BMP4", "cis", C_CRC, "CRC")}
ZOOM = {"C": (74.32, 74.74), "D": (53.755, 53.845)}
RSID = {"BMP4": R.CHR11_INSTRUMENT_RSID, "LEP": R.LEP_INSTRUMENT_RSID}


def two_tone(ax, x, y, parts, size, transform=None, **kw):
    """Left-aligned text made of (string, colour) pieces placed one after another."""
    t = ax.text(x, y, parts[0][0], color=parts[0][1], fontsize=size, ha="left", transform=transform or ax.transAxes, **kw)
    for s, c in parts[1:]:
        t = ax.annotate(s, xycoords=t, xy=(1, 0), va="bottom", ha="left", color=c, fontsize=size, **kw)
    return t


def panel_a(ax):
    g = pd.read_csv(B.GEN / "tier1_combined_v7_final.csv")
    S.frame(ax)
    x = g["PP.H4"]; y = -np.log10(g.p_wald_pkg.clip(lower=1e-40))
    bonf = -np.log10(g.bonferroni.iloc[0])
    ax.add_patch(plt.Rectangle((0.8, bonf), 0.25, 40, color="#ececec", zorder=0))
    ax.axhline(bonf, color=S.GREY, ls="--", lw=1.4, zorder=1); ax.axvline(0.8, color=S.GREY, ls="--", lw=1.4, zorder=1)
    sig = g.p_wald_pkg < g.bonferroni.iloc[0]
    ax.scatter(x[~sig], y[~sig], s=40, color="#c8c8c8", lw=0, zorder=2)
    at = {"BMP4": (0.30, 30.0), "LEP": (0.30, 22.5), "MSMB": (0.16, 16.5), "CD74": (0.16, 12.6), "IL17A": (0.16, 4.8)}
    for _, h in g[sig].iterrows():
        xx, yy = h["PP.H4"], -np.log10(max(h.p_wald_pkg, 1e-40))
        cn = "DLBCL" if h.cancer == "LYMPH" else h.cancer
        col = B.ccol(cn)
        both = xx > 0.8
        ax.scatter(xx, yy, s=300 if both else 80, color=col if both else "#bdbdbd", edgecolor=S.INK, lw=1.6 if both else 1.1, zorder=5)
        tx, ty = at[h.protein]
        last = two_tone(ax, tx, ty, [(h.protein, col), (" × ", INK_K), (cn, col)], S.FS["annot"] + (1 if both else -3), transform=ax.transData)
        if xx > tx:                       # point to the right of the label: line from the label's end (resolved at draw)
            ax.annotate("", (xx, yy), xytext=(1.04, 0.45), textcoords=last,
                        arrowprops=dict(arrowstyle="-", color=S.GREY, lw=1.1))
        else:
            ax.annotate("", (xx, yy), xytext=(tx - 0.01, ty + 0.6), arrowprops=dict(arrowstyle="-", color=S.GREY, lw=0.8))
    ax.set_xlim(-0.04, 1.05); ax.set_ylim(-0.8, 34)
    ax.set_xlabel("Colocalisation PP.H4", fontsize=S.FS["label"]); ax.set_ylabel("Wald ratio −log$_{10}$ P", fontsize=S.FS["label"])


def manhattans(fig, cell):  # fig used for text extents
    sub = cell.subgridspec(2, 1, hspace=0.24)
    data = {k: pd.read_csv(B.BUILD / f"manhattan_{k}.csv") for k in ("CRC", "BRC")}
    ends = pd.concat(data.values()).groupby("chr").bp.max()
    offs = (ends * 1.02).cumsum().shift(fill_value=0)
    cap = 30
    axes = []
    for i, (k, col) in enumerate((("CRC", C_CRC), ("BRC", C_BRC))):
        ax = fig.add_subplot(sub[i])
        d = data[k]
        lp0 = -np.log10(d.p.clip(lower=1e-300))
        keep = (lp0 < 3) | (lp0 >= 5) | (np.random.default_rng(1).random(len(d)) < 0.03 * 10 ** ((lp0 - 3) * 0.762))
        d, lp = d[keep], lp0[keep]
        xc = d.bp + d.chr.map(offs)
        for par, c_ in ((0, "#8796a3"), (1, "#c3ccd4")):
            m = d.chr % 2 == par
            ax.scatter(xc[m], lp[m].clip(upper=cap), s=4, c=c_, linewidths=0, rasterized=True, zorder=2)
        over = lp > cap
        ax.scatter(xc[over], np.full(over.sum(), cap), s=24, marker="^", c="#8796a3", linewidths=0, zorder=2)
        ax.axhline(GW, color="#c0392b", lw=0.9, ls="--", zorder=3)
        for tag, (chrom, pos, prot, kind, lcol, gw) in LOCI.items():
            if gw != k:
                continue
            x_ = pos + offs[chrom]
            ax.axvspan(x_ - 8e6, x_ + 8e6, color=lcol, alpha=0.40, zorder=1)
            lx = {"C": offs[5], "D": offs[10], "E": offs[15] + 3e7}[tag]
            parts = [(f"{prot}-{kind}", lcol), (" × ", INK_K), (gw, lcol)]
            n0 = len(ax.texts)
            last = two_tone(ax, lx, cap - 4.5, parts, S.FS["annot"] + 1, transform=ax.transData)
            first = ax.texts[n0]
            anchor, at = (first, (-0.04, 0.5)) if tag == "E" else (last, (1.06, 0.5))   # resolved at draw time
            ax.annotate("", xy=(x_, cap * 0.55), xytext=at, textcoords=anchor,
                        arrowprops=dict(arrowstyle="-", color=INK_K, lw=1.4), zorder=6)
        two_tone(ax, 0.006, 0.86, [("GWAS", GW_C), (" for ", INK_K), (k, col)], S.FS["annot"] + 2)
        ax.set_xlim(0, offs.iloc[-1] + ends.iloc[-1]); ax.set_ylim(0, cap + 3)
        ax.set_yticks([0, 10, 20, 30]); ax.tick_params(labelsize=S.FS["tick"] - 2)
        ax.set_ylabel("−log$_{10}$ P", fontsize=S.FS["label"] - 1)
        ax.set_xticks([offs[c] + ends[c] / 2 for c in ends.index])
        ax.set_xticklabels([str(c) if c <= 18 or c % 2 == 0 else "" for c in ends.index], fontsize=S.FS["tick"] - 3)
        hi = {"CRC": (11, 14), "BRC": (16,)}[k]          # chromosomes zoomed in C-E
        for lab, c in zip(ax.get_xticklabels(), ends.index):
            if c in hi:
                lab.set_color(col); lab.set_fontsize(S.FS["tick"] + 2)
        if i == 1:
            ax.set_xlabel("Chromosome", fontsize=S.FS["label"])
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        axes.append(ax)
    return axes[0]


def track(ax, s, probe, kind, name, ncol, ytop, genes, first, chrom, xcol, pointer):
    tcol = PQ if kind == "pQTL" else GW_C
    bottom = kind == "GWAS"
    ax.set_facecolor(BG[kind])
    cap = ytop * 0.94
    s = s.copy()
    s["bin"] = pd.cut((s["pos"] - probe).abs() / 1000, bins=[-np.inf, 50, 100, 200, 350, np.inf], labels=R.BINS)
    for k, (g, (g0, g1)) in enumerate(genes.items()):
        ax.axvspan(g0, g1, color="#000000", alpha=0.05, zorder=0)
        for gx in (g0, g1):
            ax.axvline(gx, color=INK_K, ls=":", lw=1.4, zorder=2)
        if True:
            ax.text((g0 + g1) / 2, ytop * ((0.62 if not bottom else 0.68) - 0.09 * k), g, ha="center", va="top", fontsize=12, style="italic",
                    color="black", zorder=15)
    for key in R.BINS[::-1]:
        m = s[s["bin"] == key]
        ax.scatter(m["pos"] / 1e6, m["neglog10p"].clip(upper=cap), s=20, c=R.DIST_COL[key], linewidths=0, alpha=0.9,
                   zorder=3 + R.BINS.index(key))
    ax.axvline(probe / 1e6, color=ncol, lw=1.2, ls="-", alpha=0.6, zorder=2)
    ax.axhline(GW, color="#c0392b", lw=0.9, ls="--", zorder=2)
    row = s.loc[(s["pos"] - probe).abs().idxmin()]
    py = min(row.neglog10p, cap)
    ax.scatter(probe / 1e6, py, s=170, marker="D", color=ncol, edgecolor=S.INK, lw=1.6, zorder=12)
    if pointer:
        ax.annotate(pointer, (probe / 1e6, py), xytext=((0.04, 0.55) if pointer == "top cis variant" else (0.60, 0.70)),
                    textcoords="axes fraction", ha="left",
                    fontsize=13, color=S.INK, arrowprops=dict(arrowstyle="-", color=S.INK, lw=1.0), zorder=14)
    two_tone(ax, 0.02, 0.86, [(kind, tcol), (" for ", INK_K), (name, ncol)], 15)
    ax.set_ylim(0, ytop)
    ax.tick_params(labelsize=13)
    ax.tick_params(axis="y", colors=tcol)
    ax.spines["left"].set_color(tcol); ax.spines["left"].set_linewidth(2.0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    if first:
        ax.set_ylabel(f"{kind}\n−log$_{{10}}$ P", fontsize=14, color=tcol)
    else:
        ax.tick_params(axis="y", labelleft=False)
    ax.tick_params(axis="x", colors=xcol, labelbottom=True)       # both tracks show the shared position axis
    ax.set_xlabel(f"Chromosome {chrom} position (Mb)", fontsize=13, color=xcol)


def pack(sets, pad):
    """Greedy rows: sets that do not overlap (with pad) share a row. sets = [(start, end)] sorted by start."""
    ends, rows = [], []
    for a, b in sets:
        for r, e in enumerate(ends):
            if a - pad > e:
                ends[r] = b; rows.append(r); break
        else:
            ends.append(b); rows.append(len(ends) - 1)
    return rows


def credible_sets(prot, z):
    pip = pd.read_csv(B.GEN / "B10g_susie_pips_1kg38.csv")
    pip = pip[pip.protein == prot]
    out = {}
    for k in ("pqtl", "gwas"):
        cs = pip[pip[f"cs_{k}"].notna()]
        sets = sorted((g.pos.min() / 1e6, g.pos.max() / 1e6) for _, g in cs.groupby(f"cs_{k}"))
        out[k] = list(zip(sets, pack(sets, 0.02 * (z[1] - z[0]))))
    return out


NROWS = {}   # filled in loci(): the same pQTL/GWAS split for C and D


def susie(ax, above, prot, probe, col, z, first, chrom):
    """Credible sets on a zoomed window: one bar per set (pQTL above the line, GWAS below); diamond = instrument;
    shaded column = where the pQTL set holding the instrument overlaps a GWAS set."""
    z0, z1 = z
    above.axvspan(z0, z1, color="#000000", alpha=0.10, zorder=1)
    for xa, xb in ((z0, 0), (z1, 1)):
        ax.figure.add_artist(ConnectionPatch((xa, 0), (xb, 1), coordsA="data", coordsB="axes fraction", axesA=above,
                                             axesB=ax, color="#666666", lw=1.3, ls="--"))
    cs = credible_sets(prot, z)
    held = None
    for k, sign, c_ in (("pqtl", 1, PQ), ("gwas", -1, GW_C)):
        for (a_, b_), r in cs[k]:
            yy = sign * (r + 1)
            half = max(0, 0.008 * (z1 - z0) - (b_ - a_)) / 2
            ax.plot([a_ - half, b_ + half], [yy, yy], color=c_, lw=9, solid_capstyle="round", zorder=3)
            if k == "pqtl" and a_ <= probe / 1e6 <= b_:
                held = (a_, b_)
    if held:
        for (a_, b_), r in cs["gwas"]:
            if min(held[1], b_) > max(held[0], a_):
                ax.axvspan(max(held[0], a_), min(held[1], b_), color=col, alpha=0.30, zorder=1)
    ax.axhline(0, color=S.SPINE, lw=1)
    ax.scatter(probe / 1e6, 0, s=170, marker="D", color=col, edgecolor=S.INK, lw=1.6, zorder=6)
    ax.annotate(RSID[prot], (probe / 1e6, 0), xytext=(0.60, 0.70), textcoords="axes fraction", ha="left", fontsize=13,
                color=S.INK, arrowprops=dict(arrowstyle="-", color=S.INK, lw=1.0), zorder=14)
    ax.text(0.012, 0.95, "pQTL", transform=ax.transAxes, ha="left", va="top", fontsize=13, color=PQ)
    ax.text(0.012, 0.05, "GWAS", transform=ax.transAxes, ha="left", va="bottom", fontsize=13, color=GW_C)
    ax.set_ylim(-NROWS["gwas"] - 0.9, NROWS["pqtl"] + 0.9); ax.set_yticks([]); ax.set_xlim(z0, z1)
    ax.set_facecolor("#fbfbfb")
    ax.spines[["top", "right"]].set_visible(False)
    ax.xaxis.set_major_locator(plt.MaxNLocator(4))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v * 1000:,.0f}"))
    ax.tick_params(axis="x", labelsize=12, colors=col)
    ax.set_xlabel(f"Chromosome {chrom} position (kb)", fontsize=13, color=col)
    if first:
        ax.set_ylabel("Credible\nsets", fontsize=14)


def loci(fig, cells):
    man, trans, merged, cis, cis_sum, land, brow = R.load()
    lq, lg = R.load_lep()
    tq, tg = trans[trans.source == "pQTL"], trans[trans.source == "GWAS"]
    cq, cg = cis[cis.source == "pQTL"], cis[cis.source == "GWAS"]
    cis_probe = int(cq.loc[cq.neglog10p.idxmax(), "pos"])
    spec = [("C", tq, tg, R.CHR11_INSTRUMENT, {"CHRDL2": R.CHRDL2_SPAN_MB, "POLD3": R.POLD3_SPAN_MB}, R.CHR11_INSTRUMENT_RSID),
            ("D", lq, lg, R.LEP_INSTRUMENT, {"FTO": R.FTO_SPAN_MB}, R.LEP_INSTRUMENT_RSID),
            ("E", cq, cg, cis_probe, {"BMP4": R.BMP4_SPAN_MB}, "top cis variant")]
    slot = None
    for k in ("pqtl", "gwas"):
        NROWS[k] = max(1 + max([r for _, r in credible_sets(pr, ZOOM[t])[k]], default=-1)
                       for t, pr in (("C", "BMP4"), ("D", "LEP")))
    for j, (cell, (tag, q_, g_, probe, genes, ptr)) in enumerate(zip(cells, spec)):
        chrom, _, prot, kind, col, cancer = LOCI[tag]
        sub = cell.subgridspec(3, 1, hspace=0.30, height_ratios=[1, 1, 0.55])
        a1 = fig.add_subplot(sub[0]); a2 = fig.add_subplot(sub[1])   # not shared: own tick labels
        track(a1, q_, probe, "pQTL", f"{prot}-{kind}", col, 50.0, genes, j == 0, chrom, col, ptr)
        track(a2, g_, probe, "GWAS", cancer, col, 30.0, genes, j == 0, chrom, col, ptr)
        a2.set_xlim(a1.get_xlim())
        two_tone(a1, 0.0, 1.06, [(f"{prot}-{kind}", col), (" × ", INK_K), (cancer, col)], S.FS["annot"] + 3)
        S.letter(fig, a1, tag, dx=-0.035 if j == 0 else -0.012, dy=0.016)
        if tag in ZOOM:
            a3 = fig.add_subplot(sub[2])
            box = a3.get_position(); a3.set_position([box.x0, box.y0 - 0.035, box.width, box.height])
            susie(a3, a2, prot, probe, col, ZOOM[tag], j == 0, chrom)
            z0, z1 = ZOOM[tag]                       # GWAS-track tick labels inside the zoom window are shown below
            fig.canvas.draw()
            a2.set_xticks([v for v in a2.get_xticks() if not (z0 - 0.02 <= v <= z1 + 0.02)
                           and a2.get_xlim()[0] <= v <= a2.get_xlim()[1]])
            a2.set_xlabel(f"Chromosome {chrom} position" + chr(10) + "(Mb)", fontsize=13, color=col, loc="right", labelpad=2)
        else:
            slot = sub[2]
    return slot


def legend(fig, slot):
    ax = fig.add_subplot(slot); ax.axis("off")
    h = [Line2D([], [], ls="none", marker="o", ms=11, color=R.DIST_COL[b], label=R.BIN_LABEL[b]) for b in R.BINS]
    h.append(Line2D([], [], ls="none", marker="D", ms=11, mfc="white", mec=S.INK, mew=1.6, label="instrument"))
    ax.legend(handles=h, loc="center", ncol=3, frameon=False, fontsize=S.FS["legend"] - 4,
              title="Distance from the instrument", title_fontsize=S.FS["legend"] - 4, columnspacing=1.2, handletextpad=0.3)


def main():
    fig = plt.figure(figsize=(26, 24.5))
    gs = fig.add_gridspec(2, 1, height_ratios=[0.75, 1.25], hspace=0.24, left=0.06, right=0.985, top=0.965, bottom=0.05)
    top = gs[0].subgridspec(1, 6, wspace=0.55)
    axA = fig.add_subplot(top[0, 0:2]); panel_a(axA)
    axB = manhattans(fig, top[0, 2:6])
    S.letter(fig, axA, "A", dx=-0.045, dy=0.008)
    S.letter(fig, axB, "B", dx=-0.035, dy=0.008)
    bot = gs[1].subgridspec(1, 3, wspace=0.16)
    legend(fig, loci(fig, [bot[0, 0], bot[0, 1], bot[0, 2]]))
    for c in (1, 2):                                  # light guides: read C-E by column
        x0 = bot[0, c - 1].get_position(fig).x1; x1 = bot[0, c].get_position(fig).x0
        xm = (x0 + x1) / 2 + 0.006
        y0, y1 = bot[0, c].get_position(fig).y0 - 0.02, bot[0, c].get_position(fig).y1 + 0.03
        fig.add_artist(Line2D([xm, xm], [y0, y1], color="#d0d0d0", lw=1.6))
    S.save(fig, "fig7_v7n_genetics.png")


if __name__ == "__main__":
    main()
