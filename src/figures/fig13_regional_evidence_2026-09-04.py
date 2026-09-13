#!/usr/bin/env python3
"""
fig13_locus_story_mosaic.py — SLIDE 13 rebuild: the BMP4 x CRC locus story,
told once, in reading order.

WHY THIS REPLACES fig13_smr_coloc_mosaic.py
--------------------------------------------
The previous slide-13 mosaic argued against itself. Reviewed 2026-08-31:

  1. NO ESTABLISHING SHOT. It opened directly on a 1 Mb zoom.
  2. THE LOUDEST ANNOTATION UNDERCUT THE FINDING. A double-headed arrow
     reading "instrument-to-lead 200.8 kb apart" was the most prominent
     element on the panel meant to ESTABLISH the result. That framing is
     also wrong — see THE PLATEAU below.
  3. THE BETA-BETA SCATTER RAISED QUESTIONS IT COULD NOT ANSWER. r = 0.46
     across 685 SNPs with 55% of them piled at the origin. Fixed, see
     THE EFFECT PLANE below.
  4. THE TWO LOCUS PANELS LOOKED ALIKE, so the central contrast was
     invisible.

THE CONTRAST THIS FIGURE IS BUILT AROUND
-----------------------------------------
Both windows carry a strong CRC GWAS signal (chr11q13.4 p = 1.4e-30;
chr14 BMP4 p = 2.9e-26), so the chr14 control is NOT a "nothing here"
region — it is a real cancer locus driven by a DIFFERENT variant. The
comparison is therefore visual, and needs no statistics:

    chr11: the cancer signal is present where the protein signal is.
    chr14: it is not.

Quantified at each column's probed variant (its own pQTL peak):
    chr11   CRC p = 6.1e-30   —  98% of that region's peak height
    chr14   CRC p = 0.78      —   0% of that region's peak height

THE PLATEAU (recomputed 2026-08-31)
------------------------------------
The chr11 CRC signal is a PLATEAU, not a peak: 38 SNPs above -log10p = 25
spanning 353 kb. The instrument sits at 97.9% of peak height. Which single
SNP is nominally "tallest" inside that plateau is noise, so the old
"200.8 kb from the lead" caveat measured distance to an arbitrary point.
Consequence: neither track marks its own tallest SNP. Both tracks in a
column mark the SAME probed variant, because the question colocalization
asks is "is there a cancer signal where the protein signal is?"

THE EFFECT PLANE (panel c, ported from fig14_panels.py's 15d)
-------------------------------------------------------------
The old panel plotted every SNP within 100 kb regardless of whether it
carried any pQTL signal — mostly noise, which is why r looked like 0.46.
Restricted to actual instruments at the standard genome-wide line
(p_pQTL < 5e-8): n = 178, only 4% near the origin, r = 0.974. The
threshold is the same 5e-8 already drawn on every other panel here, not a
tuned cosmetic choice.

PANEL MAP — reads top to bottom: method -> result -> zoom -> caveat
-------------------------------------------------------------------
  a | b | c   filter cascade (slide 14b) | decision plane (old 13b) |
              effect plane (slide 15d)        ... what was done
  d           shared genome-wide CRC Manhattan, BOTH windows flagged
  e | f       chr11q13.4 | chr14 BMP4 gene — each a two-track stack
              (plasma BMP4 above, colorectal cancer below)
  g           WHY a chr11 variant moves a chr14 protein

COLOUR — three independent channels, so nothing collides:
  point hue         distance from the probed variant. STANDARD RAMP,
                    identical to fig15_panels.py, so slide 16 needs no
                    change and 13/16 read as one system.
  track tint/label  which trait (pQTL teal / GWAS rose)
  header + badge    which region (chr11 VIOLET = the finding,
                    chr14 grey = control) — this colour means ONLY
                    "chr11 / the BMP4xCRC survivor", nowhere else. An
                    earlier version also used a near-identical purple for
                    the CHRDL2 gene box, which made a specific gene look
                    like the same thing as the finding itself. CHRDL2 now
                    has its own colour (sienna, GENE_COL), never violet.

WHAT IS DELIBERATELY NOT ON THIS FIGURE
----------------------------------------
Relocated to caption / limitations, NOT hidden — see
`figurev5/SLIDE_13_REBUILD_2026-08-31.md`: the 200.8 kb arrow, the
"tallest pQTL SNP not used" provenance note, and the four-line disclaimer
stack that previously ran above the result.

PANEL G CARRIES THE MECHANISM, NOT JUST THE LABEL. Asserting "BMP4 is on
chr14" beside "the signal is on chr11 inside CHRDL2" reads cold as a plain
error. The panel now draws the route: CHRDL2 is a secreted protein that
binds BMP4 in circulation, so a chr11 variant altering CHRDL2 alters how
much BMP4 the assay detects. That is what a trans-pQTL is. The same
mechanism is why attribution stays open, and the panel says both halves.

METHOD NAMING (updated 2026-09-04): the 316-pair screen behind panel a
still uses a single-SNP Wald ratio and coloc.abf, not official SMR/HEIDI.
BMP4 x CRC specifically was promoted out of that screen and re-analysed
with the real SMR 1.3.1 software and its HEIDI test — that is what panel
e's badge now reports, and it did not resolve heterogeneity.

Sources (derived CSVs already in the repo — no raw archive needed):
    revise_plan/smr_coloc/11_figures/source_fig12_crc_manhattan.csv
    revise_plan/smr_coloc/11_figures/source_fig12_bmp4_crc_locus.csv
    revise_plan/smr_coloc/11_figures/source_fig12_bmp4_crc_merged.csv
    revise_plan/smr_coloc/11_figures/source_fig_bmp4_cis_locus.csv
    revise_plan/smr_coloc/11_figures/source_fig_bmp4_cis_summary.csv
    revise_plan/smr_coloc/11_figures/source_fig02_v11_evidence_landscape.csv
    revise_plan/smr_coloc/11_figures/source_fig05_v11_causal_forest.csv
Output: figurev5/output/fig13_locus_story_mosaic.{png,pdf}
Run:    python fig13_locus_story_mosaic.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.path import Path as MplPath
import numpy as np
import pandas as pd

ROOT = Path("e:/Proteomics")
SRC = ROOT / "revise_plan" / "smr_coloc" / "11_figures"
QC  = ROOT / "revise_plan" / "smr_coloc" / "official_smr_heidi" / "qc"
OUT = ROOT / "figurev5" / "output"
SCREEN = (ROOT / "revise_plan" / "smr_coloc" / "official_smr_heidi" /
          "screen" / "tier1_combined_316_two_term.csv")
OUT.mkdir(parents=True, exist_ok=True)

DPI = 300
GW_LINE = 5e-8

# The instrument actually used by the Wald ratio: rs61389091. NOT the stored
# lead_pos (74,721,142), which is absent from the merged pQTL x GWAS file and
# is a known provenance bug in the source table. See the 2026-08-18 audit.
CHR11_INSTRUMENT = 74_716_876
CHR11_INSTRUMENT_RSID = "rs61389091"

# LEP x BRC instrument, added 2026-09-04. Position confirmed as the pQTL
# peak in the accepted QC table (16:53772541, p = 8.5e-13).
LEP_INSTRUMENT = 53_772_541
LEP_INSTRUMENT_RSID = "rs56094641"

# GRCh38 gene spans, confirmed against NCBI Gene.
BMP4_SPAN_MB = (53.949736, 53.956891)      # chr14 — the gene itself
CHRDL2_SPAN_MB = (74.696429, 74.731426)    # chr11 — holds the instrument
POLD3_SPAN_MB = (74.592582, 74.669341)     # chr11 — adjacent, own CRC signal
# chr16 — holds the LEP instrument in intron 1. NOT yet confirmed against
# NCBI Gene the way the three spans above were; flagged in the changelog.
# Verify before submission.
FTO_SPAN_MB = (53.703963, 54.121941)

# THREE INDEPENDENT ENCODINGS, one per question. An earlier draft recoloured
# the POINTS warm/cool per region, which broke consistency with slide 16 and
# spent the most valuable channel on the least information. Corrected:
#
#   point hue        = distance from the probed variant  (STANDARD RAMP,
#                      identical to fig15_panels.py, so slide 16 needs no
#                      change and the two slides read as one system)
#   track tint+label = which trait   (pQTL / GWAS)
#   header + badge   = which region  (chr11 finding / chr14 control)
#
# Because region colour appears only in headers, badges and the Manhattan
# window shading — never inside a point cloud — it cannot collide with the
# distance ramp.

# STANDARD distance ramp — values copied verbatim from fig15_panels.py's
# DIST_COL. Do not diverge from it: slide 13 and slide 16 must match.
BINS = ["<50", "50-100", "100-200", "200-350", ">350"]
DIST_COL = {"<50": "#c9352c", "50-100": "#f0a83a", "100-200": "#57b95b",
            "200-350": "#4a9fd4", ">350": "#3b4d61"}
BIN_LABEL = {"<50": "< 50 kb", "50-100": "50–100 kb", "100-200": "100–200 kb",
             "200-350": "200–350 kb", ">350": "> 350 kb"}

# trait identity — the project's existing pQTL/GWAS pair, from
# fig13_smr_coloc_mosaic.py's C_PQTL / C_GWAS
C_PQTL = "#1f6f8b"
C_GWAS = "#c1666b"
TRAIT_COL = {"pQTL": C_PQTL, "GWAS": C_GWAS}
TRAIT_BG = {"pQTL": "#f4fafc", "GWAS": "#fdf6f6"}
# Per-locus trait labels. These were hardcoded to BMP4/CRC, which silently
# mislabelled the LEP x BRC panel when it was added on 2026-09-04.
TRAIT_LABEL = {"pQTL": "plasma BMP4 pQTL", "GWAS": "colorectal cancer GWAS"}
LABELS_BMP4 = {"pQTL": "plasma BMP4 pQTL", "GWAS": "colorectal cancer GWAS"}
LABELS_LEP  = {"pQTL": "plasma LEP pQTL",  "GWAS": "breast cancer GWAS"}

# region identity. MUST be visually distinct from BOTH the trait pair
# (pQTL teal #1f6f8b / GWAS rose #c1666b) AND the DIST_COL ramp (red / orange
# / green / blue / dark-slate) — the original crimson + blue-grey pair
# collided with both, so a reader could not tell "which chromosome" from
# "which trait" apart. Violet and neutral grey sit outside every other hue
# used in this figure.
#
# C_FIND HAS EXACTLY ONE MEANING: the BMP4 x CRC survivor / the 11q13.4
# locus it defines. It is used, and ONLY used, for: the cascade's final
# stage, the decision-plane star, the effect-plane slope+lead SNP, and
# panel d/e's chr11 header/badge/window. A first draft ALSO used a
# near-identical purple (#6a4c93) for the CHRDL2 gene box in panels e-g,
# so a reader could not tell "the finding" from "one specific gene inside
# it" apart — the exact confusion this palette is designed to prevent.
# CHRDL2 now gets its own colour (GENE_COL, below), unrelated to C_FIND.
C_FIND = "#7c3fa0"    # chr11 / the BMP4 x CRC finding — violet, and ONLY that
C_CTRL = "#6b6f76"    # chr14 — the control — true grey, reads as "no signal"

# Per-gene identity, used consistently in EVERY panel that draws a gene span
# (e, f, g) — one colour per gene, none of them C_FIND, so "which gene" and
# "is this the finding" are never the same visual question.
GENE_COL = {
    "BMP4": C_CTRL,           # chr14 — ties to the control column
    "CHRDL2": "#8a5a2e",      # sienna — deliberately NOT purple/orange/red,
                              # so it cannot be misread as C_FIND or the
                              # DIST_COL <50/50-100 ramp stops sitting right
                              # next to it in panel e
    "POLD3": "#4c7a4c",       # green — matches its own box in panel g
    "FTO": "#8a5a2e",         # sienna, same role as CHRDL2: the gene that
                              # HOLDS the instrument. Deliberately the same
                              # colour as CHRDL2 rather than a new hue,
                              # because it plays the identical part in the
                              # trans story and a new colour would imply a
                              # distinction that isn't there.
}
C_INK = "#222222"
C_MUTE = "#6b7680"
DROP_COL = "#c9c9c9"

plt.rcParams.update({
    "font.family": "Arial", "font.size": 11, "font.weight": "bold",
    "axes.labelweight": "bold", "axes.titleweight": "bold",
    "axes.linewidth": 1.2, "xtick.major.width": 1.2, "ytick.major.width": 1.2,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
})


def load_lep():
    """LEP x BRC chr16 regional tracks, built from the official-SMR QC
    tables (the accepted, allele-matched variant sets actually used).

    There is no pre-derived source_fig* CSV for this locus the way there is
    for BMP4, so the two tracks are assembled here into the same shape
    _track expects: pos, neglog10p, dist_kb.
    """
    q = pd.read_csv(QC / "LEP_BRC_pqtl_accepted.csv")
    g = pd.read_csv(QC / "LEP_BRC_gwas_accepted.csv")
    out = []
    for d, poscol in ((q, "Bp"), (g, "source_pos")):
        t = pd.DataFrame({
            "pos": d[poscol].astype(int),
            "neglog10p": -np.log10(d["p"].clip(lower=1e-300)),
        })
        t["dist_kb"] = (t["pos"] - LEP_INSTRUMENT).abs() / 1000.0
        out.append(t)
    return out[0], out[1]


def load():
    man = pd.read_csv(SRC / "source_fig12_crc_manhattan.csv")
    man["logp"] = -np.log10(man["p"].clip(lower=1e-300))
    trans = pd.read_csv(SRC / "source_fig12_bmp4_crc_locus.csv")
    merged = pd.read_csv(SRC / "source_fig12_bmp4_crc_merged.csv")
    cis = pd.read_csv(SRC / "source_fig_bmp4_cis_locus.csv")
    cis_sum = pd.read_csv(SRC / "source_fig_bmp4_cis_summary.csv").iloc[0]
    land = pd.read_csv(SRC / "source_fig02_v11_evidence_landscape.csv")
    forest = pd.read_csv(SRC / "source_fig05_v11_causal_forest.csv")
    brow = forest[(forest["protein"] == "BMP4")
                  & (forest["cancer"] == "CRC")].iloc[0]
    return man, trans, merged, cis, cis_sum, land, brow


# ─────────────────────────────────────────── panel a: shared genome Manhattan
def panel_genome(ax, man):
    """ONE CRC genome-wide Manhattan, BOTH windows flagged.

    Full width, and the y-axis is capped at a FIXED 42, not scaled to the
    genome max (113.6, at chr18) — see the cap note inside the function.
    Scaling to the true max buried both flagged windows (29.8, 25.5) and the
    entire rest of the genome in the bottom quarter of the plot, which is
    what "the y-axis is barely visible" meant: not that the panel was too
    short, but that its own range made everything in it too short.
    """
    man = man.sort_values(["chr", "bp"]).copy()
    offs, cum = {}, 0.0
    for c in range(1, 23):
        offs[c] = cum
        sub = man[man["chr"] == c]
        cum += (sub["bp"].max() if len(sub) else 0) * 1.02
    man["xcum"] = man["bp"] + man["chr"].map(offs)

    alt = man["chr"] % 2 == 0
    ax.scatter(man.loc[alt, "xcum"], man.loc[alt, "logp"], s=0.9, c="#c3ccd4",
               linewidths=0, rasterized=True, zorder=2)
    ax.scatter(man.loc[~alt, "xcum"], man.loc[~alt, "logp"], s=0.9, c="#97a3ae",
               linewidths=0, rasterized=True, zorder=2)
    ax.axhline(-np.log10(GW_LINE), color="#c0392b", lw=0.9, ls="--", zorder=3)

    # FIXED, LOW cap — not the genome max. The true genome-wide max is 113.6
    # (chr18); scaling the axis to that (the previous version's
    # man["logp"].max() * 1.42) squeezed the bulk of the genome, and the two
    # windows this panel exists to show (29.8, 25.5), into the bottom quarter
    # of the plot. Five other loci exceed this cap (chr18 to 113.6, chr8 to
    # 92.8, chr15 to 59.0, chr11-elsewhere to 54.8, chr20 to 45.1) — they are
    # not the comparison this panel makes and are disclosed, not hidden, via
    # the note below rather than being drawn off-scale.
    ymax = 42.0
    n_clipped = int((man["logp"] > ymax).sum())
    clip_max = man["logp"].max()
    for chrom, mb, col, name, ptr, ytop in [
            (11, 74.716876, C_FIND, "chr11q13.4", "left column ↓", 0.97),
            (14, 53.953000, C_CTRL, "chr14 — BMP4 gene", "right column ↓", 0.60)]:
        centre = mb * 1e6 + offs[chrom]
        ax.axvspan(centre - 7e6, centre + 7e6, color=col, alpha=0.35, zorder=1)
        best = man[(man["chr"] == chrom)
                   & (man["bp"] / 1e6 > mb - 0.5)
                   & (man["bp"] / 1e6 < mb + 0.5)]["logp"].max()
        ax.annotate(f"{name}   ·   {ptr}\nbest CRC $p$ = {10**-best:.0e}",
                    xy=(centre, best), xytext=(centre, ymax * ytop),
                    ha="center", va="top", fontsize=9.2, fontweight="bold",
                    color=col, zorder=6, linespacing=1.35,
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=1.5,
                                    shrinkA=3, shrinkB=4))

    ax.set_xlim(0, man["xcum"].max())
    ax.set_ylim(0, ymax)
    ax.set_xticks([offs[c] + (man[man["chr"] == c]["bp"].max() / 2)
                   for c in range(1, 23)])
    ax.set_xticklabels([str(c) for c in range(1, 23)], fontsize=9.5)
    ax.set_xlabel("chromosome", fontsize=11, labelpad=5)
    ax.set_ylabel("CRC GWAS\n$-\\log_{10}\\ p$", fontsize=11)
    ax.tick_params(axis="y", labelsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    # Disclosure for the cap, not a hidden truncation. Every position INSIDE
    # the axes collides with something: bottom-right sits in the densest
    # part of the point cloud (most SNPs, genome-wide, are near y=0);
    # top-right collided with the chr11 callout, which spans nearly the
    # full plot height by design. Placed OUTSIDE the data area entirely, in
    # the margin below the x-axis, where nothing else is drawn.
    ax.text(1.0, -0.32,
            f"$y$-axis capped at {ymax:.0f} for legibility  ·  {n_clipped} SNPs "
            f"run higher elsewhere (to $p$≈1e−{clip_max:.0f}, chr18) — not "
            "part of this comparison",
            transform=ax.transAxes, ha="right", va="top", fontsize=8,
            fontweight="bold", color="#9aa4ad")
    ax.set_title("d   Both regions carry a real colorectal cancer signal — "
                 "the question is which one also moves plasma BMP4",
                 loc="left", fontsize=13, fontweight="bold", pad=9)


# ─────────────────────────────────── panels b/c: two-track locus stacks
def _track(ax, s, src, probe_pos, probe_col, show_x, chrom, ylim_top,
           labels=None):
    """One track of a column's stack — the slide-16 locus style, unchanged.

    Points use the STANDARD distance ramp shared with fig15_panels.py, so
    both slides read as one system. Which trait a track shows is carried by
    its background tint and label colour, not by recolouring the data.

    Both tracks in a column use the SAME probe (that column's pQTL peak) and
    NEITHER marks its own tallest SNP — see THE PLATEAU in the module
    docstring. The annotation states the signal strength AT the probe, which
    is the quantity colocalization actually uses.

    probe_col: the marker was a single fixed dark red in both columns, which
    read as "the same SNP" even though chr11's diamond is the real
    Wald-ratio instrument and chr14's is an unrelated, weaker candidate SNP
    on a different chromosome. Now tied to region identity — C_FIND
    (violet) for the real instrument, C_CTRL (grey) for the checked
    candidate — so the marker itself, not just the badge, says which is
    which.
    """
    ax.set_facecolor(TRAIT_BG[src])
    tcol = TRAIT_COL[src]

    s = s.copy()
    dkb = (s["pos"] - probe_pos).abs() / 1000
    s["bin"] = pd.cut(dkb, bins=[-np.inf, 50, 100, 200, 350, np.inf],
                      labels=BINS)
    for key in BINS[::-1]:
        m = s[s["bin"] == key]
        if len(m):
            ax.scatter(m["pos"] / 1e6, m["neglog10p"], s=7,
                       c=DIST_COL[key], linewidths=0, alpha=0.9,
                       zorder=3 + BINS.index(key))

    at = np.nan
    if len(s):
        row = s.loc[(s["pos"] - probe_pos).abs().idxmin()]
        at = float(row["neglog10p"])
        ax.scatter([probe_pos / 1e6], [at], s=95, marker="D",
                   facecolor=probe_col, edgecolor="white", linewidth=1.3,
                   zorder=10)
    ax.axvline(probe_pos / 1e6, color=probe_col, lw=1.0, ls=":", zorder=2)
    ax.axhline(-np.log10(GW_LINE), color="#c0392b", lw=0.9, ls="--", zorder=2)

    peak = s["neglog10p"].max() if len(s) else np.nan
    pct = (at / peak * 100) if peak and peak > 0 else np.nan
    # SHOW, DON'T TELL: one compact line — trait name, the number, the %.
    # An earlier version spelled out "at the probed variant ... of this
    # region's peak" as a run-on sentence at 8.4pt, which is where most of
    # the "impossible to read" text sat.
    ax.text(0.985, 0.93, (labels or TRAIT_LABEL)[src], transform=ax.transAxes,
            ha="right", va="top", fontsize=10.5, fontweight="bold",
            color=tcol, zorder=11)
    ax.text(0.985, 0.80, f"$p$ = {10 ** -at:.0e}   ·   {pct:.0f}% of peak",
            transform=ax.transAxes, ha="right", va="top", fontsize=9.2,
            fontweight="bold", color=tcol, zorder=11)

    ax.set_ylim(0, ylim_top)
    ax.set_ylabel(f"{src}\n$-\\log_{{10}}\\ p$", fontsize=8.6, color=tcol)
    ax.tick_params(axis="y", colors=tcol, labelsize=8.2)
    ax.spines["left"].set_color(tcol)
    ax.spines["left"].set_linewidth(2.0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color="#ffffff", lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelsize=8.2)
    if show_x:
        ax.set_xlabel(f"chr{chrom} position (Mb)", fontsize=9.2)
    else:
        plt.setp(ax.get_xticklabels(), visible=False)
    return at


def _genes(ax, spans):
    """Gene spans, coloured from the shared GENE_COL map — never C_FIND, so a
    specific gene is never visually mistaken for "the finding" itself."""
    for name, (g0, g1) in spans.items():
        col = GENE_COL[name]
        ax.axvspan(g0, g1, color=col, alpha=0.15, zorder=0)
        ax.text((g0 + g1) / 2, ax.get_ylim()[1] * 0.97, name, ha="center",
                va="top", fontsize=8.2, fontweight="bold", color=col,
                style="italic", zorder=6, clip_on=True)


def _verdict(ax, text, sub, color):
    ax.text(0.022, 0.955, text, transform=ax.transAxes, ha="left", va="top",
            fontsize=12, fontweight="bold", color="white", zorder=20,
            bbox=dict(boxstyle="round,pad=0.40", facecolor=color,
                      edgecolor="none"))
    ax.text(0.028, 0.80, sub, transform=ax.transAxes, ha="left", va="top",
            fontsize=9, fontweight="bold", color=color, zorder=20)


# ──────────────────────────────────── panel d: filter cascade (slide 14b)
def _ribbon(ax, x0, x1, y0_lo, y0_hi, y1_lo, y1_hi, color, alpha=0.55):
    xm = (x0 + x1) / 2.0
    verts = [(x0, y0_lo), (xm, y0_lo), (xm, y1_lo), (x1, y1_lo), (x1, y1_hi),
             (xm, y1_hi), (xm, y0_hi), (x0, y0_hi), (x0, y0_lo)]
    codes = [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
             MplPath.LINETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
             MplPath.CLOSEPOLY]
    ax.add_patch(mpatches.PathPatch(MplPath(verts, codes), facecolor=color,
                                    edgecolor="none", alpha=alpha, zorder=2))


def panel_decision_plane(ax, scr):
    """All 316 analysable pairs at once, so the two promoted ones are SHOWN
    sitting apart rather than asserted in a list.

    Replaces both retired panels (the 257-pair cascade and the old decision
    plane) with one plane built on the corrected screen table. An earlier
    2026-09-04 version of this figure used a text block here; that was
    telling, not showing, and this project already worked out in the
    2026-08-18 entry that a bar or a plane reads in one look where a list
    does not.

    Axes: colocalisation posterior against Wald evidence. The promoted pair
    is the one that has to be high on BOTH, which a plane shows and a
    cascade cannot.
    """
    bonf = float(scr["bonferroni_threshold_316"].iloc[0])
    d = scr.dropna(subset=["PP.H4", "p_wald_two_term"]).copy()
    d["y"] = -np.log10(d["p_wald_two_term"].clip(lower=1e-300))
    passes = d["passes_two_term_bonferroni_descriptive"] == True   # noqa: E712
    promoted = passes & (d["PP.H4"] > 0.80)
    near = passes & ~promoted
    rest = ~passes

    ax.axhspan(-np.log10(bonf), 40, xmin=(0.80 + 0.02) / 1.04, color=C_FIND,
               alpha=0.08, zorder=0)
    ax.scatter(d.loc[rest, "PP.H4"], d.loc[rest, "y"], s=13, c="#c3ccd4",
               edgecolors="none", alpha=0.85, zorder=2,
               label=f"neither threshold  (n={int(rest.sum())})")
    ax.scatter(d.loc[near, "PP.H4"], d.loc[near, "y"], s=62, c="#c47a2c",
               edgecolors="white", linewidths=0.8, zorder=3,
               label=f"Wald only  (n={int(near.sum())})")
    ax.scatter(d.loc[promoted, "PP.H4"], d.loc[promoted, "y"], s=210,
               c=C_FIND, marker="*", edgecolors="white", linewidths=1.1,
               zorder=5, label=f"promoted  (n={int(promoted.sum())})")
    ax.axhline(-np.log10(bonf), color="#33414d", lw=0.9, ls="--", zorder=1)
    ax.axvline(0.80, color="#33414d", lw=0.9, ls="--", zorder=1)
    ax.text(0.03, -np.log10(bonf) + 0.9, f"Bonferroni  $p$ < {bonf:.1e}",
            fontsize=7.8, fontweight="bold", color="#33414d")
    ax.text(0.815, 1.2, "PP.H4 > 0.80", fontsize=7.8, fontweight="bold",
            color="#33414d", rotation=90, va="bottom")

    for lab, dx, dy_ in [("BMP4 × CRC", -78, 6), ("LEP × BRC", -74, -30)]:
        prot = lab.split(" ")[0]
        r = d[(d["protein"] == prot) & promoted]
        if len(r):
            r = r.iloc[0]
            ax.annotate(lab, xy=(r["PP.H4"], r["y"]), xytext=(dx, dy_),
                        textcoords="offset points", fontsize=9.5,
                        fontweight="bold", color=C_FIND, ha="left",
                        arrowprops=dict(arrowstyle="-", color=C_FIND, lw=1.0))
    mm = d[near]
    if len(mm):
        r = mm.iloc[0]
        ax.annotate(f"{r['protein']} × {r['cancer']}\nno shared signal",
                    xy=(r["PP.H4"], r["y"]), xytext=(24, -6),
                    textcoords="offset points", fontsize=8.2,
                    fontweight="bold", color="#c47a2c", ha="left",
                    linespacing=1.35,
                    arrowprops=dict(arrowstyle="-", color="#c47a2c", lw=0.9))

    ax.set_xlim(-0.04, 1.04)
    ax.set_ylim(-1.0, 40)
    ax.set_xlabel("colocalization posterior, PP.H4", fontsize=9)
    ax.set_ylabel("Wald evidence,  $-\\log_{10}\\ p$", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper center", frameon=False, fontsize=8.2,
              handletextpad=0.4, borderpad=0.2)
    ax.set_title("a   316 pairs screened — two clear both thresholds",
                 loc="left", fontsize=11.5, fontweight="bold", pad=6)


# ──────────────────────────────── panel f: effect plane (slide 15d)
def panel_effect(ax, merged, brow):
    """Per-SNP CRC effects against BMP4 pQTL effects, instruments only.

    THRESHOLD NOTE: restricted to p_pQTL < 5e-8, the standard genome-wide
    line already drawn on every other panel here. The superseded panel used
    every SNP within 100 kb regardless of pQTL signal — 685 points, 55% of
    them at the origin, r = 0.46 — which measured mostly noise. On actual
    instruments: n = 178, 4% near the origin, r = 0.974. This is a
    principled restriction to variants that are instruments, not a tuned
    cosmetic choice.

    Alignment with the slope is compatible with proportional effects. It
    does NOT distinguish mediation from horizontal pleiotropy, and the
    title says so.
    """
    d = merged[merged["p_pqtl"] < GW_LINE].copy()
    lead = merged.loc[merged["p_pqtl"].idxmin()]
    b = float(brow["b_SMR"])
    r = np.corrcoef(d["b_pqtl"], d["b_gwas"])[0, 1]

    ax.axhline(0, color="#d5dbe0", lw=0.9, zorder=1)
    ax.axvline(0, color="#d5dbe0", lw=0.9, zorder=1)
    xs = np.array([d["b_pqtl"].min(), d["b_pqtl"].max()]) * 1.12
    ax.plot(xs, b * xs, color=C_FIND, lw=2.0, zorder=4,
            label=f"Wald slope $b$ = {b:.2f}")
    ax.errorbar(d["b_pqtl"], d["b_gwas"], yerr=d["se_gwas"], xerr=d["se_pqtl"],
                fmt="none", ecolor="#dde3e8", elinewidth=0.6, zorder=2)
    sc = ax.scatter(d["b_pqtl"], d["b_gwas"], s=22, c=d["dist_kb"],
                    cmap="viridis_r", linewidths=0.3, edgecolor="white",
                    zorder=5)
    ax.scatter([lead["b_pqtl"]], [lead["b_gwas"]], s=140, marker="D",
               facecolor=C_FIND, edgecolor="white", linewidth=1.4, zorder=7,
               label="lead pQTL SNP")
    cb = plt.colorbar(sc, ax=ax, fraction=0.040, pad=0.02)
    cb.set_label("distance from lead (kb)", fontsize=8.4, fontweight="bold")
    cb.ax.tick_params(labelsize=7.6)

    ax.text(0.035, 0.965, f"$r$ = {r:.2f}\n{len(d)} genome-wide\nsignificant pQTL SNPs",
            transform=ax.transAxes, ha="left", va="top", fontsize=8.6,
            fontweight="bold", color=C_INK, linespacing=1.4)
    ax.set_xlabel("BMP4 pQTL effect  ($b_{pQTL}$)", fontsize=9)
    ax.set_ylabel("CRC GWAS effect  ($b_{GWAS}$)", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color="#f4f4f4", lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.94)
    ax.set_title("b   BMP4 × CRC: the same variants move both traits, proportionally",
                 loc="left", fontsize=11.5, fontweight="bold", pad=6)


# ──────────────────────────────────────────────── panel g: gene context
def panel_genes(ax):
    """WHY a chr11 variant moves a chr14 protein — the mechanism, drawn.

    An earlier version of this panel just asserted the two facts side by
    side: "BMP4 is on chr14" and "the signal is on chr11, inside CHRDL2".
    Read cold that looks like a plain error — the reviewer's reaction is
    "you pointed at the wrong chromosome and called it BMP4". The panel has
    to carry the RATIONALE, because there is a real one: CHRDL2 is a
    secreted protein that physically binds BMP4 in circulation, so a variant
    changing CHRDL2 changes how much BMP4 the assay sees, without BMP4's own
    gene being involved. That is what a trans-pQTL is, and they are common
    in plasma proteomics.

    The same mechanism is exactly why attribution stays open, and the panel
    says both halves. Conceding the competing explanation in the figure is
    what turns an apparent blunder into a stated limitation.
    """
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # ── chr14 track: the gene itself ────────────────────────────────────
    ax.text(0.1, 9.3, "chr14", fontsize=8.8, fontweight="bold", color=C_CTRL)
    ax.add_patch(mpatches.Rectangle((1.5, 8.5), 4.6, 0.46,
                                    facecolor="#e6eaec", edgecolor="none"))
    ax.add_patch(mpatches.Rectangle((3.4, 8.5), 0.62, 0.46, facecolor=C_CTRL,
                                    edgecolor="none"))
    ax.text(3.71, 8.24, "BMP4", ha="center", va="top", fontsize=8.2,
            fontweight="bold", color=C_CTRL, style="italic")
    ax.annotate("", xy=(7.4, 8.73), xytext=(6.3, 8.73),
                arrowprops=dict(arrowstyle="-|>", color=C_CTRL, lw=1.6))
    ax.text(7.6, 8.73, "BMP4 protein\nsecreted into plasma", ha="left",
            va="center", fontsize=8.2, fontweight="bold", color=C_CTRL,
            linespacing=1.35)

    # ── chr11 track: the instrument, in a different gene ────────────────
    ax.text(0.1, 6.5, "chr11", fontsize=8.8, fontweight="bold", color=C_FIND)
    ax.add_patch(mpatches.Rectangle((1.5, 5.7), 4.6, 0.46,
                                    facecolor="#fbeaea", edgecolor="none"))
    ax.add_patch(mpatches.Rectangle((2.1, 5.7), 1.05, 0.46,
                                    facecolor=GENE_COL["POLD3"], edgecolor="none"))
    ax.text(2.62, 5.44, "POLD3", ha="center", va="top", fontsize=8.2,
            fontweight="bold", color=GENE_COL["POLD3"], style="italic")
    ax.add_patch(mpatches.Rectangle((3.45, 5.7), 0.85, 0.46,
                                    facecolor=GENE_COL["CHRDL2"], edgecolor="none"))
    ax.text(3.88, 5.44, "CHRDL2", ha="center", va="top", fontsize=8.2,
            fontweight="bold", color=GENE_COL["CHRDL2"], style="italic")
    ax.scatter([3.88], [6.55], marker="v", s=78, c=C_FIND, zorder=5)
    ax.text(4.10, 6.62, CHR11_INSTRUMENT_RSID, ha="left", va="center",
            fontsize=8.2, fontweight="bold", color=C_FIND)
    ax.annotate("", xy=(7.4, 5.93), xytext=(6.3, 5.93),
                arrowprops=dict(arrowstyle="-|>", color=GENE_COL["CHRDL2"],
                                lw=1.6))
    ax.text(7.6, 5.93, "CHRDL2 protein\nsecreted BMP antagonist", ha="left",
            va="center", fontsize=8.2, fontweight="bold",
            color=GENE_COL["CHRDL2"], linespacing=1.35)

    # ── the binding step: the whole explanation ─────────────────────────
    # vertical arrow from the CHRDL2 protein up to the BMP4 protein, so the
    # route reads as a physical interaction rather than a floating remark
    # sits immediately right of the two protein labels: placed further out it
    # read as a floating mark rather than a link between them
    ax.annotate("", xy=(9.85, 8.45), xytext=(9.85, 6.20),
                arrowprops=dict(arrowstyle="-|>", color=GENE_COL["CHRDL2"],
                                lw=2.0))
    ax.text(10.10, 7.32, "binds BMP4\nin circulation", ha="left", va="center",
            fontsize=8.4, fontweight="bold", color=GENE_COL["CHRDL2"],
            linespacing=1.35)
    ax.annotate("", xy=(13.7, 7.32), xytext=(12.5, 7.32),
                arrowprops=dict(arrowstyle="-|>", color=C_FIND, lw=2.0))
    ax.text(13.95, 7.32, "so a chr11 variant changes\nMEASURED plasma BMP4",
            ha="left", va="center", fontsize=8.8, fontweight="bold",
            color=C_FIND, linespacing=1.4)

    # SHOW, DON'T TELL: the diagram above already draws the mechanism (binds
    # BMP4 -> changes measured plasma BMP4). Text adds only the two things the
    # diagram cannot: the name for this kind of signal, and the one caveat it
    # implies. An earlier version restated the diagram in two run-on
    # paragraphs at 8.8pt — cut to two short lines.
    ax.text(0.1, 3.55,
            "A $trans$-pQTL: the chr14 $BMP4$ locus itself does not "
            "colocalize (panel e), so this route through $CHRDL2$ is the "
            "signal.",
            ha="left", va="top", fontsize=10, fontweight="bold",
            color=C_INK, linespacing=1.5)
    ax.text(0.1, 2.15,
            "Because the route runs through $CHRDL2$, the cancer link may "
            "run through $CHRDL2$ or $POLD3$ rather than BMP4 — not "
            "separable on current data.",
            ha="left", va="top", fontsize=10, fontweight="bold",
            color=C_MUTE, linespacing=1.5)
    ax.set_title("f   Why a chr11 variant moves a chr14 protein — and why "
                 "that keeps the gene question open", loc="left",
                 fontsize=11.5, fontweight="bold", pad=4)


def main():
    man, trans, merged, cis, cis_sum, land, brow = load()
    scr = pd.read_csv(SCREEN)

    fig = plt.figure(figsize=(19.5, 15.2))
    # Reading order is top-to-bottom: what was done (a-c) -> where in the
    # genome (d) -> the comparison that carries the result (e, f) -> why the
    # gene question stays open (g). An earlier draft opened on the locus
    # zooms and left the screen to the bottom, so the figure had to be read
    # upward to make sense.
    #
    # Row 2 (panel d) does not need a large share of the height any more —
    # its FIXED y-axis cap (see panel_genome) is what fixed its legibility,
    # not its size. The room that freed goes to row 3 (e, f), the panels that
    # actually carry the result, and to trimming panel g's row now its prose
    # is two lines instead of two paragraphs (see panel_genes).
    gs = gridspec.GridSpec(3, 6, figure=fig,
                           height_ratios=[0.86, 1.30, 0.42],
                           hspace=0.40, wspace=0.62,
                           left=0.052, right=0.978, top=0.888, bottom=0.040)

    panel_decision_plane(fig.add_subplot(gs[0, 0:4]), scr)
    panel_effect(fig.add_subplot(gs[0, 4:6]), merged, brow)

    tq = trans[trans["source"] == "pQTL"]
    tg = trans[trans["source"] == "GWAS"]
    cq = cis[cis["source"] == "pQTL"]
    cg = cis[cis["source"] == "GWAS"]
    y_q = max(tq["neglog10p"].max(), cq["neglog10p"].max()) * 1.16
    y_g = max(tg["neglog10p"].max(), cg["neglog10p"].max()) * 1.16
    cis_probe = int(cq.loc[cq["neglog10p"].idxmax(), "pos"])

    # THREE stacks, not two (2026-09-04): both promoted loci plus the cis
    # control, so the two promoted pairs are shown side by side instead of
    # BMP4 alone. Both promoted loci share C_FIND because they share a
    # status — promoted, trans, unresolved. Introducing a second "finding"
    # hue would imply a distinction that does not exist, and would collide
    # with the distance ramp. Grey stays reserved for the control.
    lq, lg = load_lep()
    y_q = max(y_q, lq["neglog10p"].max() * 1.16)
    y_g = max(y_g, lg["neglog10p"].max() * 1.16)

    sub_d = gs[1, 0:2].subgridspec(2, 1, hspace=0.10)
    ax_d1, ax_d2 = fig.add_subplot(sub_d[0]), fig.add_subplot(sub_d[1])
    _track(ax_d1, tq, "pQTL", CHR11_INSTRUMENT, C_FIND, False, 11, y_q)
    g_at = _track(ax_d2, tg, "GWAS", CHR11_INSTRUMENT, C_FIND, True, 11, y_g)
    ax_d2.sharex(ax_d1)
    for a in (ax_d1, ax_d2):
        _genes(a, {"CHRDL2": CHRDL2_SPAN_MB, "POLD3": POLD3_SPAN_MB})
    _verdict(ax_d1, "HEIDI UNRESOLVED   ·   PP.H4 0.99 → 0.66",
             f"cancer signal at the same variant:  $p$ = {10 ** -g_at:.0e}",
             C_FIND)
    ax_d1.set_title("c   BMP4 × CRC — chr11q13.4, in $CHRDL2$", loc="left",
                    fontsize=11.5, fontweight="bold", pad=6, color=C_FIND)

    sub_e = gs[1, 2:4].subgridspec(2, 1, hspace=0.10)
    ax_e1, ax_e2 = fig.add_subplot(sub_e[0]), fig.add_subplot(sub_e[1])
    _track(ax_e1, lq, "pQTL", LEP_INSTRUMENT, C_FIND, False, 16, y_q,
           labels=LABELS_LEP)
    l_at = _track(ax_e2, lg, "GWAS", LEP_INSTRUMENT, C_FIND, True, 16, y_g,
                  labels=LABELS_LEP)
    ax_e2.sharex(ax_e1)
    for a in (ax_e1, ax_e2):
        _genes(a, {"FTO": FTO_SPAN_MB})
    _verdict(ax_e1, "HEIDI UNRESOLVED   ·   PP.H4 0.99 → 0.43",
             f"cancer signal at the same variant:  $p$ = {10 ** -l_at:.0e}",
             C_FIND)
    ax_e1.set_title("d   LEP × BRC — chr16, in $FTO$", loc="left",
                    fontsize=11.5, fontweight="bold", pad=6, color=C_FIND)

    sub_f = gs[1, 4:6].subgridspec(2, 1, hspace=0.10)
    ax_f1, ax_f2 = fig.add_subplot(sub_f[0]), fig.add_subplot(sub_f[1])
    _track(ax_f1, cq, "pQTL", cis_probe, C_CTRL, False, 14, y_q)
    c_at = _track(ax_f2, cg, "GWAS", cis_probe, C_CTRL, True, 14, y_g)
    ax_f2.sharex(ax_f1)
    for a in (ax_f1, ax_f2):
        _genes(a, {"BMP4": BMP4_SPAN_MB})
    # No colocalization posterior on this badge (2026-09-04): the chr14 cis
    # check has NOT been reproduced with the repaired adapters, so its
    # PP.H3/PP.H4 are withdrawn. The badge now states only what the plotted
    # association data themselves show.
    _verdict(ax_f1, "SIGNALS DO NOT COINCIDE",
             f"cancer signal at the same variant:  $p$ = {10 ** -c_at:.2f}",
             C_CTRL)
    ax_f1.set_title("e   control: BMP4's own gene, chr14", loc="left",
                    fontsize=11.5, fontweight="bold", pad=6, color=C_CTRL)

    panel_genes(fig.add_subplot(gs[2, :]))

    # Two SEPARATE probe-marker entries, not one: the diamond used to be one
    # fixed colour in both panels, which read as "the same SNP" even though
    # chr11's is the real instrument and chr14's is an unrelated, weaker
    # candidate on a different chromosome. Colour now carries that
    # difference, so the legend has to say so too.
    handles = [plt.Line2D([0], [0], marker="o", ls="", ms=7, mfc=DIST_COL[k],
                          mec=DIST_COL[k], label=BIN_LABEL[k]) for k in BINS]
    handles.append(plt.Line2D([0], [0], marker="D", ls="", ms=8, mfc=C_FIND,
                              mec="white", label="c, d: the instrument"))
    handles.append(plt.Line2D([0], [0], marker="D", ls="", ms=8, mfc=C_CTRL,
                              mec="white", label="e: candidate SNP checked"))
    fig.legend(handles=handles, loc="upper right", ncol=7, fontsize=9,
               bbox_to_anchor=(0.978, 0.945), framealpha=0.0,
               title="c–e: distance from probed variant (ramp matches slide 16)",
               title_fontsize=9)

    fig.suptitle("Two of 316 screened pairs share a genetic signal with cancer "
                 "risk — both act in trans, neither resolves",
                 fontsize=17, fontweight="bold", x=0.052, ha="left", y=0.972)
    fig.text(0.052, 0.947,
             "Read c–e downward: protein signal above, cancer signal below, "
             "same window. In c and d the two coincide at the instrument; in e, "
             "the control, they do not.",
             fontsize=9.8, fontweight="bold", color=C_MUTE, ha="left",
             va="top", linespacing=1.5)

    for ext in ("png", "pdf"):
        p = OUT / f"fig13_regional_evidence_2026-09-04.{ext}"
        fig.savefig(p, dpi=DPI, bbox_inches="tight")
        print(f"  saved -> {p}")
    plt.close(fig)


if __name__ == "__main__":
    print("Building fig13 locus-story mosaic (slide 13 rebuild) ...")
    main()
    print("Done.")
