#!/usr/bin/env python3
"""
fig6_parta_sankey_bubble.py  (v2 -- single-panel Sankey)
Static Sankey: 12 cancer single-panels -> protein function categories.
Filled cubic-Bezier ribbons in pure matplotlib (no external Sankey library).

Changes from v1:
  - Bubble panel removed; Sankey stands alone.
  - Full cancer names (Lung Cancer, Endometrial Cancer, ...).
  - K= removed from left labels -- node height already encodes panel size.
  - Right labels show unique-protein count (non-redundant with node height).
  - Figsize narrowed to 16x13 (single column).

Output: figurev5/output/fig6_parta_sankey.pdf/.png
"""
from __future__ import annotations
from pathlib import Path
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.path import Path as MPath
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PKG  = ROOT / "revise_plan" / "locked25_final_results_package"
OUT  = ROOT / "figurev6" / "output"
OUT.mkdir(parents=True, exist_ok=True)

# ── cancer metadata ───────────────────────────────────────────────────────────
CANCERS = ["AML","BRC","CLL","CRC","CVX","ENDC","GLIOM","LUNGC","LYMPH","MYEL","OVC","PRC"]
CANCER_FULL = {
    "AML":   "AML",
    "BRC":   "Breast Cancer",
    "CLL":   "CLL",
    "CRC":   "Colorectal Cancer",
    "CVX":   "Cervical Cancer",
    "ENDC":  "Endometrial Cancer",
    "GLIOM": "Glioma",
    "LUNGC": "Lung Cancer",
    "LYMPH":"DLBCL",
    "MYEL":  "Myeloma",
    "OVC":   "Ovarian Cancer",
    "PRC":   "Prostate Cancer",
}
CANCER_COLOR = {
    "AML":"#b22222","BRC":"#c97b63","CLL":"#7a3e9d","CRC":"#d68600","CVX":"#c13d86",
    "ENDC":"#8e5d2c","GLIOM":"#1a8db8","LUNGC":"#2e8b57","LYMPH":"#3856a6",
    "MYEL":"#8c564b","OVC":"#d1495b","PRC":"#008b8b",
}

# ── protein -> function category ──────────────────────────────────────────────
FUNC = {
    "FLT3":"Immune/Hematop.","CD244":"Immune/Hematop.","FCER2":"Immune/Hematop.",
    "TCL1A":"Immune/Hematop.","SLAMF7":"Immune/Hematop.","FCRLB":"Immune/Hematop.",
    "MZB1":"Immune/Hematop.","TNFRSF13C":"Immune/Hematop.","TNFRSF10A":"Immune/Hematop.",
    "C1QA":"Immune/Hematop.","PDCD1":"Immune/Hematop.","CORO1A":"Immune/Hematop.",
    "PGLYRP1":"Immune/Hematop.",
    "CXCL8":"Cytokine","CXCL13":"Cytokine","CXCL17":"Cytokine",
    "BMP4":"Growth factor","BMP6":"Growth factor","ARTN":"Growth factor",
    "NTF3":"Growth factor","IGFBP1":"Growth factor","AREG":"Growth factor",
    "ENPP2":"Growth factor","LEFTY2":"Growth factor","PROK1":"Growth factor",
    "NTRK3":"Growth factor","ADGRG1":"Growth factor","PSPN":"Growth factor",
    "SDC4":"ECM/Adhesion","SDC1":"ECM/Adhesion","SPOCK1":"ECM/Adhesion",
    "LGALS4":"ECM/Adhesion","CDH15":"ECM/Adhesion","CNTN5":"ECM/Adhesion",
    "CNTN3":"ECM/Adhesion","TSPAN1":"ECM/Adhesion","BCAN":"ECM/Adhesion",
    "FAP":"ECM/Adhesion","CRTAC1":"ECM/Adhesion","CDHR2":"ECM/Adhesion","S100A4":"ECM/Adhesion",
    "LTA4H":"Enzyme/Protease","ADAMTS13":"Enzyme/Protease","KLK14":"Enzyme/Protease",
    "KLK13":"Enzyme/Protease","THOP1":"Enzyme/Protease","HAGH":"Enzyme/Protease",
    "PRDX6":"Enzyme/Protease","PRDX5":"Enzyme/Protease","GLO1":"Enzyme/Protease",
    "CA3":"Enzyme/Protease","PSMD9":"Enzyme/Protease","ALPP":"Enzyme/Protease",
    "QPCT":"Enzyme/Protease","AFP":"Enzyme/Protease",
    "WFDC2":"Cancer marker","PAEP":"Cancer marker","CGA":"Cancer marker",
    "MSMB":"Cancer marker","TACSTD2":"Cancer marker","SEZ6L2":"Cancer marker",
    "SPINK4":"Cancer marker","CEACAM5":"Cancer marker",
    "PSIP1":"Signaling","ILKAP":"Signaling","NFATC1":"Signaling",
    "TRAF2":"Signaling","BID":"Signaling","TFRC":"Signaling",
    "GFAP":"Neural/Struct.","NEFL":"Neural/Struct.",
    "GCG":"Metabolic/Endocr.","LEP":"Metabolic/Endocr.","PLIN1":"Metabolic/Endocr.",
    "CHGB":"Metabolic/Endocr.","VAT1":"Metabolic/Endocr.",
    "XG":"Metabolic/Endocr.","FKBP1B":"Metabolic/Endocr.","CRNN":"Metabolic/Endocr.",
    "F3":"Metabolic/Endocr.",
}
FUNC_ORDER = [
    "Immune/Hematop.","Cytokine","Growth factor","ECM/Adhesion",
    "Enzyme/Protease","Cancer marker","Signaling","Metabolic/Endocr.","Neural/Struct.",
]
# Set2 palette -- distinct from CANCER_COLOR so category nodes don't look
# like a cancer identity (e.g. old purple matched CLL exactly).
FUNC_COL = {
    "Immune/Hematop.":  "#238b6b",
    "Cytokine":         "#d85f35",
    "Growth factor":    "#6ca52a",
    "ECM/Adhesion":     "#5e75b4",
    "Enzyme/Protease":  "#c94f9c",
    "Cancer marker":    "#b77900",
    "Signaling":        "#9b7139",
    "Metabolic/Endocr.":"#3e87b5",
    "Neural/Struct.":   "#666666",
}

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 10,
    "axes.linewidth": 0.9,
})
DPI = 400
LABEL_SIZE = 22

# ── load deploy lists + AUC ───────────────────────────────────────────────────
si_files = list(PKG.rglob("single_internal.csv"))
si_df    = pd.read_csv(si_files[0])

def get_deploy(cancer):
    row = si_df[si_df["target_class"].str.upper() == cancer.upper()]
    return ([p.strip() for p in str(row.iloc[0]["deploy_proteins"]).split(";") if p.strip()]
            if not row.empty else [])

def get_auc(cancer):
    row = si_df[si_df["target_class"].str.upper() == cancer.upper()]
    return float(row.iloc[0]["best_panel_test_auc"]) if not row.empty else None

deploy = {c: get_deploy(c) for c in CANCERS}
aucs   = {c: get_auc(c)    for c in CANCERS}

# ── edge table: (cancer, category) -> assignment count ───────────────────────
edges = {}
for c in CANCERS:
    for cat, n in Counter(FUNC.get(p, "Other") for p in deploy[c]).items():
        if cat in FUNC_COL:
            edges[(c, cat)] = n

cats_present = [c for c in FUNC_ORDER if any((can, c) in edges for can in CANCERS)]

# unique proteins per category (for right-side label; non-redundant with height)
unique_per_cat = {}
for cat in cats_present:
    unique_per_cat[cat] = len({p for c in CANCERS for p in deploy[c]
                                if FUNC.get(p, "Other") == cat})

# ── Bezier ribbon ─────────────────────────────────────────────────────────────
def draw_ribbon(ax, x0, y0_bot, y0_top, x1, y1_bot, y1_top, color, alpha=0.52):
    """Filled cubic-Bezier ribbon connecting two vertical segments."""
    xm = (x0 + x1) / 2.0
    verts = [
        (x0, y0_bot),
        (xm, y0_bot), (xm, y1_bot), (x1, y1_bot),
        (x1, y1_top),
        (xm, y1_top), (xm, y0_top), (x0, y0_top),
        (x0, y0_bot),
    ]
    codes = [
        MPath.MOVETO,
        MPath.CURVE4, MPath.CURVE4, MPath.CURVE4,
        MPath.LINETO,
        MPath.CURVE4, MPath.CURVE4, MPath.CURVE4,
        MPath.CLOSEPOLY,
    ]
    ax.add_patch(mpatches.PathPatch(
        MPath(verts, codes),
        facecolor=color, edgecolor="none", alpha=alpha, zorder=2,
    ))

# ── Sankey layout constants ───────────────────────────────────────────────────
SCALE  = 1.0    # 1 data unit per protein
NODE_W = 0.28   # node rectangle width
PAD_L  = 1.0    # gap between left (cancer) nodes
PAD_R  = 1.2    # gap between right (category) nodes
MIN_H  = 2.5    # minimum display height for tiny nodes

cancer_raw = {c: sum(n for (can, cat), n in edges.items() if can == c) for c in CANCERS}
cat_raw    = {cat: sum(n for (can, cat2), n in edges.items() if cat2 == cat)
              for cat in cats_present}

# Left column: stack bottom (PRC) -> top (AML)
y = 0.0
left_pos = {}
for c in reversed(CANCERS):
    h = max(cancer_raw[c], MIN_H) * SCALE
    left_pos[c] = {"bot": y, "top": y + h, "mid": y + h / 2, "raw": cancer_raw[c]}
    y += h + PAD_L
total_left_h = y - PAD_L

# Right column: stack bottom (last FUNC_ORDER) -> top (first)
y = 0.0
right_pos = {}
for cat in reversed(cats_present):
    h = max(cat_raw[cat], MIN_H) * SCALE
    right_pos[cat] = {"bot": y, "top": y + h, "mid": y + h / 2, "raw": cat_raw[cat]}
    y += h + PAD_R
total_right_h = y - PAD_R

total_h = max(total_left_h, total_right_h) + 1.0

# X coordinates
X_LN_L = 0.0
X_LN_R = NODE_W          # ribbon start
X_RN_L = 5.5             # ribbon end
X_RN_R = 5.5 + NODE_W

# ── Figure (single panel) ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12.5, 13))
ax.set_xlim(-1.4, 8.5)
ax.set_ylim(-0.5, total_h)
ax.axis("off")

# ── ribbons ───────────────────────────────────────────────────────────────────
left_fill  = {c:   left_pos[c]["top"]   for c in CANCERS}
right_fill = {cat: right_pos[cat]["top"] for cat in cats_present}

for c in CANCERS:            # top -> bottom on left
    for cat in cats_present: # top -> bottom on right
        n = edges.get((c, cat), 0)
        if n == 0:
            continue
        h = n * SCALE
        y0_top = left_fill[c];   y0_bot = y0_top - h;  left_fill[c]   = y0_bot
        y1_top = right_fill[cat]; y1_bot = y1_top - h; right_fill[cat] = y1_bot
        draw_ribbon(ax, X_LN_R, y0_bot, y0_top, X_RN_L, y1_bot, y1_top,
                    CANCER_COLOR[c], alpha=0.50)

# ── left (cancer) nodes ───────────────────────────────────────────────────────
for c in CANCERS:
    pos = left_pos[c]
    col = CANCER_COLOR[c]
    h   = pos["top"] - pos["bot"]

    ax.add_patch(mpatches.Rectangle(
        (X_LN_L, pos["bot"]), NODE_W, h,
        facecolor=col, edgecolor="white", linewidth=0.9, zorder=4,
    ))

    label = "DLBCL" if c == "LYMPH" else c

    ax.text(X_LN_L - 0.18, pos["mid"], label,
            ha="right", va="center", fontsize=LABEL_SIZE,
            fontweight="bold", color=col, clip_on=False,
            linespacing=1.35)

# ── right (category) nodes ────────────────────────────────────────────────────
for cat in cats_present:
    pos   = right_pos[cat]
    fc    = FUNC_COL[cat]
    h     = pos["top"] - pos["bot"]
    n_uniq = unique_per_cat[cat]

    ax.add_patch(mpatches.Rectangle(
        (X_RN_L, pos["bot"]), NODE_W, h,
        facecolor=fc, edgecolor="white", linewidth=0.9, zorder=4,
    ))
    ax.text(X_RN_R + 0.18, pos["mid"],
            f"{cat}  n={n_uniq}",
            ha="left", va="center", fontsize=LABEL_SIZE,
            fontweight="bold", color=fc, clip_on=False,
            linespacing=1.35)

# ── save ──────────────────────────────────────────────────────────────────────
for ext in ("pdf", "png"):
    p = OUT / f"fig6_parta_sankey.{ext}"
    fig.savefig(p, dpi=DPI, bbox_inches="tight")
    print(f"saved -> {p}")
plt.close(fig)
print("done.")
