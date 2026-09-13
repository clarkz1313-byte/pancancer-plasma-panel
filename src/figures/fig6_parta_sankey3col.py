#!/usr/bin/env python3
"""
fig6_parta_sankey3col.py  (v2)
Three-column static Sankey: cancer -> function category -> protein
Pure matplotlib filled Bezier ribbons.

v2 fixes:
  - FUNC_COL uses Set2 palette, completely distinct from CANCER_COLOR.
    (Previously shared identical hex codes, causing purple CLL = purple
     Immune/Hematop. confusion.)
  - Middle category nodes widened to 1.2 units; horizontal 2-line labels
    inside nodes replace unreadable rotated text.
  - Protein labels diagonal (35 deg) to allow larger font.
  - Cancer labels larger (fontsize 11).
  - Protein column has visible gap between category groups.

Output: figurev5/output/fig6_parta_sankey3col.pdf/.png
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

ROOT = Path("e:/Proteomics")
PKG  = ROOT / "revise_plan" / "locked25_final_results_package"
OUT  = ROOT / "figurev5" / "output"
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
# Vivid/saturated palette -- identity colors for each cancer
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

# Set2 palette -- completely separate from CANCER_COLOR so ribbons change
# color meaningfully as they cross from cancer identity to functional role.
FUNC_COL = {
    "Immune/Hematop.":  "#66c2a5",   # mint green
    "Cytokine":         "#fc8d62",   # orange-salmon
    "Growth factor":    "#a6d854",   # lime green
    "ECM/Adhesion":     "#8da0cb",   # periwinkle blue
    "Enzyme/Protease":  "#e78ac3",   # orchid pink
    "Cancer marker":    "#e6ab02",   # golden yellow (darker than ffd92f for contrast)
    "Signaling":        "#e5c494",   # warm tan
    "Metabolic/Endocr.":"#80b1d3",   # sky blue
    "Neural/Struct.":   "#b3b3b3",   # neutral gray
}

# 2-line labels for inside the wide middle nodes
CAT_LABEL = {
    "Immune/Hematop.":   "Immune /\nHematopoietic",
    "Cytokine":          "Cytokine",
    "Growth factor":     "Growth\nFactor",
    "ECM/Adhesion":      "ECM /\nAdhesion",
    "Enzyme/Protease":   "Enzyme /\nProtease",
    "Cancer marker":     "Cancer\nMarker",
    "Signaling":         "Signaling",
    "Metabolic/Endocr.": "Metabolic /\nEndocrine",
    "Neural/Struct.":    "Neural /\nStructural",
}

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 10,
    "axes.linewidth": 0.8,
})
DPI = 300

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

# ── edge tables ───────────────────────────────────────────────────────────────
edges_lm = {}
for c in CANCERS:
    for cat, n in Counter(FUNC.get(p, "Other") for p in deploy[c]).items():
        if cat in FUNC_COL:
            edges_lm[(c, cat)] = n

cats_present = [c for c in FUNC_ORDER if any((can, c) in edges_lm for can in CANCERS)]

protein_to_cat = {p: cat for p, cat in FUNC.items() if cat in cats_present}
protein_counts = {p: sum(1 for c in CANCERS if p in deploy[c])
                  for p in protein_to_cat}
protein_counts = {p: n for p, n in protein_counts.items() if n > 0}

def prot_key(p):
    cat = protein_to_cat[p]
    return (FUNC_ORDER.index(cat), -protein_counts[p], p)

proteins_ordered = sorted(protein_counts.keys(), key=prot_key)
edges_mp = {(protein_to_cat[p], p): protein_counts[p] for p in proteins_ordered}

print(f"Proteins: {len(proteins_ordered)}, Total L: {sum(edges_lm.values())}, "
      f"Total R: {sum(edges_mp.values())}")

# ── Bezier ribbon ─────────────────────────────────────────────────────────────
def draw_ribbon(ax, x0, y0_bot, y0_top, x1, y1_bot, y1_top, color, alpha=0.45):
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

# ── layout constants ──────────────────────────────────────────────────────────
SCALE   = 1.0
NODE_WC = 0.35   # cancer node width
NODE_WM = 0.50   # category node width (labels go outside to the right)
NODE_WP = 0.30   # protein node width
PAD_C   = 1.0
PAD_M   = 1.2
PAD_P   = 0.18
MIN_HC  = 2.5
MIN_HM  = 2.5
MIN_HP  = 0.80
# Extra gap inserted between protein groups (between categories)
GAP_GROUP = 1.2

# ── Column 1: cancer ──────────────────────────────────────────────────────────
cancer_raw = {c: sum(n for (can, cat), n in edges_lm.items() if can == c) for c in CANCERS}
y = 0.0
left_pos = {}
for c in reversed(CANCERS):
    h = max(cancer_raw[c], MIN_HC) * SCALE
    left_pos[c] = {"bot": y, "top": y + h, "mid": y + h / 2, "raw": cancer_raw[c]}
    y += h + PAD_C
total_left_h = y - PAD_C

# ── Column 2: category ────────────────────────────────────────────────────────
cat_raw = {cat: sum(n for (can, cat2), n in edges_lm.items() if cat2 == cat)
           for cat in cats_present}
y = 0.0
mid_pos = {}
for cat in reversed(cats_present):
    h = max(cat_raw[cat], MIN_HM) * SCALE
    mid_pos[cat] = {"bot": y, "top": y + h, "mid": y + h / 2, "raw": cat_raw[cat]}
    y += h + PAD_M
total_mid_h = y - PAD_M

# ── Column 3: protein (ordered by category, with inter-group gaps) ────────────
y = 0.0
right_pos = {}
prev_cat  = None
for p in reversed(proteins_ordered):
    cat = protein_to_cat[p]
    # Insert a larger gap at each category boundary
    if prev_cat is not None and cat != prev_cat:
        y += GAP_GROUP - PAD_P   # extra gap already includes base PAD_P
    h = max(protein_counts[p], MIN_HP) * SCALE
    right_pos[p] = {"bot": y, "top": y + h, "mid": y + h / 2, "raw": protein_counts[p]}
    y += h + PAD_P
    prev_cat = cat
total_right_h = y - PAD_P

total_h = max(total_left_h, total_mid_h, total_right_h) + 3.0

print(f"Column heights -- cancer: {total_left_h:.1f}  category: {total_mid_h:.1f}  "
      f"protein: {total_right_h:.1f}")

# ── X coordinates ─────────────────────────────────────────────────────────────
X_CL = 0.0;             X_CR = NODE_WC
X_ML = 5.0;             X_MR = X_ML + NODE_WM   # 5.0 -> 5.5
X_PL = X_MR + 5.2;     X_PR = X_PL + NODE_WP   # 10.7 -> 11.0

# ── Figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(24, 28))
ax.set_xlim(-5.0, X_PR + 5.5)
ax.set_ylim(-1.5, total_h)
ax.axis("off")

# ── Left ribbons: cancer -> category (cancer-colored) ────────────────────────
lf_C = {c:   left_pos[c]["top"]  for c in CANCERS}
lf_M = {cat: mid_pos[cat]["top"] for cat in cats_present}

for c in CANCERS:
    for cat in cats_present:
        n = edges_lm.get((c, cat), 0)
        if n == 0:
            continue
        h = n * SCALE
        y0t = lf_C[c];   y0b = y0t - h;  lf_C[c]   = y0b
        y1t = lf_M[cat]; y1b = y1t - h;  lf_M[cat] = y1b
        draw_ribbon(ax, X_CR, y0b, y0t, X_ML, y1b, y1t, CANCER_COLOR[c], alpha=0.45)

# ── Right ribbons: category -> protein (category-colored) ────────────────────
rf_M = {cat: mid_pos[cat]["top"]  for cat in cats_present}
rf_P = {p:   right_pos[p]["top"]  for p in proteins_ordered}

for cat in cats_present:
    for p in proteins_ordered:
        if protein_to_cat.get(p) != cat:
            continue
        n = protein_counts[p]
        h = n * SCALE
        y0t = rf_M[cat]; y0b = y0t - h; rf_M[cat] = y0b
        y1t = rf_P[p];   y1b = y1t - h; rf_P[p]   = y1b
        draw_ribbon(ax, X_MR, y0b, y0t, X_PL, y1b, y1t, FUNC_COL[cat], alpha=0.45)

# ── Cancer nodes (Column 1) ───────────────────────────────────────────────────
for c in CANCERS:
    pos = left_pos[c]
    col = CANCER_COLOR[c]
    h   = pos["top"] - pos["bot"]
    ax.add_patch(mpatches.Rectangle(
        (X_CL, pos["bot"]), NODE_WC, h,
        facecolor=col, edgecolor="white", linewidth=0.8, zorder=4,
    ))
    auc = aucs.get(c)
    label = CANCER_FULL[c]
    if auc:
        label += f"\nAUC {auc:.2f}"
    ax.text(X_CL - 0.20, pos["mid"], label,
            ha="right", va="center", fontsize=11,
            fontweight="bold", color=col, clip_on=False, linespacing=1.35)

# ── Category nodes (Column 2) — labels to the RIGHT, outside node ─────────────
for cat in cats_present:
    pos = mid_pos[cat]
    fc  = FUNC_COL[cat]
    h   = pos["top"] - pos["bot"]
    ax.add_patch(mpatches.Rectangle(
        (X_ML, pos["bot"]), NODE_WM, h,
        facecolor=fc, edgecolor="white", linewidth=0.9, zorder=4,
    ))
    # Label outside the node to the right; white semi-transparent box so
    # text stays readable above ribbon traffic.
    ax.text(X_MR + 0.25, pos["mid"],
            CAT_LABEL[cat],
            ha="left", va="center", fontsize=9,
            fontweight="bold", color=fc,
            linespacing=1.2, clip_on=False, zorder=6,
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.78))

# ── Protein nodes (Column 3) — diagonal labels, category group dividers ───────
prev_cat_p = None
for p in proteins_ordered:
    pos = right_pos[p]
    cat = protein_to_cat[p]
    fc  = FUNC_COL[cat]
    h   = pos["top"] - pos["bot"]

    # Faint horizontal rule at each category boundary (drawn before node)
    if prev_cat_p is not None and cat != prev_cat_p:
        rule_y = pos["top"] + GAP_GROUP / 2
        ax.axhline(rule_y, xmin=0, xmax=1, color="#dddddd", lw=0.6,
                   zorder=1, clip_on=True)

    ax.add_patch(mpatches.Rectangle(
        (X_PL, pos["bot"]), NODE_WP, h,
        facecolor=fc, edgecolor="white", linewidth=0.5, zorder=4,
    ))
    # Diagonal label (35 deg) anchored at left-center of node
    ax.text(X_PR + 0.18, pos["mid"], p,
            ha="left", va="center",
            rotation=35, rotation_mode="anchor",
            fontsize=8.5, fontstyle="italic", fontweight="bold",
            color=fc, clip_on=False)

    prev_cat_p = cat

# ── Column headers ────────────────────────────────────────────────────────────
header_y = total_h - 0.5
for x_mid, label in [
    ((X_CL + X_CR) / 2,    "Cancer panel"),
    ((X_ML + X_MR) / 2,    "Function category"),
    ((X_PL + X_PR) / 2,    "Protein"),
]:
    ax.text(x_mid, header_y, label,
            ha="center", va="bottom", fontsize=12,
            fontweight="bold", color="#222222")

ax.set_title(
    "Single-panel proteins: cancer  ->  function category  ->  protein  (12 cancers)",
    fontsize=13, fontweight="bold", pad=12,
)

# ── save ──────────────────────────────────────────────────────────────────────
for ext in ("pdf", "png"):
    out_p = OUT / f"fig6_parta_sankey3col.{ext}"
    fig.savefig(out_p, dpi=DPI, bbox_inches="tight")
    print(f"saved -> {out_p}")
plt.close(fig)
print("done.")
