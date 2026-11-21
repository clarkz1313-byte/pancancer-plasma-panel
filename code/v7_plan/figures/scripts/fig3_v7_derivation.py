"""v7 Figure 3: differential evidence and development-only panel derivation.

Top block (A-E): the unchanged figurev6 builder (fig3_slide3_mosaic / fig3_panels), as in the v5.0 wrapper
overleaf-tools/make_fig3_v50_derivation.py, with v7 inputs: the multiclass list is the 26-protein panel
(LOCKED_25 constant replaced by the v7 list) and the disease-specific deployed panels are the v7
development-only panels. Volcano and LASSO-rank inputs are the development-set screens (unchanged).
Bottom block (F-H): fig3de_v7_dev_selection.py (development CV size choice), letters shifted to F-H.
Both blocks are stacked at a common width. Nothing in figurev6/output is written.
Output: figures/output/fig3_v7_derivation.png
"""
import importlib
import sys

import numpy as np
import pandas as pd
from PIL import Image

import v7_style as S

V5 = S.ROOT / "revise_plan/v5_seed52_common_split"
BUILD = S.V7 / "figures/build/fig3"
BUILD.mkdir(parents=True, exist_ok=True)
CODES = {"aml": "AML", "brc": "BRC", "cll": "CLL", "crc": "CRC", "cvx": "CVX", "endc": "ENDC",
         "gliom": "GLIOM", "lungc": "LUNGC", "lymph": "LYMPH", "myel": "MYEL", "ovc": "OVC", "prc": "PRC"}


def inputs():
    frames = []
    for slug, code in CODES.items():
        d = pd.read_csv(V5 / "part_a_single" / slug / "tables" / f"{slug}_dea.csv")
        frames.append(pd.DataFrame({"target_class": code, "protein": d.protein, "fold_change": d.fold_change,
                                    "p_value": d.p_value, "significant_bonf": d.significant_bonf,
                                    "minus_log10_raw_p": -np.log10(d.p_value), "bonferroni_raw_p_threshold": 0.05 / len(d),
                                    "label_in_figure": False}))
    volcano = BUILD / "bonferroni_volcano_source_v7.csv"
    pd.concat(frames, ignore_index=True).to_csv(volcano, index=False)
    pan = pd.read_csv(S.V7 / "panels/panel_memberships_v7.csv")
    rows = []
    for code in CODES.values():
        prots = list(pan[pan.panel == f"single_{'DLBCL' if code == 'LYMPH' else code}"].protein)
        rows.append({"target_class": code, "deploy_panel_size": len(prots), "deploy_proteins": ";".join(prots)})
    summary = BUILD / "single_panel_protein_summary_v7.csv"
    pd.DataFrame(rows).to_csv(summary, index=False)
    mc = list(pan[pan.panel == "multiclass_26"].protein)
    assert len(mc) == 26
    return volcano, summary, mc


def main():
    volcano, summary, mc = inputs()
    import fig3_panels as F
    import fig3_slide3_mosaic as M
    for mod in (F, M):
        mod.VOLCANO_SRC, mod.SUPP_S1, mod.OUT = volcano, summary, BUILD
        if hasattr(mod, "LOCKED_25"):
            mod.LOCKED_25 = mc
    F.PART_A_SINGLE = V5 / "part_a_single"
    M.build_slide3_mosaic(stem="fig3_v7_top")
    # bottom block with letters F-H
    src = (S.V7 / "figures/scripts/fig3de_v7_dev_selection.py").read_text(encoding="utf-8")
    src = src.replace('((axD, "D"), (axE, "E"), (tiles[0], "F"))', '((axD, "F"), (axE, "G"), (tiles[0], "H"))')
    src = src.replace('S.save(fig, "fig3de_v7_dev_selection.png")', 'S.save(fig, "../build/fig3/fig3_v7_bottom.png")')
    exec(compile(src, "fig3de_v7_dev_selection_FGH", "exec"), {"__name__": "fig3_bottom", "__file__": "x"})
    top = Image.open(BUILD / "fig3_v7_top.png").convert("RGB")
    bot = Image.open(BUILD / "fig3_v7_bottom.png").convert("RGB")
    w = max(top.width, bot.width)
    top = top.resize((w, round(top.height * w / top.width)), Image.LANCZOS)
    bot = bot.resize((w, round(bot.height * w / bot.width)), Image.LANCZOS)
    gap = int(0.02 * w)
    out = Image.new("RGB", (w, top.height + gap + bot.height), "white")
    out.paste(top, (0, 0)); out.paste(bot, (0, top.height + gap))
    p = S.OUT / "fig3_v7_derivation.png"
    out.save(p, optimize=True)
    print("wrote", p, out.size)


if __name__ == "__main__":
    sys.exit(main())
