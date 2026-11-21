"""Manuscript v5.0 Figure 3 (differential evidence and panel derivation).

Runs the unchanged figurev6 builder (fig3_slide3_mosaic.build_slide3_mosaic,
fig3_panels) with its disease-specific inputs pointed at v5:
  column A (volcano)  - v5 development-set one-versus-rest screen. In v5 this
                        screen is identical to the multiclass train-only
                        screen (same 962 patients; checked 2026-09-29).
  column C (LASSO)    - v5 rankings from revise_plan/v5_seed52_common_split/
                        part_a_single, deployed sizes from the v5 selection.
  columns B, D, E     - multiclass inputs, unchanged.
Derived inputs are written to revise_plan/v5_seed52_common_split/figure_inputs/.
Output: overleaf-source-git/figures_300/fig3_v50_derivation_mosaic.png (new
file; the v4.9 fig3_slide3_derivation_mosaic.png is untouched).
"""
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
V5 = ROOT / "revise_plan/v5_seed52_common_split"
INP = V5 / "figure_inputs"
LIVE = ROOT / "overleaf-source-git/figures_300/fig3_v50_derivation_mosaic.png"
LIVE_WIDTH = 4955          # width of the v4.9 figures_300 copy of Figure 3 (4955 x 5172)
CODES = {"aml": "AML", "brc": "BRC", "cll": "CLL", "crc": "CRC", "cvx": "CVX", "endc": "ENDC",
         "gliom": "GLIOM", "lungc": "LUNGC", "lymph": "LYMPH", "myel": "MYEL", "ovc": "OVC", "prc": "PRC"}

sys.path.insert(0, str(ROOT / "figurev6/scripts"))


def build_inputs() -> tuple[Path, Path]:
    INP.mkdir(parents=True, exist_ok=True)
    frames = []
    for slug, code in CODES.items():
        d = pd.read_csv(V5 / "part_a_single" / slug / "tables" / f"{slug}_dea.csv")
        n = len(d)
        frames.append(pd.DataFrame({
            "target_class": code, "protein": d.protein, "fold_change": d.fold_change,
            "p_value": d.p_value, "significant_bonf": d.significant_bonf,
            "minus_log10_raw_p": -np.log10(d.p_value), "bonferroni_raw_p_threshold": 0.05 / n,
            "label_in_figure": False}))
    volcano = INP / "bonferroni_volcano_source_v5.csv"
    pd.concat(frames, ignore_index=True).to_csv(volcano, index=False)

    sel = pd.read_csv(V5 / "panels/panel_size_selection_v5.csv")
    sel["target_class"] = sel.cancer.replace({"DLBCL": "LYMPH"})
    s1 = pd.DataFrame({"target_class": sel.target_class, "deploy_panel_size": sel.n_final,
                       "deploy_proteins": sel.proteins})
    summary = INP / "single_panel_protein_summary_v5.csv"
    s1.to_csv(summary, index=False)
    return volcano, summary


def main() -> None:
    volcano, summary = build_inputs()
    import fig3_panels as F
    import fig3_slide3_mosaic as M
    out = INP / "fig3_build"
    out.mkdir(parents=True, exist_ok=True)
    for mod in (F, M):
        mod.VOLCANO_SRC, mod.SUPP_S1, mod.OUT = volcano, summary, out
    F.PART_A_SINGLE = V5 / "part_a_single"
    M.build_slide3_mosaic(stem="fig3_v50_derivation_mosaic")
    master = out / "fig3_v50_derivation_mosaic.png"
    im = Image.open(master)
    im.resize((LIVE_WIDTH, round(im.height * LIVE_WIDTH / im.width)), Image.LANCZOS).save(LIVE, optimize=True)
    print(LIVE)


if __name__ == "__main__":
    main()
