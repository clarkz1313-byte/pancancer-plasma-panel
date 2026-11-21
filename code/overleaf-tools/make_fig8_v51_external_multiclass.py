"""Manuscript v5.1 FORK of make_fig8_v50_external_multiclass.py (unchanged). Only change: the LUNGC, CRC and PRC
predictions come from the patient-grouped reruns (scripts 68 and 69; Sol audit F1),
so specimens from one patient are never split across training and test.
Output: overleaf-source-git/figures_300/fig9_v51_multiclass_external_mosaic.png (new file); build folder
revise_plan/v5_seed52_common_split/figure_inputs/fig9_v51_multiclass_external_mosaic_build/.

Original docstring follows.
Manuscript v5.0 Figure 8 (25-protein marker-set refits, 12 external endpoints).

Re-renders the slide-9 screening mosaic with the unchanged figurev6 drawing code
(fig89_stacked_mosaic.build_stacked_mosaic, fig89_common) and the same arguments
as its __main__ block; only the prediction inputs change:
  PARTB_PRED - v5 runs of the 25-protein set (script 42/53):
               revise_plan/v5_seed52_common_split/external/multiclass_25/<code>/result/
Every endpoint reproduces the v4.9 production AUC to three decimals; the lung
endpoint's participant-grouped splits were redrawn (9,900 vs 9,959 prediction
rows), which moves one control across 0.50 (TN 375 vs 374). Rebuilding keeps the
figure consistent with the v5 text and Table S10.
Nothing in figurev6/output is written; the build goes to
revise_plan/v5_seed52_common_split/figure_inputs/fig8_build/.
Output: overleaf-source-git/figures_300/fig8_v50_multiclass_external_mosaic.png (new file).
"""
from pathlib import Path
import sys

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
V5 = ROOT / "revise_plan/v5_seed52_common_split"
BUILD = V5 / "figure_inputs/fig9_v51_multiclass_external_mosaic_build"
REF = ROOT / "overleaf-source-git/figures_300/fig9_slide9_screening_mosaic.png"
LIVE = ROOT / "overleaf-source-git/figures_300/fig9_v51_multiclass_external_mosaic.png"
sys.path.insert(0, str(ROOT / "figurev6/scripts"))


def main() -> None:
    BUILD.mkdir(parents=True, exist_ok=True)
    import fig89_common as FC
    import fig89_stacked_mosaic as M

    pred = {}
    for code in FC.CANCER_ORDER:
        folder = "DLBCL" if code == "LYMPH" else code
        p = V5 / "external/multiclass_25" / folder / "result/locked25_grouped_predictions.csv"
        if folder == "LUNGC":
            p = V5 / "external_lung_patient_grouped/multiclass_25/LUNGC/result/locked25_grouped_predictions.csv"
        elif folder in ("CRC", "PRC"):
            p = V5 / "external_patient_grouped/multiclass_25" / folder / "result/locked25_grouped_predictions.csv"
        assert p.exists(), p
        pred[code] = p
    M.OUT = FC.OUT = BUILD

    stem = "fig9_v51_multiclass_external_mosaic"
    M.build_stacked_mosaic(
        pred, thr=0.50, part="B", add_macro=True,
        gauge_metric="sens", gauge_target=0.90,
        dca_style="area", lr_kind="LR-",
        suptitle="External marker-set evaluation: 25-protein set | "
                 "12 cancer endpoints in external datasets | fixed threshold 0.50",
        stem=stem)
    ref = Image.open(REF).size
    im = Image.open(BUILD / f"{stem}.png")
    im.resize((ref[0], round(im.height * ref[0] / im.width)), Image.LANCZOS).save(LIVE, optimize=True)
    print(LIVE, "(v4.9 size", ref, ")")


if __name__ == "__main__":
    main()
