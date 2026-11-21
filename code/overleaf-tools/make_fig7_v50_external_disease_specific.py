"""Manuscript v5.0 Figure 7 (disease-specific marker-set refits, 12 external endpoints).

Re-renders the slide-8 confirmation mosaic with the unchanged figurev6 drawing
code (fig89_stacked_mosaic.build_stacked_mosaic, fig89_common) and the same
arguments as its __main__ block; only the prediction inputs change:
  PARTA_PRED - v5 disease-specific runs (script 42/53):
               revise_plan/v5_seed52_common_split/external/v5_disease_specific/<code>/result/
               (was: external_validation_roadmap/*/results_single_panel/, seed-42 panels)
Figure 8 (25-protein set) is not rebuilt: its v5 predictions equal the v4.9
production runs (script 53 gate).
Nothing in figurev6/output is written; the build goes to
revise_plan/v5_seed52_common_split/figure_inputs/fig7_build/.
Output: overleaf-source-git/figures_300/fig7_v50_disease_specific_external_mosaic.png (new file).
"""
from pathlib import Path
import sys

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
V5 = ROOT / "revise_plan/v5_seed52_common_split"
BUILD = V5 / "figure_inputs/fig7_build"
REF = ROOT / "overleaf-source-git/figures_300/fig8_slide8_confirmation_mosaic.png"
LIVE = ROOT / "overleaf-source-git/figures_300/fig7_v50_disease_specific_external_mosaic.png"
sys.path.insert(0, str(ROOT / "figurev6/scripts"))


def main() -> None:
    BUILD.mkdir(parents=True, exist_ok=True)
    import fig89_common as FC
    import fig89_stacked_mosaic as M

    pred = {}
    for code in FC.CANCER_ORDER:
        folder = "DLBCL" if code == "LYMPH" else code
        p = V5 / "external/v5_disease_specific" / folder / "result/locked25_grouped_predictions.csv"
        assert p.exists(), p
        pred[code] = p
    M.OUT = FC.OUT = BUILD

    stem = "fig7_v50_disease_specific_external_mosaic"
    M.build_stacked_mosaic(
        pred, thr=0.80, part="A", add_macro=False,
        gauge_metric="spec", gauge_target=0.90,
        dca_style="area", lr_kind="LR+",
        suptitle="External marker-set evaluation: cancer-specific panels | "
                 "12 cancer endpoints in external datasets | fixed threshold 0.80",
        stem=stem)
    ref = Image.open(REF).size
    im = Image.open(BUILD / f"{stem}.png")
    im.resize((ref[0], round(im.height * ref[0] / im.width)), Image.LANCZOS).save(LIVE, optimize=True)
    print(LIVE, "(v4.9 size", ref, ")")


if __name__ == "__main__":
    main()
