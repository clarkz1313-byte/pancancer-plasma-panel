"""v7 Figure 8 (disease-specific lists) and Supplementary Figure (26-protein list): single-cancer tissue tests.

Same unchanged figurev6 drawing code and arguments as the v5.1 wrappers (overleaf-tools/
make_fig7_v51_external_disease_specific.py and make_fig8_v51_external_multiclass.py); only the prediction inputs
change to the v7 B6 runs (results/B6_external/<variant>/<cancer>/; CRC and PRC from result_patient_grouped, as in
the B6 metrics). Nothing in figurev6/output is written.
Outputs: figures/output/fig8_v7_external_disease_specific.png, figS_v7_external_multiclass26.png
"""
import shutil

import v7_style as S

B6 = S.RES / "B6_external"
BUILD = S.V7 / "figures/build/fig8"


def preds(variant):
    out = {}
    for code in S.FC.CANCER_ORDER:
        folder = "DLBCL" if code == "LYMPH" else code
        p = B6 / variant / folder / "result_patient_grouped/locked25_grouped_predictions.csv"
        if not p.exists():
            p = B6 / variant / folder / "result/locked25_grouped_predictions.csv"
        assert p.exists(), p
        out[code] = p
    return out


def main():
    BUILD.mkdir(parents=True, exist_ok=True)
    import fig89_stacked_mosaic as M
    M.OUT = S.FC.OUT = BUILD
    M.build_stacked_mosaic(preds("disease_specific"), thr=0.80, part="A", add_macro=False, gauge_metric="spec",
                           gauge_target=0.90, dca_style="area", lr_kind="LR+", suptitle="", stem="fig8_v7_external_disease_specific")
    M.build_stacked_mosaic(preds("multiclass_26"), thr=0.50, part="B", add_macro=True, gauge_metric="sens",
                           gauge_target=0.90, dca_style="area", lr_kind="LR-", suptitle="", stem="figS_v7_external_multiclass26")
    for stem in ("fig8_v7_external_disease_specific", "figS_v7_external_multiclass26"):
        shutil.copyfile(BUILD / f"{stem}.png", S.OUT / f"{stem}.png")
        print("wrote", S.OUT / f"{stem}.png")


if __name__ == "__main__":
    main()
