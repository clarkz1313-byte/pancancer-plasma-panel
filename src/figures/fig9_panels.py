#!/usr/bin/env python3
"""
Fig 9 — SLIDE 9: external validation, MULTI-PANEL (Part B screening).

Seven rows, in slide order. Part B is ONE locked 25-protein multiclass
model evaluated on 12 external cohorts, so a 13th purple macro tile IS
statistically defensible and every row carries one (13 tiles wide). Row 3
is the deliberate exception: its aggregate tile is POOLED RAW COUNTS
("Pooled total"), not a macro mean of rates -- summing confusion counts is
meaningful, averaging rates across wildly different cohort sizes is not.

  9a  probability box plots       (+ macro)
  9b  ROC curves                  (+ macro, interpolated on shared FPR grid)
  9c  confusion matrices          (+ pooled total; fixed threshold 0.50)
  9d  sensitivity, ICON ARRAY     (0.90 target; the 100-square geometry is
                                   deliberately different from slide 8's
                                   ring gauge so the screening and
                                   confirmation rows can't be confused)
  9e  decision curve analysis     (area style = screening yield)
  9f  LR- (rule-out strength)     (dot, log scale, geometric-mean macro)
  9g  external calibration        (reliability + per-tile ECE/Brier)

  9h  sensitivity, RING gauge — NOT ON THE SLIDE. Kept as a drop-in
      alternative to 9d's icon array if the ring reads better at final
      layout size. Pick one; do not put both on slide 9.

All shared rendering lives in fig89_common.py -- slide 8 imports the same
builders. Do not fork a builder into this file.

Threshold note: Part B uses a FIXED 0.50 threshold (slide 8 / Part A uses a
fixed 0.80). Both are prespecified; neither is fitted to the cohort being
evaluated. Screening is judged on catching cases, so Part B's cutoff is the
looser of the two -- but it is still fixed, not optimised per cohort.
Changed 2026-08-16 from a per-cohort Youden threshold, which selected the
cutoff using the test cohort's own labels and so biased every Part B number
optimistically. See SCREEN_THR below and fig_v5_changelog.md.

Output: figurev5/output/fig9{a..h}_partb_*.pdf/.png
Run:    python fig9_panels.py
"""
from __future__ import annotations

from fig89_common import (
    PARTB_PRED,
    build_boxplots,
    build_roc,
    build_cm,
    build_metric_mosaic,
    build_ring_mosaic,
    build_dca_mosaic,
    build_lr_mosaic,
    build_calibration_mosaic,
)

SENS_TARGET = 0.90   # dashed target marker on the sensitivity row

# Part B is the SCREENING stage, so it is scored at a fixed, pre-specified
# probability cutoff -- not at a per-cohort Youden threshold.
#
# Why this changed (2026-08-16): Youden maximises sens+spec-1 using the test
# cohort's own labels, which is a mild test-set peek and biases every Part B
# number optimistically. A fixed cutoff never looks at the labels. 0.50 (not
# Part A's 0.80) because this row must *catch* cases and hand them to Part A
# for confirmation: macro sensitivity 0.927 at 0.50 vs 0.866 at 0.80, and at
# 0.80 the weakest class (CVX) catches only 15 of 28 cases. Part A remains at
# 0.80 -- screen wide here, confirm strict there.
SCREEN_THR = 0.50


if __name__ == "__main__":
    print("Building fig9 panels (slide 9 — multi panel / Part B) ...")

    build_boxplots(PARTB_PRED,
        "25-protein marker set | predicted cancer probability",
        "fig9a_partb_probability_boxplots", add_macro=True)

    build_roc(PARTB_PRED,
        "25-protein marker set | ROC curves",
        "fig9b_partb_roc_curves", add_macro=True)

    build_cm(PARTB_PRED, SCREEN_THR,
        "25-protein marker set | confusion matrices, fixed threshold 0.50",
        "fig9c_partb_confusion_matrices", add_macro=True)

    build_metric_mosaic(PARTB_PRED, SCREEN_THR, "sens", SENS_TARGET,
        "25-protein marker set | sensitivity, 95% CI\n"
        "Fixed threshold 0.50 | 0.90 reference",
        "fig9d_partb_sensitivity_ci", add_macro=True)

    build_dca_mosaic(PARTB_PRED, None, "area",
        "25-protein marker set | decision curve analysis\n"
        "Shaded area = advantage over treat-all",
        "fig9e_partb_dca", add_macro=True)

    build_lr_mosaic(PARTB_PRED, SCREEN_THR, "LR-",
        "25-protein marker set | LR- (rule-out strength)\n"
        "Log scale | lower values indicate stronger negative evidence | threshold 0.50",
        "fig9f_partb_lr_minus", add_macro=True)

    build_calibration_mosaic(PARTB_PRED,
        "25-protein marker set | external calibration\n"
        "Predicted score vs. observed rate, external cohorts",
        "fig9g_partb_calibration", add_macro=True)

    # Not on the slide -- alternative geometry for row 4, kept so the ring
    # vs. icon-array choice can be made at final layout size.
    build_ring_mosaic(PARTB_PRED, SCREEN_THR, "sens", SENS_TARGET,
        "25-protein marker set | sensitivity, 95% CI (ring)\n"
        "Fixed threshold 0.50 | 0.90 reference",
        "fig9h_partb_sensitivity_ring")

    print("\nDone.")
