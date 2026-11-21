#!/usr/bin/env python3
"""
Fig 8 — SLIDE 8: external validation, SINGLE-PANEL (Part A confirmation).

Seven rows, in slide order. Part A is 12 INDEPENDENT binary confirmation
panels -- one model per cancer, not one model evaluated 12 times -- so
there are NO macro/aggregate tiles anywhere on this slide. Every row is
12 tiles wide.

  8a  probability box plots
  8b  ROC curves
  8c  confusion matrices          (fixed 0.80 confirmation threshold)
  8d  specificity, RING gauge     (0.90 target; ring geometry is
                                   deliberately different from slide 9's
                                   icon array so the confirmation and
                                   screening rows can't be confused)
  8e  decision curve analysis     (line style = precision of the call)
  8f  LR+ (rule-in strength)      (bar, log scale)
  8g  external calibration        (reliability + per-tile ECE/Brier)

All shared rendering lives in fig89_common.py -- slide 9 imports the same
builders. Do not fork a builder into this file.

Threshold note: Part A uses a fixed 0.80 confirmation threshold on external
data. This is NOT a clinical operating point -- these external cohorts are
85-90% case-enriched research cohorts. See the painful-trio sections of
figurev5/fig_v5_changelog.md before reinterpreting these numbers.

Output: figurev5/output/fig8{a..g}_parta_*.pdf/.png
Run:    python fig8_panels.py
"""
from __future__ import annotations

from fig89_common import (
    PARTA_PRED,
    build_boxplots,
    build_roc,
    build_cm,
    build_ring_mosaic,
    build_dca_mosaic,
    build_lr_mosaic,
    build_calibration_mosaic,
)

CONFIRM_THR = 0.80   # fixed confirmation threshold, all 12 Part A panels
SPEC_TARGET = 0.90   # dashed target marker on the ring gauge


if __name__ == "__main__":
    print("Building fig8 panels (slide 8 — single panel / Part A) ...")

    build_boxplots(PARTA_PRED,
        "Cancer-specific panels | predicted cancer probability",
        "fig8a_parta_probability_boxplots")

    build_roc(PARTA_PRED,
        "Cancer-specific panels | ROC curves",
        "fig8b_parta_roc_curves")

    build_cm(PARTA_PRED, CONFIRM_THR,
        "Cancer-specific panels | confusion matrices, threshold 0.80",
        "fig8c_parta_confusion_matrices")

    build_ring_mosaic(PARTA_PRED, CONFIRM_THR, "spec", SPEC_TARGET,
        "Cancer-specific panels | specificity, 95% CI\n"
        "Fixed threshold 0.80 | 0.90 reference",
        "fig8d_parta_specificity_ring")

    build_dca_mosaic(PARTA_PRED, CONFIRM_THR, "line",
        "Cancer-specific panels | decision curve analysis\n"
        "Net benefit across threshold probabilities",
        "fig8e_parta_dca")

    build_lr_mosaic(PARTA_PRED, CONFIRM_THR, "LR+",
        "Cancer-specific panels | LR+ (rule-in strength)\n"
        "Log scale | higher values indicate stronger positive evidence",
        "fig8f_parta_lr_plus")

    build_calibration_mosaic(PARTA_PRED,
        "Cancer-specific panels | external calibration\n"
        "Predicted score vs. observed rate, external cohorts",
        "fig8g_parta_calibration")

    print("\nDone.")
