#!/usr/bin/env python3
"""Export the current figurev6/output mosaics directly at 600 dpi.

This handoff helper intentionally reads from ``figurev6/output`` rather than
the historical ``final_figures_400dpi`` directory, which may contain an older
revision of a figure.  It preserves physical dimensions, writes PNGs to the
authoritative delivery folder, and verifies the embedded DPI metadata.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image

import export_final_figures_600dpi as exporter

Image.MAX_IMAGE_PIXELS = None

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "output"
DESTINATION = ROOT / "final_figures_600dpi"
STEMS = (
    "fig2_slide2_cohort_mosaic",
    "fig3_slide3_derivation_mosaic",
    "fig4_slide4_structure_mosaic",
    "fig5_slide5_multiclass_mosaic",
    "fig6_slide6_pathway_mosaic",
    "fig7_slide7_singleclass_mosaic",
    "fig8_slide8_confirmation_mosaic",
    "fig9_slide9_screening_mosaic",
    "fig10_5_ppi_combined",
    "fig12_pathway_permutation_mosaic",
    "fig13_regional_evidence_2026-09-05",
)


def main() -> None:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    for stem in STEMS:
        source = SOURCE / f"{stem}.png"
        destination = DESTINATION / source.name
        if not source.exists():
            raise FileNotFoundError(source)
        size = exporter.export(source, destination, 600.0, 400.0)
        exporter.verify(destination, size[:2], 600.0)
        print(f"{destination.relative_to(ROOT)}: {size[0]} x {size[1]} px @ 600 dpi")


if __name__ == "__main__":
    main()
