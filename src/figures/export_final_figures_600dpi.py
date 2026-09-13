#!/usr/bin/env python3
"""Export the approved v6 figures
at 400 and 600 dpi while preserving their physical dimensions.

The v6 figures are already rendered at 400 dpi and are scaled once to each
target resolution. Readability-oriented source edits are performed by each
figure's generating script before this lossless export step.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image

Image.MAX_IMAGE_PIXELS = None

ROOT = Path(__file__).resolve().parents[1]
V6_DIR = ROOT / "final_figures_400dpi"
V5_DIR = ROOT.parent / "figurev5" / "output"
OUT_400 = ROOT / "final_figures_400dpi"
OUT_600 = ROOT / "final_figures_600dpi"

V6_STEMS = (
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
V5_IMPORTS = {}


def embedded_dpi(image: Image.Image, fallback: float) -> float:
    dpi = image.info.get("dpi")
    if not dpi:
        return fallback
    x, y = map(float, dpi)
    if abs(x - y) > 1.0:
        raise ValueError(f"non-square source DPI: {dpi}")
    return (x + y) / 2.0


def export(source: Path, destination: Path, target_dpi: float,
           fallback_dpi: float) -> tuple[int, int, float]:
    with Image.open(source) as image:
        image.load()
        src_dpi = embedded_dpi(image, fallback_dpi)
        scale = target_dpi / src_dpi
        size = tuple(max(1, round(n * scale)) for n in image.size)
        if size != image.size:
            image = image.resize(size, Image.Resampling.LANCZOS)
        image.save(destination, format="PNG", dpi=(target_dpi, target_dpi),
                   compress_level=6)
    return size[0], size[1], src_dpi


def verify(path: Path, size: tuple[int, int], target_dpi: float) -> None:
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        dpi = image.info.get("dpi")
        if image.size != size or not dpi or any(abs(float(d) - target_dpi) > 1.0 for d in dpi):
            raise RuntimeError(f"verification failed for {path.name}: {image.size}, {dpi}")


def main() -> None:
    OUT_400.mkdir(parents=True, exist_ok=True)
    OUT_600.mkdir(parents=True, exist_ok=True)
    sources = [(V6_DIR / f"{stem}.png", 400.0) for stem in V6_STEMS]
    sources.extend((path, 300.0) for path in V5_IMPORTS.values())
    missing = [path.name for path, _ in sources if not path.exists()]
    if missing:
        raise FileNotFoundError(", ".join(missing))

    for source, fallback in sources:
        stem = source.stem
        for target_dpi, out_dir in ((400.0, OUT_400), (600.0, OUT_600)):
            destination = out_dir / source.name
            size = export(source, destination, target_dpi, fallback)
            verify(destination, size[:2], target_dpi)
            print(f"{destination.relative_to(ROOT)}: {size[0]} x {size[1]} px @ {target_dpi:.0f} dpi "
                  f"(source {size[2]:.1f} dpi)")


if __name__ == "__main__":
    main()
