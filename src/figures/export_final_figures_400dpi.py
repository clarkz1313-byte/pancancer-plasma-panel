#!/usr/bin/env python3
"""Export the approved final figure PNGs at 400 dpi.

The three PIL-composed mosaics (Figures 4, 5, and 7) do not carry PNG DPI
metadata, but were assembled at the project's 300-dpi working scale.  They
therefore use 300 dpi as the explicit source assumption.  Other figures use
their embedded PNG metadata.  Pixel dimensions are scaled to preserve the
same physical size at 400 dpi; an existing 400-dpi figure is not resampled.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image

# These are trusted, locally generated scientific figures; several mosaics
# intentionally exceed Pillow's generic web-image warning threshold.
Image.MAX_IMAGE_PIXELS = None


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "final_figures"
OUTPUT_DIR = ROOT / "final_figures_400dpi"
TARGET_DPI = 400.0
DEFAULT_SOURCE_DPI = 300.0

EXPECTED_STEMS = (
    "fig3_slide3_derivation_mosaic",
    "fig4_slide4_structure_mosaic",
    "fig5_slide5_multiclass_mosaic",
    "fig7_slide7_singleclass_mosaic",
    "fig8_slide8_confirmation_mosaic",
    "fig9_slide9_screening_mosaic",
    "fig10_5_ppi_combined",
    "fig12_pathway_permutation_mosaic",
)


def _source_dpi(image: Image.Image) -> float:
    dpi = image.info.get("dpi")
    if not dpi:
        return DEFAULT_SOURCE_DPI
    x_dpi, y_dpi = map(float, dpi)
    if abs(x_dpi - y_dpi) > 1.0:
        raise ValueError(f"non-square source DPI: {dpi}")
    return (x_dpi + y_dpi) / 2.0


def export_one(source: Path, destination: Path) -> tuple[int, int, float]:
    with Image.open(source) as image:
        image.load()
        source_dpi = _source_dpi(image)
        scale = TARGET_DPI / source_dpi
        target_size = tuple(max(1, round(n * scale)) for n in image.size)
        if target_size != image.size:
            image = image.resize(target_size, Image.Resampling.LANCZOS)
        image.save(destination, format="PNG", dpi=(TARGET_DPI, TARGET_DPI),
                   compress_level=6)
    return target_size[0], target_size[1], source_dpi


def verify_export(path: Path, expected_size: tuple[int, int]) -> None:
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        dpi = image.info.get("dpi")
        if image.size != expected_size:
            raise RuntimeError(f"unexpected dimensions for {path.name}: {image.size}")
        if not dpi or any(abs(float(value) - TARGET_DPI) > 1.0 for value in dpi):
            raise RuntimeError(f"unexpected DPI for {path.name}: {dpi}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sources = [SOURCE_DIR / f"{stem}.png" for stem in EXPECTED_STEMS]
    missing = [path.name for path in sources if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing approved final figures: {', '.join(missing)}")

    print(f"Exporting {len(sources)} figures to {OUTPUT_DIR}")
    for source in sources:
        destination = OUTPUT_DIR / source.name
        width, height, source_dpi = export_one(source, destination)
        verify_export(destination, (width, height))
        print(
            f"  {source.name}: source {source_dpi:.1f} dpi -> "
            f"{width} x {height} px @ 400 dpi"
        )


if __name__ == "__main__":
    main()
