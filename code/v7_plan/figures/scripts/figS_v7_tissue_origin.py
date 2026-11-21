"""v7 Supplementary Figure: tissue and cell of origin of the disease-specific panels (moved from Fig 7 A-B,
author 17 Oct).
(A) One ring per panel: one wedge per panel protein; coloured = gene mainly expressed in the cancer's own tissue or cell
    of origin (HPA 25.1); black tick = number expected by chance; * BH q < 0.05 (B9b).
(B) The 21 such proteins in the 12 panels against random panels of the same sizes drawn from all measured proteins,
    from the proteins that passed each cancer's screen, and from each cancer's LASSO-ranked proteins (mean, 95% range
    of 10,000 draws; B9e).
Drawing functions from fig7_v7c_biology.py. Output: figures/output/figS_v7_tissue_origin.png
"""
import v7_style as S
import fig7_v7c_biology as B

plt = S.plt


def main():
    fig = plt.figure(figsize=(26, 8.6))
    gs = fig.add_gridspec(1, 6, wspace=0.55, left=0.03, right=0.985, top=0.93, bottom=0.12)
    axA = B.panel_a(fig, gs[0, 0:4])
    axB = fig.add_subplot(gs[0, 4:6]); B.panel_b(axB)
    axB.set_position(axB.get_position().translated(0.03, 0))
    S.letter(fig, axA, "A", dx=-0.02, dy=0.01)
    S.letter(fig, axB, "B", dx=-0.075, dy=0.01)
    S.save(fig, "figS_v7_tissue_origin.png")


if __name__ == "__main__":
    main()
