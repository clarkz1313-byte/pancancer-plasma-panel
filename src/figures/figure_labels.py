"""Display names for figurev5 without changing internal class keys."""

DISPLAY_LABELS = {"LYMPH": "DLBCL"}


def display_label(value):
    """Return the manuscript label for an internal cancer code."""
    return DISPLAY_LABELS.get(str(value), str(value))


def display_pair(protein, cancer):
    """Format a protein-cancer pair using manuscript display labels."""
    return f"{protein} x {display_label(cancer)}"


# ─── panel letters for multi-panel mosaics ───────────────────────────────────
# Slides 12-16 render as single deck-ready mosaics from one script each, but
# carried no panel letters, so a caption could not refer to "panel c". This
# draws the letter just outside an axes' top-left corner, in axes coordinates,
# so it tracks the panel wherever GridSpec puts it (no hard-coded figure
# fractions to re-tune when a layout changes).
def panel_letter(ax, letter, dx=-0.055, dy=1.055, size=26, color="#111111"):
    """Draw a bold panel letter at the top-left of `ax`.

    dx/dy are in axes coordinates: slightly left of and above the axes box,
    which is the convention already used by the journal-style figures.
    """
    ax.text(dx, dy, letter, transform=ax.transAxes,
            ha="left", va="bottom", fontsize=size, fontweight="bold",
            color=color, clip_on=False, zorder=1000)
    return ax
