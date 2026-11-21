"""Shared publication-readability helpers for Figure v6."""

from __future__ import annotations

from matplotlib.figure import Figure
from matplotlib.text import Text


def enable_text_scaling(
    factor: float = 1.35,
    minimum: float = 10.0,
    axis_label_scale: float = 1.0,
) -> None:
    """Scale text once before saving, with an optional axis-title adjustment.

    Dense multi-panel figures need readable annotation, but their axis titles
    should not dominate the data.  ``axis_label_scale`` is applied after the
    general readability scaling so the final visual hierarchy can match the
    restrained Figure 4B axis-label treatment.
    """

    if getattr(Figure, "_figurev6_text_scaling_enabled", False):
        return

    original_savefig = Figure.savefig

    def savefig_with_readable_text(self: Figure, *args, **kwargs):
        if not getattr(self, "_figurev6_text_scaled", False):
            for artist in self.findobj(match=Text):
                artist.set_fontsize(max(minimum, artist.get_fontsize() * factor))
            if axis_label_scale != 1.0:
                for axis in self.axes:
                    for label in (axis.xaxis.label, axis.yaxis.label):
                        if label.get_text():
                            label.set_fontsize(label.get_fontsize() * axis_label_scale)
            self._figurev6_text_scaled = True
        return original_savefig(self, *args, **kwargs)

    Figure.savefig = savefig_with_readable_text
    Figure._figurev6_text_scaling_enabled = True
