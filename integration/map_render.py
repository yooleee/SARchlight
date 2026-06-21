# =============================================================================
# integration/map_render.py
# -----------------------------------------------------------------------------
# Responsible for: Rendering the dashboard map's UNIFIED BASE image per frame — gray DEM
#                  hillshade + the graded-alpha posterior + the sector overlay — borderless,
#                  in the exact grid coordinate frame so the browser's vector overlays align.
# Role in project: The server half of the hybrid dashboard map. Reuses the demo's renderer
#                  (draw_belief_layer + _draw_sectors + hillshade) so the dashboard base looks
#                  identical to the 3-drone showcase, while the live drones/route/markers are
#                  drawn as crisp vectors on top in the browser.
# Alignment contract (the crux): the image is the DATA AREA ONLY (no axes/margins), with
#   imshow's default extent [-0.5, n-0.5] and origin='lower'. So cell (r, c)'s CENTER sits at
#   image fraction ((c+0.5)/n_cols, 1 - (r+0.5)/n_rows) — identical to dashboard_projection's
#   cell-center _cell_to_norm. Image and vectors therefore share one frame by construction.
# Assumptions: matplotlib is already a project dep (the demo uses it). Rendering happens in the
#              server's stepper thread (off the request path), cached per frame index.
# =============================================================================

from __future__ import annotations

import io
from typing import Optional, Sequence

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np

from src.common.grid import GridSpec
# Reuse the demo's belief + sector rendering (DRY — one look for the GIFs and the dashboard).
from src.demo.search_demo import _draw_sectors
from src.demo.showcase import draw_belief_layer, hillshade

# Square output edge in pixels. The 160x160 grid is square, so a square image fills the data
# area with no letterboxing; the browser then object-fill-stretches it to the panel.
_SIZE_PX = 1000
_DPI = 100


def render_base_frame(
    grid: GridSpec,
    posterior: np.ndarray,
    hill: np.ndarray,
    *,
    planner=None,
    ranked_sectors: Sequence = (),
    drones: Sequence = (),
    size_px: int = _SIZE_PX,
) -> bytes:
    """
    Render one base map frame (hillshade + graded-alpha posterior + sectors) to PNG bytes.

    Args:
        grid: The shared GridSpec (fixes the coordinate frame / extent).
        posterior: (n_rows, n_cols) belief to overlay with graded alpha (the REAL posterior,
            not a blob approximation).
        hill: Precomputed hillshade for the grid (pass it in so the caller computes it once).
        planner: The SectorPlanner, needed only to draw the sector overlay. None -> no sectors
            (used for the guide-home phase, where the posterior is frozen and sectors are moot).
        ranked_sectors: The frame's ranked sectors (for the top-K outlines + POA labels).
        drones: The frame's DroneViews (to outline each drone's assigned sector in its color).
        size_px: Square output edge in pixels.

    Returns:
        PNG-encoded bytes of the borderless base image (north-up).

    Why:
        This is the unified base the dashboard needs: by reusing draw_belief_layer (graded alpha
        composites the belief INTO the terrain, the demo's integrated look) and _draw_sectors,
        the dashboard map matches the showcase exactly. Rendering the DATA AREA ONLY (axes off,
        axes box filling the figure, default imshow extent) is what makes the pixel frame equal
        the cell-center normalization the browser vectors use.
    """
    fig = plt.figure(figsize=(size_px / _DPI, size_px / _DPI), dpi=_DPI)
    # Axes fill the ENTIRE figure (no margins) -> the data area is the whole image.
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()

    draw_belief_layer(ax, posterior, hill)  # gray hillshade + graded-alpha posterior (origin='lower')
    if planner is not None and len(ranked_sectors):
        _draw_sectors(ax, planner, list(ranked_sectors), list(drones))

    # Pin the extent to imshow's default cell-edge extent and force the axes to FILL the box
    # (aspect 'auto' avoids any letterboxing), so cell centers land at ((c+0.5)/n, ...).
    ax.set_xlim(-0.5, grid.n_cols - 0.5)
    ax.set_ylim(-0.5, grid.n_rows - 0.5)  # origin='lower' -> ascending y is north-up
    ax.set_aspect("auto")

    buffer = io.BytesIO()
    # pad_inches=0 (and NO bbox_inches='tight', which would re-introduce padding) keeps it borderless.
    fig.savefig(buffer, format="png", dpi=_DPI, pad_inches=0)
    plt.close(fig)
    return buffer.getvalue()


def base_hillshade(grid: GridSpec) -> np.ndarray:
    """
    The hillshade for the grid (thin pass-through to the demo's hillshade, for the server to cache).

    Args:
        grid: The shared GridSpec.

    Returns:
        (n_rows, n_cols) shaded relief in [0, 1].

    Why:
        The hillshade never changes within a run, so the server computes it ONCE and passes it to
        every render_base_frame call. Exposing it here keeps map_render the single import surface
        for the base map (the server doesn't reach into the demo modules directly).
    """
    return hillshade(grid)
