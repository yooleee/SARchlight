# =============================================================================
# tests/test_map_render.py
# -----------------------------------------------------------------------------
# Responsible for: Verifying the hybrid map base renderer (integration/map_render.py) and the
#                  cell-center normalization (dashboard_projection._cell_to_norm) AGREE on the
#                  coordinate frame — the alignment guarantee the whole hybrid map rests on.
# Role in project: If the base image and the vector overlays don't share a frame, the drones
#                  and route would float off the terrain. This test pins that they don't, by
#                  rendering a single hot cell and checking its brightest pixel lands exactly
#                  where _cell_to_norm places it.
# Assumptions: No rasters needed — we pass a flat hillshade and a synthetic posterior, so the
#              test is deterministic and fast (the real DEM hillshade is exercised elsewhere).
# =============================================================================

from __future__ import annotations

import io

import numpy as np
from PIL import Image

from src.common.config import BrainConfig
from src.common.grid import GridSpec
from integration.dashboard_projection import _cell_to_norm
from integration.map_render import render_base_frame


def _grid() -> GridSpec:
    return GridSpec.from_config(BrainConfig())  # 160 x 160


def test_cell_to_norm_is_cell_center():
    """
    Scenario: normalize a few known cells.
    Why it matters: the base image (a matplotlib imshow spanning [-0.5, n-0.5]) places cell
    centers at (c+0.5)/n — so the vector frame must use that exact form, not the old corner form.
    """
    grid = _grid()
    n = grid.n_cols
    # Cell (0,0) center -> (0.5/n, 1 - 0.5/n); flipped y keeps north up.
    assert _cell_to_norm((0, 0), grid) == {"x": round(0.5 / n, 4), "y": round(1 - 0.5 / n, 4)}
    # Last cell -> near (1, 0). (Tolerance covers _cell_to_norm's 4-decimal rounding.)
    last = _cell_to_norm((n - 1, n - 1), grid)
    assert abs(last["x"] - (n - 0.5) / n) < 1e-4 and abs(last["y"] - 0.5 / n) < 1e-4
    # x increases with column; y decreases with row (north up).
    assert _cell_to_norm((10, 20), grid)["x"] < _cell_to_norm((10, 80), grid)["x"]
    assert _cell_to_norm((20, 10), grid)["y"] > _cell_to_norm((80, 10), grid)["y"]


def test_base_frame_is_a_valid_borderless_png():
    """
    Scenario: render a normal posterior to a base frame.
    Why it matters: the server serves these bytes as image/png and the browser stretches them to
    the map panel; assert a real, square, non-degenerate PNG (so the stretch is predictable).
    """
    grid = _grid()
    rng = np.random.default_rng(0)
    posterior = rng.random((grid.n_rows, grid.n_cols)) ** 4  # a peaky-ish belief
    hill = np.full((grid.n_rows, grid.n_cols), 0.5)
    png = render_base_frame(grid, posterior, hill, size_px=400)

    img = Image.open(io.BytesIO(png))
    assert img.format == "PNG"
    assert img.size == (400, 400)
    assert np.asarray(img.convert("RGB")).std() > 2.0  # real variation, not a solid fill


def test_hot_cell_lands_where_cell_to_norm_says():
    """
    Scenario: a posterior that's flat-tiny everywhere except ONE bright cell, rendered over a flat
    hillshade; find the brightest pixel and compare its image fraction to _cell_to_norm(cell).
    Why it matters: this is the alignment PROOF — it ties the base image's pixel frame to the
    vector normalization. If they ever diverge, the drones/route would drift off the terrain.
    """
    grid = _grid()
    hot = (40, 120)  # off-center on both axes so x and y are tested distinctly
    posterior = np.full((grid.n_rows, grid.n_cols), 1e-6)
    posterior[hot] = 1.0
    hill = np.full((grid.n_rows, grid.n_cols), 0.5)

    size = 1000
    png = render_base_frame(grid, posterior, hill, size_px=size)
    arr = np.asarray(Image.open(io.BytesIO(png)).convert("RGB")).astype(int)

    # The hot cell renders as bright inferno yellow (high R+G) over mid-gray; argmax finds it.
    brightness = arr[:, :, 0] + arr[:, :, 1]
    py, px = np.unravel_index(int(np.argmax(brightness)), brightness.shape)
    frac_x, frac_y = px / size, py / size

    expected = _cell_to_norm(hot, grid)
    # Tolerance ~2 cells (1 cell = 1/160 ≈ 0.006 of the frame) covers block size + antialiasing.
    assert abs(frac_x - expected["x"]) < 0.015, f"x off: {frac_x} vs {expected['x']}"
    assert abs(frac_y - expected["y"]) < 0.015, f"y off: {frac_y} vs {expected['y']}"
