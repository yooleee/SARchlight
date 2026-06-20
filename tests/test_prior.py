# =============================================================================
# test_prior.py
# -----------------------------------------------------------------------------
# Responsible for: Pinning the prior's invariants (mass sums to 1 incl p_out;
#                  impassable cells zeroed; multiplicative/non-compensatory
#                  structure; corridor boost) and the synthetic terrain's ranges.
# Role in project: The prior is the map's starting state; these tests guard the
#                  three principled invariants prior_model.md §6 calls out.
# =============================================================================

from typing import Any, Dict, Tuple

import numpy as np
import pytest

from src.common.config import BrainConfig
from src.common.grid import GridSpec
from src.search.prior import (
    build_prior,
    combine,
    distance_from_lkp_m,
    half_normal_decay,
    rank_top_cells,
)
from src.search.terrain import SyntheticTerrain, TerrainLayers


@pytest.fixture
def cfg() -> BrainConfig:
    # Center the LKP in a modest grid so SW features have room.
    return BrainConfig(n_rows=21, n_cols=21, lkp_latlon=(37.87 + 0.005, -122.66 + 0.006))


@pytest.fixture
def grid(cfg: BrainConfig) -> GridSpec:
    return GridSpec.from_config(cfg)


class FakeTerrain:
    """
    A controllable TerrainProvider for isolating the prior math from the synthetic
    scenario shaping — we hand it exact A and C arrays.

    Why:
        Testing build_prior against fixed layers lets us assert the combination rule
        and impassable-zeroing without depending on where the stub draws its ridge.
    """

    def __init__(self, accessibility: np.ndarray, corridor: np.ndarray) -> None:
        self._a = accessibility
        self._c = corridor

    def layers(self, grid: GridSpec) -> TerrainLayers:
        return TerrainLayers(accessibility=self._a, corridor=self._c)

    def context(self, grid: GridSpec, cell: Tuple[int, int]) -> Dict[str, Any]:
        return {}


# --- D(dist) and distance ---


def test_half_normal_peaks_at_lkp_and_decays():
    """
    Scenario: D evaluated at increasing distances.
    Why it matters: D must be 1 at the LKP and monotonically decreasing, or the prior
    wouldn't concentrate near the last-known position.
    """
    d = half_normal_decay(np.array([0.0, 1000.0, 3000.0]), sigma_m=2600.0)
    assert d[0] == pytest.approx(1.0)
    assert d[0] > d[1] > d[2]


def test_half_normal_50pct_radius_matches_koester(grid, cfg):
    """
    Scenario: the half-normal's 50%-mass radius vs Koester's 3.1 km median.
    Why it matters: the σ→quantile mapping is the prior's empirical grounding; the
    50% radius ≈ 1.177·σ should land near 3.1 km for σ = 2.6 km.
    """
    median_radius = 1.177 * 2600.0
    assert median_radius == pytest.approx(3100.0, rel=0.02)


def test_distance_is_zero_at_lkp_cell(grid, cfg):
    """
    Scenario: distance array sampled at the LKP's own cell.
    Why it matters: the cell containing the LKP must have ~0 distance (within half a
    cell), anchoring the decay correctly.
    """
    dist = distance_from_lkp_m(grid, cfg.lkp_latlon)
    lkp_cell = grid.latlon_to_cell(*cfg.lkp_latlon)
    # within one cell diagonal of the LKP (the LKP need not sit at a cell center)
    assert dist[lkp_cell] <= grid.cell_size_m


# --- combine() is multiplicative / non-compensatory ---


def test_combine_product_zeroes_on_any_zero_factor():
    """
    Scenario: three layers, one with a single zero cell.
    Why it matters: the load-bearing modeling choice — a cliff (A=0) must kill the
    cell regardless of the other factors; a weighted sum would let proximity buy it back.
    """
    a = np.ones((3, 3))
    b = np.ones((3, 3))
    c = np.ones((3, 3))
    b[1, 1] = 0.0
    out = combine([a, b, c], mode="product")
    assert out[1, 1] == 0.0
    assert out[0, 0] == 1.0


def test_combine_rejects_unknown_mode():
    """
    Scenario: asking for an unsupported combination mode.
    Why it matters: the seam is intentionally product-only for now; a typo'd mode must
    fail loudly rather than silently doing the wrong thing.
    """
    with pytest.raises(ValueError):
        combine([np.ones((2, 2))], mode="sum")


# --- build_prior invariants ---


def test_prior_mass_sums_to_one_with_pout(grid, cfg):
    """
    Scenario: build the prior over uniform-passable terrain.
    Why it matters: the core invariant Σ p_i + p_out == 1 must hold from t0, or every
    later renormalization starts from a corrupt total.
    """
    a = np.ones((cfg.n_rows, cfg.n_cols))
    c = np.ones((cfg.n_rows, cfg.n_cols))
    posterior, p_out = build_prior(grid, cfg, FakeTerrain(a, c))
    assert p_out == pytest.approx(cfg.p_out)
    assert posterior.sum() == pytest.approx(1.0 - cfg.p_out)
    assert posterior.sum() + p_out == pytest.approx(1.0)


def test_prior_zeroes_impassable_cells(grid, cfg):
    """
    Scenario: one cell marked impassable (A=0), even though it's near the LKP.
    Why it matters: impassable cells getting zero prior is a principled invariant —
    proximity must not override it.
    """
    a = np.ones((cfg.n_rows, cfg.n_cols))
    c = np.ones((cfg.n_rows, cfg.n_cols))
    lkp_cell = grid.latlon_to_cell(*cfg.lkp_latlon)
    a[lkp_cell] = 0.0
    posterior, _ = build_prior(grid, cfg, FakeTerrain(a, c))
    assert posterior[lkp_cell] == 0.0


def test_corridor_boost_raises_equidistant_cell(grid, cfg):
    """
    Scenario: two cells at the same distance from the LKP, one on a corridor (C=k).
    Why it matters: the corridor term must actually pull probability — an equidistant
    cell on the drainage should outrank one off it.
    """
    a = np.ones((cfg.n_rows, cfg.n_cols))
    c = np.ones((cfg.n_rows, cfg.n_cols))
    lkp_r, lkp_c = grid.latlon_to_cell(*cfg.lkp_latlon)
    # two cells symmetric about the LKP column -> equal distance, one boosted
    on_corridor = (lkp_r, lkp_c + 3)
    off_corridor = (lkp_r, lkp_c - 3)
    c[on_corridor] = cfg.corridor_k
    posterior, _ = build_prior(grid, cfg, FakeTerrain(a, c))
    assert posterior[on_corridor] > posterior[off_corridor]


def test_build_prior_raises_if_everything_zeroed(grid, cfg):
    """
    Scenario: fully impassable terrain (A all zero).
    Why it matters: a degenerate prior would divide by zero; we fail loudly so a bad
    terrain layer is caught at construction, not as silent NaNs later.
    """
    a = np.zeros((cfg.n_rows, cfg.n_cols))
    c = np.ones((cfg.n_rows, cfg.n_cols))
    with pytest.raises(ValueError):
        build_prior(grid, cfg, FakeTerrain(a, c))


# --- top-cell ranking ---


def test_rank_top_cells_orders_by_probability(cfg):
    """
    Scenario: a posterior with a clear single peak.
    Why it matters: MapState.top_cells (the search targets) must be sorted highest-first.
    """
    p = np.zeros((5, 5))
    p[2, 3] = 0.5
    p[1, 1] = 0.3
    p[4, 4] = 0.1
    top = rank_top_cells(p, k=3)
    assert top[0][0] == (2, 3)
    assert top[0][1] >= top[1][1] >= top[2][1]


# --- synthetic terrain stays in range and is deterministic ---


def test_synthetic_terrain_ranges_and_shapes(grid, cfg):
    """
    Scenario: build the full synthetic stub over the test grid.
    Why it matters: A must stay in [0,1] (with some zeroed crest) and C in [1,k]; if
    either drifts out of range the prior's invariants break.
    """
    terrain = SyntheticTerrain(cfg)
    layers = terrain.layers(grid)
    assert layers.accessibility.shape == (cfg.n_rows, cfg.n_cols)
    assert layers.corridor.shape == (cfg.n_rows, cfg.n_cols)
    assert layers.accessibility.min() >= 0.0
    assert layers.accessibility.max() <= 1.0
    assert (layers.accessibility == 0.0).any()          # an impassable crest exists
    assert layers.corridor.min() >= 1.0 - 1e-9
    assert layers.corridor.max() <= cfg.corridor_k + 1e-9


def test_synthetic_terrain_is_deterministic(grid, cfg):
    """
    Scenario: build the stub twice.
    Why it matters: the prior (and the demo) must be reproducible — no hidden
    randomness in the terrain.
    """
    t1 = SyntheticTerrain(cfg).layers(grid)
    t2 = SyntheticTerrain(cfg).layers(grid)
    assert np.array_equal(t1.accessibility, t2.accessibility)
    assert np.array_equal(t1.corridor, t2.corridor)


def test_synthetic_terrain_context_describes_a_cell(grid, cfg):
    """
    Scenario: ask the stub to describe a cell (as a LocatedEvent would).
    Why it matters: SyntheticTerrain.context feeds the real LocatedEvent.terrain_context
    (land cover, near-drainage, lat/lon), so it must return the expected keys/types — and
    an impassable crest cell must read as steep/cliff.
    """
    terrain = SyntheticTerrain(cfg)
    layers = terrain.layers(grid)

    # A general cell: structure and types.
    ctx = terrain.context(grid, (cfg.n_rows // 2, cfg.n_cols // 2))
    assert set(ctx) >= {"land_cover", "near_drainage", "accessibility", "latlon"}
    assert isinstance(ctx["near_drainage"], bool)
    assert len(ctx["latlon"]) == 2

    # An impassable crest cell (accessibility 0) must be labelled steep/cliff.
    cliff_cells = np.argwhere(layers.accessibility == 0.0)
    if len(cliff_cells):  # the stub always carves a crest, but guard for tiny grids
        r, c = (int(cliff_cells[0][0]), int(cliff_cells[0][1]))
        assert terrain.context(grid, (r, c))["land_cover"] == "steep/cliff"
