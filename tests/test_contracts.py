# =============================================================================
# test_contracts.py
# -----------------------------------------------------------------------------
# Responsible for: The MapState serialization seam (to_dict) — the contract method
#                  that lets state cross a process/socket boundary later unchanged.
# Role in project: Small but important: this is the seam we promised to "build now,
#                  defer the transport," so it must actually produce plain JSON-able
#                  data (arrays -> lists, enum -> string).
# =============================================================================

import json

import numpy as np

from src.common.config import BrainConfig
from src.common.contracts import MapState, Status
from src.common.grid import GridSpec


def _map_state(next_target):
    cfg = BrainConfig(n_rows=4, n_cols=4)
    grid = GridSpec.from_config(cfg)
    return MapState(
        grid_spec=grid,
        update_count=2,
        timestamp=5.0,
        posterior=np.full((4, 4), 1.0 / 16),
        coverage=np.zeros((4, 4)),
        top_cells=[((1, 2), 0.5)],
        next_target=next_target,
        status=Status.SEARCHING,
        p_out=0.07,
    )


def test_to_dict_is_json_serializable_and_well_typed():
    """
    Scenario: serialize a populated MapState.
    Why it matters: the read model must become plain, JSON-dumpable data — arrays as
    nested lists, status as its string value, grid_spec flattened — or a future
    dashboard/Redis boundary can't consume it without a contract change.
    """
    ms = _map_state(next_target=(1, 2))
    d = ms.to_dict()

    # Must round-trip through json with no custom encoder.
    text = json.dumps(d)
    assert isinstance(text, str)

    assert isinstance(d["posterior"], list) and isinstance(d["posterior"][0], list)
    assert d["status"] == "searching"          # enum -> its string value
    assert d["next_target"] == [1, 2]          # tuple -> list
    assert d["top_cells"] == [[[1, 2], 0.5]]   # nested cells/probs as lists
    assert d["grid_spec"]["n_rows"] == 4


def test_to_dict_handles_absent_next_target():
    """
    Scenario: a MapState with no next_target (degenerate/fully-covered map).
    Why it matters: the None branch must serialize as JSON null, not crash on list().
    """
    d = _map_state(next_target=None).to_dict()
    assert d["next_target"] is None
