import numpy as np
import pandas as pd

# cell types of the model:pc,gc,ent,dcs
cell_types = ("sc", "pc", "ent", "gc", "dcs")

# Construct a dictionary which convert cell kind to integer to store them in a numpy array format
cell_to_int = {j: i for i, j in enumerate(cell_types)}
int_to_cell = {i: j for i, j in enumerate(cell_types)}

######################################
# JUMPS
# the jump types of the model (in terms of identity of the new cell, whether it is a differentiation
# or a division)
jump_types = ("sc", "pc", "gc", "ex", "ent")

# Construct a dictionary which convert jump type to integer to store them in a numpy array format
jump_to_int = {i: x for x, i in enumerate(jump_types)}
int_to_jump = {x: i for x, i in enumerate(jump_types)}


def spatial_prob(y, y_min, y_max):
    """
    Spatial probability weights based on y position within tissue.

    Parameters
    ----------
    x, y   : float  — cell centroid coordinates
    y_min  : float  — minimum y in the tissue
    y_max  : float  — maximum y in the tissue
    """
    # Normalise y to [0, 1]
    y_norm = (y - y_min) / (y_max - y_min)

    # ── SC: concentrated in bottom 10% ───────────────────────────────────────
    # Sharp Gaussian centred at y=0, effectively zero above 10%
    sc = np.exp(-((y_norm - 0.0) ** 2) / (2 * 0.05 ** 2))

    # ── PC: bottom 10%–40%, linearly decreasing ───────────────────────────────
    if y_norm < 0.10:
        pc = 0.0
    elif y_norm <= 0.40:
        pc = 1.0 - (y_norm - 0.10) / (0.40 - 0.10)  # 1 → 0 over [0.1, 0.4]
    else:
        pc = 0.0

    # ── ENT & GOB: 25%–100%, linearly increasing ──────────────────────────────
    if y_norm < 0.25:
        ent = gc = 0.0
    else:
        ent = gc = (y_norm - 0.25) / (1.0 - 0.25)  # 0 → 1 over [0.25, 1.0]

    return {"sc": sc, "pc": pc, "ent": ent, "gc": gc}


def assign_cell_types(eptm, axis, spatial_prob_func=None, random_seed=None):
    """
    Assign cell types to cells in a Tyssue epithelium model.

    Parameters
    ----------
    eptm : Epithelium
        A Tyssue vertex model object with a face DataFrame
    spatial_prob_func : callable or None
        Function(x, y) -> dict mapping cell_type -> spatial weight.
        If None, assignment is purely proportion-based.
    random_seed : int or None
        For reproducibility.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    cell_types = ["sc", "pc", "ent", "gc"]
    base_props = np.array([0.03, 0.22, 0.564, 0.186])  # must sum to 1

    n_cells = len(eptm.face_df)

    # ── Compute per-cell type probabilities ──────────────────────────────────
    if spatial_prob_func is not None:
        # Get centroid of each cell
        centroids = eptm.face_df[[axis]].values  # shape (n_cells, 2)

        # Build probability matrix: rows = cells, cols = cell types
        prob_matrix = np.zeros((n_cells, len(cell_types)))
        y_min = min(eptm.face_df[axis])
        y_max = max(eptm.face_df[axis])
        for i, (y) in enumerate(centroids):
            spatial_weights = spatial_prob_func(y, y_min, y_max)  # dict {type: weight}
            for j, ct in enumerate(cell_types):
                prob_matrix[i, j] = base_props[j] * spatial_weights.get(ct, 1.0)

        # Normalise each row so probabilities sum to 1
        row_sums = prob_matrix.sum(axis=1, keepdims=True)
        prob_matrix /= row_sums
    else:
        # All cells share the same flat proportions
        prob_matrix = np.tile(base_props, (n_cells, 1))

    # ── Sample one cell type per cell ────────────────────────────────────────
    assigned = [
        np.random.choice(cell_types, p=prob_matrix[i])
        for i in range(n_cells)
    ]

    eptm.face_df["cell_type"] = assigned
    return eptm
