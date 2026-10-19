"""Coupling-hierarchy geometry classes for the viva-lumenoid campaign.

The viva-lumenoid multiscale campaign couples an epithelial vertex model to a
collagen-IV basement membrane (BM) across a *coupling hierarchy* of geometry
levels of increasing cost / fidelity:

  1. **2D planar sheet**      -> tyssue ``PlanarGeometry`` / ``SheetGeometry``   (exists)
  2. **2D cross-section**     -> :class:`CrossSectionGeometry`                   (added here)
  3. **3D monolayer**         -> tyssue ``MonolayerGeometry``                    (exists)
  4. **closed 3D lumenoid**   -> :class:`LumenoidGeometry`                       (added here)

Levels 1 and 3 are already provided by stock tyssue and surfaced through
``vivarium_tyssue.core_maps.GEOMETRY_MAP``. This module adds the two *missing*
levels as thin geometry classes over tyssue's existing machinery, registered
under the names ``CrossSectionGeometry`` and ``LumenoidGeometry`` so composites
can select them by name exactly like the stock geometries.

Both classes follow the tyssue geometry convention: a class of ``@staticmethod``
/ ``@classmethod`` updaters with a single public ``update_all(eptm)`` entry
point that the solver calls after every vertex-position change. They import only
numpy / pandas / ``tyssue.geometry`` — no optional viz deps (vispy etc.) — so the
module imports cleanly in a headless env.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from tyssue.geometry.planar_geometry import AnnularGeometry
from tyssue.geometry.bulk_geometry import ClosedMonolayerGeometry


class CrossSectionGeometry(AnnularGeometry):
    """Height-resolving 2D epithelial cross-section (hierarchy level 2).

    A planar apico-basal profile built on tyssue's 2D annular machinery in which
    each cell's **height** — the apical-to-basal extent — is an explicit, directly
    readable degree of freedom (written to ``face_df['height']`` and summarised in
    ``settings['mean_cell_height']``), and the basal vertices are tracked against a
    substrate (basal adhesion, ``vert_df['basal_offset']`` and
    ``settings['basal_adhesion_strain']``). It is the cheapest level of the
    coupling hierarchy at which a basement-membrane / basal perturbation produces a
    *direct* height response, so it is used to diagnose the height channel before
    paying for a full 3D monolayer.

    Builds on :class:`~tyssue.geometry.planar_geometry.AnnularGeometry` (keeping its
    enclosed ``lumen_volume``), but computes the lumen area from the ``segment``
    edge labels rather than the ``AnnularSheet.apical_edges`` property, so it also
    works when the mesh is loaded as a plain :class:`~tyssue.Sheet`.
    """

    @classmethod
    def update_all(cls, eptm):
        # AnnularGeometry.update_all runs the planar geometry update and then calls
        # ``cls.update_lumen_volume`` (dispatches to our segment-based override).
        super().update_all(eptm)
        cls.update_heights(eptm)
        cls.update_basal_adhesion(eptm)

    @staticmethod
    def update_lumen_volume(eptm):
        """Enclosed (apical) area via the 2D shoelace over the apical edges.

        Uses the ``segment`` edge labels (works on a plain ``Sheet``); falls back
        to the ``apical_edges`` property of an ``AnnularSheet`` when no labels are
        present. Result stored in ``settings['lumen_volume']``.
        """
        edge = eptm.edge_df
        if "segment" in edge.columns:
            apical = edge.index[edge["segment"] == "apical"]
        elif hasattr(eptm, "apical_edges"):
            apical = eptm.apical_edges
        else:
            eptm.settings["lumen_volume"] = 0.0
            return
        if len(apical) == 0:
            eptm.settings["lumen_volume"] = 0.0
            return
        srce_pos = eptm.upcast_srce(eptm.vert_df[["x", "y"]]).loc[apical]
        trgt_pos = eptm.upcast_trgt(eptm.vert_df[["x", "y"]]).loc[apical]
        mid = (srce_pos + trgt_pos) / 2
        d = edge.loc[apical, ["dx", "dy"]]
        eptm.settings["lumen_volume"] = float(
            (-mid["x"].values * d["dy"].values + mid["y"].values * d["dx"].values).sum()
        )

    @staticmethod
    def update_heights(eptm):
        """Per-cell apico-basal height -> ``face_df['height']``.

        Height is read *directly* from vertex positions (an explicit DOF), not
        inferred from a prescribed sheet shape: for each face it is the distance
        between the centroid of its apical source vertices and the centroid of its
        basal source vertices. Faces with no apical/basal split fall back to the
        extent of their vertices along the apico-basal axis
        (``settings['height_axis']``, default ``'y'``).
        """
        coords = eptm.coords
        vdf = eptm.vert_df
        heights = pd.Series(np.nan, index=eptm.face_df.index, dtype=float)

        if "segment" in vdf.columns:
            e = eptm.edge_df[["face"]].copy()
            e["seg"] = eptm.upcast_srce(vdf["segment"]).values
            for c in coords:
                e[c] = eptm.upcast_srce(vdf[c]).values
            ap = e[e["seg"] == "apical"].groupby("face")[coords].mean()
            ba = e[e["seg"] == "basal"].groupby("face")[coords].mean()
            common = ap.index.intersection(ba.index)
            if len(common):
                h = np.linalg.norm(ap.loc[common].values - ba.loc[common].values, axis=1)
                heights.loc[common] = h

        missing = heights.index[heights.isna()]
        if len(missing):
            axis = eptm.settings.get("height_axis", "y")
            s = eptm.edge_df.groupby("face")["s" + axis]
            ptp = (s.max() - s.min()).reindex(missing)
            heights.loc[missing] = ptp.values

        eptm.face_df["height"] = heights.values
        valid = heights.values[~np.isnan(heights.values)]
        eptm.settings["mean_cell_height"] = float(valid.mean()) if valid.size else 0.0

    @staticmethod
    def update_basal_adhesion(eptm):
        """Track the basal vertices against a substrate (basal adhesion).

        The substrate is a reference radius ``settings['basal_radius']`` (seeded
        from the initial mean basal radius, so the membrane starts at rest). Each
        basal vertex's signed deviation from it is written to
        ``vert_df['basal_offset']`` and the RMS deviation to
        ``settings['basal_adhesion_strain']`` — the readout an external BM coupler
        reads / drives.
        """
        vdf = eptm.vert_df
        offset = np.zeros(eptm.Nv)
        strain = 0.0
        if "segment" in vdf.columns:
            center = vdf[eptm.coords].mean(axis=0).values
            rho = np.linalg.norm(vdf[eptm.coords].values - center, axis=1)
            basal = vdf["segment"].values == "basal"
            if basal.any():
                ref = eptm.settings.get("basal_radius")
                if ref is None:
                    ref = float(rho[basal].mean())
                    eptm.settings["basal_radius"] = ref
                offset[basal] = rho[basal] - ref
                strain = float(np.sqrt(np.mean(offset[basal] ** 2)))
        eptm.vert_df["basal_offset"] = offset
        eptm.settings["basal_adhesion_strain"] = strain


class LumenoidGeometry(ClosedMonolayerGeometry):
    """Closed 3D lumenoid with a pressurized lumen and a BM coupling shell (level 4).

    The top level of the coupling hierarchy: a closed, curved epithelial monolayer
    enclosing a lumen. Builds on
    :class:`~tyssue.geometry.bulk_geometry.ClosedMonolayerGeometry` (which tracks
    the enclosed ``lumen_vol`` over the ``settings['lumen_side']`` faces); the new
    parts are

    * **lumen pressure / volume** — a Laplace-type lumen pressure
      ``P = lumen_elasticity * (V0 - V)`` from the tracked lumen volume ``V`` and a
      prefered volume ``V0`` (seeded from the initial volume), written to
      ``settings['lumen_pressure']`` every update so a pressure effector or an
      external coupler can read it; and
    * **basal BM shell** — the outward-facing surface is the one the collagen
      basement membrane couples to. Its total area and face count are exposed as
      ``settings['bm_surface_area']`` / ``settings['bm_face_count']`` and the face
      set via :meth:`basal_shell` / :meth:`bm_surface`, so an external BM process
      can read or write that coupling surface by name.

    ``settings['lumen_side']`` (default ``'basal'``, matching the base class and
    giving a positive volume on an extruded-inward monolayer) names the segment
    lining the lumen; ``settings['bm_side']`` names the BM shell and defaults to
    the complementary surface.
    """

    #: segment lining the lumen when ``settings['lumen_side']`` is unset
    lumen_side = "basal"

    @classmethod
    def update_all(cls, eptm):
        eptm.settings.setdefault("lumen_side", cls.lumen_side)
        super().update_all(eptm)  # bulk geometry + lumen_vol (over lumen_side faces)
        cls.update_lumen_pressure(eptm)
        cls.update_bm_surface(eptm)

    @staticmethod
    def update_lumen_pressure(eptm):
        """Laplace-type lumen pressure from the tracked volume -> settings['lumen_pressure'].

        ``P = lumen_elasticity * (V0 - V)``: pressure rises as the lumen is
        compressed below its prefered volume ``V0`` (V<V0 -> P>0, pushing the shell
        outward). ``lumen_prefered_vol`` is seeded from the first observed volume so
        the lumen starts at mechanical rest.
        """
        V = float(eptm.settings.get("lumen_vol", 0.0) or 0.0)
        if "lumen_prefered_vol" not in eptm.settings:
            eptm.settings["lumen_prefered_vol"] = V
        V0 = float(eptm.settings.get("lumen_prefered_vol", V))
        K = float(eptm.settings.get("lumen_elasticity", 0.0) or 0.0)
        P = K * (V0 - V)
        eptm.settings["lumen_pressure"] = P
        return P

    @classmethod
    def _bm_side(cls, eptm):
        side = eptm.settings.get("bm_side")
        if side:
            return side
        lumen = eptm.settings.get("lumen_side", cls.lumen_side)
        return "apical" if lumen == "basal" else "basal"

    @classmethod
    def basal_shell(cls, eptm):
        """Face indices of the outward BM-coupling shell — the surface the collagen
        basement membrane couples to (the segment named by ``bm_side``)."""
        side = cls._bm_side(eptm)
        seg = eptm.face_df.get("segment")
        if seg is None:
            return eptm.face_df.index
        return eptm.face_df.index[seg == side]

    @classmethod
    def bm_surface(cls, eptm):
        """Rich accessor for the BM coupling surface, for an external BM process:
        its face ids, centroids (when available) and total area."""
        faces = cls.basal_shell(eptm)
        fdf = eptm.face_df.loc[faces]
        coords = eptm.coords
        centroids = fdf[coords].values if set(coords) <= set(fdf.columns) else None
        area = float(fdf["area"].sum()) if "area" in fdf.columns and len(faces) else 0.0
        return {"face_ids": [int(i) for i in faces], "centroids": centroids, "area": area}

    @classmethod
    def update_bm_surface(cls, eptm):
        """Expose the BM shell area / face count on ``settings`` each update."""
        faces = cls.basal_shell(eptm)
        if "area" in eptm.face_df.columns and len(faces):
            area = float(eptm.face_df.loc[faces, "area"].sum())
        else:
            area = 0.0
        eptm.settings["bm_surface_area"] = area
        eptm.settings["bm_face_count"] = int(len(faces))
        return faces
