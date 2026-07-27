//! Numeric kernels for the tyssue vertex-model hot loop.
//!
//! Pure Rust, no Python. Each function mirrors the corresponding tyssue
//! (`SheetGeometry` / gradient-assembly) computation closely enough to be
//! bit-comparable (to ~1e-12) against the Python reference — that equivalence
//! is enforced by `tests/test_rust_kernels_equiv.py` on the Python side.
//!
//! Convention: vertex positions are a row-major `(n_vert, dim)` slice; `srce`
//! and `trgt` are POSITIONAL vertex indices (0..n_vert), already remapped from
//! tyssue's DataFrame index by the caller.

/// Per-edge Euclidean length: `||pos[trgt] - pos[srce]||`.
///
/// Mirrors tyssue `update_length` (Euclidean norm of the edge vector) for the
/// non-periodic case. Returns one length per edge.
pub fn edge_lengths(pos: &[f64], srce: &[u32], trgt: &[u32], dim: usize) -> Vec<f64> {
    let ne = srce.len();
    let mut out = vec![0.0f64; ne];
    for e in 0..ne {
        let s = srce[e] as usize * dim;
        let t = trgt[e] as usize * dim;
        let mut acc = 0.0f64;
        for d in 0..dim {
            let delta = pos[t + d] - pos[s + d];
            acc += delta * delta;
        }
        out[e] = acc.sqrt();
    }
    out
}

/// Scatter-add per-edge vectors onto vertices by index — the edge->vertex
/// gradient assembly, i.e. the equivalent of `np.add.at(out, index, values)`.
///
/// `values` is `(n_edge, dim)` row-major; `index[e]` is the destination vertex
/// of edge `e`. Returns `(n_vert, dim)` row-major (flattened). This is the
/// operation that replaces the two per-step `groupby(...).sum()` calls in
/// `compute_gradient`.
pub fn scatter_add(values: &[f64], index: &[u32], n_vert: usize, dim: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; n_vert * dim];
    let ne = index.len();
    for e in 0..ne {
        let dst = index[e] as usize * dim;
        let src = e * dim;
        for d in 0..dim {
            out[dst + d] += values[src + d];
        }
    }
    out
}

/// Full per-step geometry update for a 3D sheet — the stateless core of
/// tyssue's `SheetGeometry.update_all`, in one pass over the edges.
///
/// Reproduces, with identical formulas and order:
///   * `update_dcoords`  — edge vectors `d = pos[trgt] - pos[srce]`
///   * `update_length`   — `||d||`
///   * `update_centroid` — face centroid = mean of source-vertex positions
///                         over the face's edges; and `r = srce_pos - centroid`
///   * `update_normals`  — `cross(r, d)` per edge (3D only)
///   * `update_areas`    — `sub_area = ||normal|| / 2`, `area = Σ sub_area`
///   * `update_perimeters` — `perimeter = Σ length`
///
/// Deliberately excludes `update_ucoords` (divides by the *stale* length from
/// the previous update — stateful) and `update_vol` (needs geometry-specific
/// vertex heights). Those stay in Python for now.
///
/// `face` is the positional face index (0..n_face) of each edge. `dim` must be
/// 3 (all demo meshes are 3D sheets/monolayers).
pub struct Geometry {
    pub dcoords: Vec<f64>,   // (n_edge, dim)
    pub length: Vec<f64>,    // (n_edge,)
    pub centroid: Vec<f64>,  // (n_face, dim)
    pub rcoords: Vec<f64>,   // (n_edge, dim)  srce_pos - face_centroid
    pub normals: Vec<f64>,   // (n_edge, 3)
    pub sub_area: Vec<f64>,  // (n_edge,)
    pub area: Vec<f64>,      // (n_face,)
    pub perimeter: Vec<f64>, // (n_face,)
}

pub fn update_geometry(
    pos: &[f64],
    srce: &[u32],
    trgt: &[u32],
    face: &[u32],
    n_face: usize,
    dim: usize,
) -> Geometry {
    assert_eq!(dim, 3, "update_geometry currently supports 3D meshes only");
    let ne = srce.len();

    let mut dcoords = vec![0.0f64; ne * dim];
    let mut length = vec![0.0f64; ne];
    let mut centroid = vec![0.0f64; n_face * dim];
    let mut rcoords = vec![0.0f64; ne * dim];
    let mut normals = vec![0.0f64; ne * 3];
    let mut sub_area = vec![0.0f64; ne];
    let mut area = vec![0.0f64; n_face];
    let mut perimeter = vec![0.0f64; n_face];
    let mut face_count = vec![0u32; n_face];

    // Pass 1: edge vectors + lengths; accumulate face centroid = mean of srce pos.
    for e in 0..ne {
        let s = srce[e] as usize * dim;
        let t = trgt[e] as usize * dim;
        let f = face[e] as usize;
        let eo = e * dim;
        let mut acc = 0.0f64;
        for d in 0..dim {
            let delta = pos[t + d] - pos[s + d];
            dcoords[eo + d] = delta;
            acc += delta * delta;
            centroid[f * dim + d] += pos[s + d];
        }
        length[e] = acc.sqrt();
        face_count[f] += 1;
    }
    for f in 0..n_face {
        let c = face_count[f];
        if c > 0 {
            let inv = 1.0 / c as f64;
            for d in 0..dim {
                centroid[f * dim + d] *= inv;
            }
        }
    }

    // Pass 2: r = srce - centroid; normal = cross(r, d); sub_area; reduce to face.
    for e in 0..ne {
        let s = srce[e] as usize * dim;
        let f = face[e] as usize;
        let eo = e * dim;
        let rx = pos[s] - centroid[f * dim];
        let ry = pos[s + 1] - centroid[f * dim + 1];
        let rz = pos[s + 2] - centroid[f * dim + 2];
        rcoords[eo] = rx;
        rcoords[eo + 1] = ry;
        rcoords[eo + 2] = rz;
        let dx = dcoords[eo];
        let dy = dcoords[eo + 1];
        let dz = dcoords[eo + 2];
        // cross(r, d)
        let nx = ry * dz - rz * dy;
        let ny = rz * dx - rx * dz;
        let nz = rx * dy - ry * dx;
        normals[eo] = nx;
        normals[eo + 1] = ny;
        normals[eo + 2] = nz;
        let sa = (nx * nx + ny * ny + nz * nz).sqrt() / 2.0;
        sub_area[e] = sa;
        area[f] += sa;
        perimeter[f] += length[e];
    }

    Geometry {
        dcoords,
        length,
        centroid,
        rcoords,
        normals,
        sub_area,
        area,
        perimeter,
    }
}

/// Result of `update_geometry_bulk` — the Monolayer/Bulk `update_all` core.
pub struct BulkGeometry {
    pub dcoords: Vec<f64>,       // (n_edge, 3)   trgt - srce
    pub length: Vec<f64>,        // (n_edge,)
    pub face_centroid: Vec<f64>, // (n_face, 3)   length-weighted (RNRGeometry)
    pub cell_centroid: Vec<f64>, // (n_cell, 3)   mean of srce over the cell
    pub rcoords: Vec<f64>,       // (n_edge, 3)   srce - face_centroid
    pub normals: Vec<f64>,       // (n_edge, 3)   cross(r, d)
    pub sub_area: Vec<f64>,      // (n_edge,)     ||normal|| / 2
    pub face_area: Vec<f64>,     // (n_face,)     Σ_face sub_area
    pub cell_area: Vec<f64>,     // (n_cell,)     Σ_cell sub_area
    pub sub_vol: Vec<f64>,       // (n_edge,)     ((f - c)·n) / 6
    pub cell_vol: Vec<f64>,      // (n_cell,)     Σ_cell sub_vol
    pub perimeter: Vec<f64>,     // (n_face,)     Σ_face length
}

/// Bulk / Monolayer `update_all` core (BulkGeometry + RNRGeometry): like the
/// sheet kernel but with a **length-weighted** face centroid, per-**cell**
/// centroids / areas / volumes, and the tetrahedral sub-volume. Reproduces
/// `MonolayerGeometry.update_all` bit-identically (minus the stale `ucoords`,
/// which the caller derives from the previous length, exactly as for the sheet).
///
/// `cell` is the positional cell index (0..n_cell) of each edge. Excludes the
/// `update_ucoords` stale-length step (stateful — done in Python).
pub fn update_geometry_bulk(
    pos: &[f64],
    srce: &[u32],
    trgt: &[u32],
    face: &[u32],
    cell: &[u32],
    n_face: usize,
    n_cell: usize,
) -> BulkGeometry {
    let dim = 3usize;
    let ne = srce.len();
    let mut dcoords = vec![0.0f64; ne * dim];
    let mut length = vec![0.0f64; ne];
    let mut perimeter = vec![0.0f64; n_face];
    let mut fc_weighted = vec![0.0f64; n_face * dim]; // Σ mid*length
    let mut face_centroid = vec![0.0f64; n_face * dim];
    let mut cell_centroid = vec![0.0f64; n_cell * dim];
    let mut cell_count = vec![0u32; n_cell];
    let mut rcoords = vec![0.0f64; ne * dim];
    let mut normals = vec![0.0f64; ne * dim];
    let mut sub_area = vec![0.0f64; ne];
    let mut face_area = vec![0.0f64; n_face];
    let mut cell_area = vec![0.0f64; n_cell];
    let mut sub_vol = vec![0.0f64; ne];
    let mut cell_vol = vec![0.0f64; n_cell];

    // Pass 1: edge vectors + lengths; accumulate perimeter, the length-weighted
    // face centroid (Σ mid*length), and the cell centroid (Σ srce, count).
    for e in 0..ne {
        let s = srce[e] as usize * dim;
        let t = trgt[e] as usize * dim;
        let f = face[e] as usize;
        let c = cell[e] as usize;
        let eo = e * dim;
        let mut acc = 0.0f64;
        for d in 0..dim {
            let delta = pos[t + d] - pos[s + d];
            dcoords[eo + d] = delta;
            acc += delta * delta;
            cell_centroid[c * dim + d] += pos[s + d];
        }
        let len = acc.sqrt();
        length[e] = len;
        perimeter[f] += len;
        cell_count[c] += 1;
        // weighted face centroid contribution: mid = (srce+trgt)/2, weight = length
        for d in 0..dim {
            let mid = (pos[s + d] + pos[t + d]) * 0.5;
            fc_weighted[f * dim + d] += mid * len;
        }
    }
    for f in 0..n_face {
        let p = perimeter[f];
        if p > 0.0 {
            let inv = 1.0 / p;
            for d in 0..dim {
                face_centroid[f * dim + d] = fc_weighted[f * dim + d] * inv;
            }
        }
    }
    for c in 0..n_cell {
        let cnt = cell_count[c];
        if cnt > 0 {
            let inv = 1.0 / cnt as f64;
            for d in 0..dim {
                cell_centroid[c * dim + d] *= inv;
            }
        }
    }

    // Pass 2: r = srce - face_centroid; normal = cross(r, d); sub_area; sub_vol
    // = ((face_centroid - cell_centroid)·normal)/6; reduce to face and cell.
    for e in 0..ne {
        let s = srce[e] as usize * dim;
        let f = face[e] as usize;
        let c = cell[e] as usize;
        let eo = e * dim;
        let rx = pos[s] - face_centroid[f * dim];
        let ry = pos[s + 1] - face_centroid[f * dim + 1];
        let rz = pos[s + 2] - face_centroid[f * dim + 2];
        rcoords[eo] = rx;
        rcoords[eo + 1] = ry;
        rcoords[eo + 2] = rz;
        let dx = dcoords[eo];
        let dy = dcoords[eo + 1];
        let dz = dcoords[eo + 2];
        let nx = ry * dz - rz * dy;
        let ny = rz * dx - rx * dz;
        let nz = rx * dy - ry * dx;
        normals[eo] = nx;
        normals[eo + 1] = ny;
        normals[eo + 2] = nz;
        let sa = (nx * nx + ny * ny + nz * nz).sqrt() / 2.0;
        sub_area[e] = sa;
        face_area[f] += sa;
        cell_area[c] += sa;
        // sub_vol = dot(face_centroid - cell_centroid, normal) / 6
        let fcx = face_centroid[f * dim] - cell_centroid[c * dim];
        let fcy = face_centroid[f * dim + 1] - cell_centroid[c * dim + 1];
        let fcz = face_centroid[f * dim + 2] - cell_centroid[c * dim + 2];
        let sv = (fcx * nx + fcy * ny + fcz * nz) / 6.0;
        sub_vol[e] = sv;
        cell_vol[c] += sv;
    }

    BulkGeometry {
        dcoords,
        length,
        face_centroid,
        cell_centroid,
        rcoords,
        normals,
        sub_area,
        face_area,
        cell_area,
        sub_vol,
        cell_vol,
        perimeter,
    }
}

/// Fused gradient for the standard 3-effector sheet model
/// (LineTension + PerimeterElasticity + FaceAreaElasticity) plus the
/// edge->vertex assembly — i.e. all of `model.compute_gradient` for that model.
///
/// It *consumes* the geometry columns tyssue already wrote (ucoords, normals,
/// sub_area, r_ak = srce-face, r_aj = trgt-face) rather than recomputing them,
/// so it reproduces `compute_gradient` exactly — including tyssue's quirk that
/// `ucoords` is normalized by the previous step's (stale) length. It only
/// replaces the pandas arithmetic + the two `groupby.sum()` reductions.
///
/// Per-edge tension coefficient is `0.5*line_active + gamma_face[f]`
/// (LineTension carries the 0.5 on half-edges; PerimeterElasticity does not).
/// Area term is `ka_face[f] * inv_area * cross(...)`, `inv_area = 1/(4*sub_area)`
/// (0 where sub_area==0). Returns grad_i as (n_vert, 3), divided by norm_factor.
#[allow(clippy::too_many_arguments)]
pub fn sheet_gradient(
    ucoords: &[f64],     // (n_edge, 3) unit edge vectors, as tyssue stored them
    normals: &[f64],     // (n_edge, 3)
    sub_area: &[f64],    // (n_edge,)
    r_ak: &[f64],        // (n_edge, 3) srce_pos - face_pos
    r_aj: &[f64],        // (n_edge, 3) trgt_pos - face_pos
    srce: &[u32],
    trgt: &[u32],
    face: &[u32],
    line_active: &[f64], // (n_edge,) line_tension * is_active
    gamma_face: &[f64],  // (n_face,) perim_elasticity*is_alive*(perimeter-prefered)
    ka_face: &[f64],     // (n_face,) area_elasticity*is_alive*(area-prefered)
    boundary: &[u8],     // (n_vert,) 1 => clamp this vertex's gradient to 0
                         // (model_factory_bound); all-zeros for model_factory.
    n_vert: usize,
    norm_factor: f64,
) -> Vec<f64> {
    let ne = srce.len();
    let mut grad = vec![0.0f64; n_vert * 3];
    for e in 0..ne {
        let f = face[e] as usize;
        let eo = e * 3;
        let coeff = 0.5 * line_active[e] + gamma_face[f];
        let sa = sub_area[e];
        let inv_area = if sa != 0.0 { 1.0 / (4.0 * sa) } else { 0.0 };
        let kia = ka_face[f] * inv_area;

        let (ux, uy, uz) = (ucoords[eo], ucoords[eo + 1], ucoords[eo + 2]);
        let (nx, ny, nz) = (normals[eo], normals[eo + 1], normals[eo + 2]);
        let (akx, aky, akz) = (r_ak[eo], r_ak[eo + 1], r_ak[eo + 2]);
        let (ajx, ajy, ajz) = (r_aj[eo], r_aj[eo + 1], r_aj[eo + 2]);

        // area gradient: srce uses cross(r_aj, normal), trgt uses cross(normal, r_ak)
        let cs_x = ajy * nz - ajz * ny;
        let cs_y = ajz * nx - ajx * nz;
        let cs_z = ajx * ny - ajy * nx;
        let ct_x = ny * akz - nz * aky;
        let ct_y = nz * akx - nx * akz;
        let ct_z = nx * aky - ny * akx;

        let gs = [-ux * coeff + kia * cs_x, -uy * coeff + kia * cs_y, -uz * coeff + kia * cs_z];
        let gt = [ux * coeff + kia * ct_x, uy * coeff + kia * ct_y, uz * coeff + kia * ct_z];

        let sv = srce[e] as usize * 3;
        let tv = trgt[e] as usize * 3;
        for d in 0..3 {
            grad[sv + d] += gs[d];
            grad[tv + d] += gt[d];
        }
    }
    // model_factory_bound: clamp boundary vertices to zero after assembly.
    for v in 0..n_vert {
        if boundary.get(v).copied().unwrap_or(0) == 1 {
            grad[v * 3] = 0.0;
            grad[v * 3 + 1] = 0.0;
            grad[v * 3 + 2] = 0.0;
        }
    }
    if norm_factor != 1.0 {
        let inv = 1.0 / norm_factor;
        for g in grad.iter_mut() {
            *g *= inv;
        }
    }
    grad
}

/// Result of `update_geometry_planar` — the 2D `PlanarGeometry.update_all` core.
pub struct PlanarGeometryOut {
    pub dcoords: Vec<f64>,  // (n_edge, 2)  trgt - srce
    pub length: Vec<f64>,   // (n_edge,)
    pub centroid: Vec<f64>, // (n_face, 2)  mean of srce over the face
    pub rcoords: Vec<f64>,  // (n_edge, 2)  srce - centroid
    pub nz: Vec<f64>,       // (n_edge,)    2D cross(r, d) = rx*dy - ry*dx (signed)
    pub sub_area: Vec<f64>, // (n_edge,)    nz / 2 (signed, matches tyssue)
    pub area: Vec<f64>,     // (n_face,)    Σ sub_area
    pub perimeter: Vec<f64>,// (n_face,)    Σ length
}

/// Stateless core of `PlanarGeometry.update_all` (2D) — the planar analogue of
/// `update_geometry`. The out-of-plane normal collapses to the signed scalar
/// `nz = rx*dy - ry*dx`, and `sub_area = nz/2` is **signed** (tyssue does not
/// take the absolute value in 2D). `dim` is 2; `face` is the positional face
/// index of each edge. Excludes the stale-length `ucoords` step (done in Python).
pub fn update_geometry_planar(
    pos: &[f64],
    srce: &[u32],
    trgt: &[u32],
    face: &[u32],
    n_face: usize,
) -> PlanarGeometryOut {
    let dim = 2usize;
    let ne = srce.len();
    let mut dcoords = vec![0.0f64; ne * dim];
    let mut length = vec![0.0f64; ne];
    let mut centroid = vec![0.0f64; n_face * dim];
    let mut rcoords = vec![0.0f64; ne * dim];
    let mut nz = vec![0.0f64; ne];
    let mut sub_area = vec![0.0f64; ne];
    let mut area = vec![0.0f64; n_face];
    let mut perimeter = vec![0.0f64; n_face];
    let mut face_count = vec![0u32; n_face];

    for e in 0..ne {
        let s = srce[e] as usize * dim;
        let t = trgt[e] as usize * dim;
        let f = face[e] as usize;
        let eo = e * dim;
        let mut acc = 0.0f64;
        for d in 0..dim {
            let delta = pos[t + d] - pos[s + d];
            dcoords[eo + d] = delta;
            acc += delta * delta;
            centroid[f * dim + d] += pos[s + d];
        }
        length[e] = acc.sqrt();
        face_count[f] += 1;
    }
    for f in 0..n_face {
        let c = face_count[f];
        if c > 0 {
            let inv = 1.0 / c as f64;
            for d in 0..dim {
                centroid[f * dim + d] *= inv;
            }
        }
    }
    for e in 0..ne {
        let s = srce[e] as usize * dim;
        let f = face[e] as usize;
        let eo = e * dim;
        let rx = pos[s] - centroid[f * dim];
        let ry = pos[s + 1] - centroid[f * dim + 1];
        rcoords[eo] = rx;
        rcoords[eo + 1] = ry;
        let dx = dcoords[eo];
        let dy = dcoords[eo + 1];
        let n = rx * dy - ry * dx; // 2D cross
        nz[e] = n;
        let sa = n / 2.0;
        sub_area[e] = sa;
        area[f] += sa;
        perimeter[f] += length[e];
    }

    PlanarGeometryOut {
        dcoords,
        length,
        centroid,
        rcoords,
        nz,
        sub_area,
        area,
        perimeter,
    }
}

/// Unit-edge gradient — the shared primitive for every length/tension-family
/// effector (LineTension, PerimeterElasticity, FaceContractility,
/// LengthElasticity, BorderElasticity). Each of those reduces to
/// `grad_srce = -ucoords * c`, `grad_trgt = +ucoords * c` for a per-edge scalar
/// `c` the caller assembles from that effector's columns (the sign folds in the
/// few that flip, e.g. Border). Returns `(grad_srce, grad_trgt)`, each
/// `(n_edge, dim)` row-major (flattened).
pub fn unit_edge_gradient(ucoords: &[f64], coeff: &[f64], dim: usize) -> (Vec<f64>, Vec<f64>) {
    let ne = coeff.len();
    let mut gs = vec![0.0f64; ne * dim];
    let mut gt = vec![0.0f64; ne * dim];
    for e in 0..ne {
        let c = coeff[e];
        let eo = e * dim;
        for d in 0..dim {
            let v = ucoords[eo + d] * c;
            gs[eo + d] = -v;
            gt[eo + d] = v;
        }
    }
    (gs, gt)
}

/// Area gradient (3D) — the shared primitive for the area-family effectors
/// (FaceAreaElasticity, SurfaceTension, CellAreaElasticity). Reproduces tyssue's
/// `sheet_gradients.area_grad` scaled by the caller's per-edge coefficient
/// `coeff` (the effector's `ka_a0` upcast to edges):
///   `inv_area = 1/(4*sub_area)` (0 where `sub_area == 0`)
///   `grad_srce = coeff * inv_area * cross(r_aj, normal)`
///   `grad_trgt = coeff * inv_area * cross(normal, r_ak)`
/// `r_ak = srce - face`, `r_aj = trgt - face`. Returns `(grad_srce, grad_trgt)`,
/// each `(n_edge, 3)` row-major.
pub fn area_gradient(
    normals: &[f64],
    r_ak: &[f64],
    r_aj: &[f64],
    sub_area: &[f64],
    coeff: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    let ne = sub_area.len();
    let mut gs = vec![0.0f64; ne * 3];
    let mut gt = vec![0.0f64; ne * 3];
    for e in 0..ne {
        let sa = sub_area[e];
        let inv = if sa != 0.0 { 1.0 / (4.0 * sa) } else { 0.0 };
        let k = coeff[e] * inv;
        let eo = e * 3;
        let (nx, ny, nz) = (normals[eo], normals[eo + 1], normals[eo + 2]);
        let (akx, aky, akz) = (r_ak[eo], r_ak[eo + 1], r_ak[eo + 2]);
        let (ajx, ajy, ajz) = (r_aj[eo], r_aj[eo + 1], r_aj[eo + 2]);
        // grad_srce = cross(r_aj, normal)
        gs[eo] = k * (ajy * nz - ajz * ny);
        gs[eo + 1] = k * (ajz * nx - ajx * nz);
        gs[eo + 2] = k * (ajx * ny - ajy * nx);
        // grad_trgt = cross(normal, r_ak)
        gt[eo] = k * (ny * akz - nz * aky);
        gt[eo + 1] = k * (nz * akx - nx * akz);
        gt[eo + 2] = k * (nx * aky - ny * akx);
    }
    (gs, gt)
}

/// Area gradient (2D planar) — the area-family primitive for `PlanarGeometry`
/// meshes, reproducing `planar_gradients.area_grad`. Here the face normal is the
/// scalar `nz` (out-of-plane) and the cross products collapse to:
///   `grad_srce = coeff * inv_area * ( r_aj_y * nz, -r_aj_x * nz)`
///   `grad_trgt = coeff * inv_area * (-r_ak_y * nz,  r_ak_x * nz)`
/// `nz` is one value per edge; positions are `(n_edge, 2)`. Returns
/// `(grad_srce, grad_trgt)`, each `(n_edge, 2)` row-major.
pub fn area_gradient_2d(
    nz: &[f64],
    r_ak: &[f64],
    r_aj: &[f64],
    sub_area: &[f64],
    coeff: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    let ne = sub_area.len();
    let mut gs = vec![0.0f64; ne * 2];
    let mut gt = vec![0.0f64; ne * 2];
    for e in 0..ne {
        let sa = sub_area[e];
        let inv = if sa != 0.0 { 1.0 / (4.0 * sa) } else { 0.0 };
        let k = coeff[e] * inv * nz[e];
        let eo = e * 2;
        gs[eo] = k * r_aj[eo + 1];
        gs[eo + 1] = -k * r_aj[eo];
        gt[eo] = -k * r_ak[eo + 1];
        gt[eo + 1] = k * r_ak[eo];
    }
    (gs, gt)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unit_edge_gradient_opposes() {
        // one 3D edge, unit vector along x, coeff 2 -> srce=-2x, trgt=+2x
        let (gs, gt) = unit_edge_gradient(&[1.0, 0.0, 0.0], &[2.0], 3);
        assert_eq!(gs, vec![-2.0, 0.0, 0.0]);
        assert_eq!(gt, vec![2.0, 0.0, 0.0]);
    }

    #[test]
    fn area_gradient_matches_manual_cross() {
        // normal +z, r_aj along +x, r_ak along +y, sub_area 0.25 -> inv_area=1
        let (gs, gt) = area_gradient(
            &[0.0, 0.0, 1.0],
            &[0.0, 1.0, 0.0],
            &[1.0, 0.0, 0.0],
            &[0.25],
            &[1.0],
        );
        // cross(r_aj=+x, n=+z) = (0*1-0*0, 0*0-1*1, 1*0-0*0) = (0,-1,0)
        assert!((gs[0] - 0.0).abs() < 1e-12 && (gs[1] + 1.0).abs() < 1e-12);
        // cross(n=+z, r_ak=+y) = (0*0-1*1, 1*0-0*0, 0*1-0*0) = (-1,0,0)
        assert!((gt[0] + 1.0).abs() < 1e-12 && (gt[1]).abs() < 1e-12);
    }

    #[test]
    fn edge_lengths_2d() {
        // two verts, one edge of length 5 (3-4-5 triangle)
        let pos = [0.0, 0.0, 3.0, 4.0];
        let got = edge_lengths(&pos, &[0], &[1], 2);
        assert!((got[0] - 5.0).abs() < 1e-12);
    }

    #[test]
    fn scatter_add_accumulates() {
        // 3 verts, dim 2; two edges both landing on vertex 1
        let values = [1.0, 2.0, 10.0, 20.0];
        let out = scatter_add(&values, &[1, 1], 3, 2);
        assert_eq!(out, vec![0.0, 0.0, 11.0, 22.0, 0.0, 0.0]);
    }

    #[test]
    fn geometry_unit_triangle() {
        // unit right triangle as one face of 3 half-edges
        let pos = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let g = update_geometry(&pos, &[0, 1, 2], &[1, 2, 0], &[0, 0, 0], 1, 3);
        assert!((g.area[0] - 0.5).abs() < 1e-12, "area={}", g.area[0]);
        assert!((g.perimeter[0] - (2.0 + 2.0f64.sqrt())).abs() < 1e-12);
        // centroid = mean of the three source vertices
        assert!((g.centroid[0] - 1.0 / 3.0).abs() < 1e-12);
        assert!((g.centroid[1] - 1.0 / 3.0).abs() < 1e-12);
        assert!((g.length[0] - 1.0).abs() < 1e-12);
    }
}
