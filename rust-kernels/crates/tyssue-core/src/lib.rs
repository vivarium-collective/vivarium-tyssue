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

#[cfg(test)]
mod tests {
    use super::*;

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
}
