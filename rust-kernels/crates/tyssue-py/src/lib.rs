//! pyo3 bindings exposing the tyssue numeric kernels as the `tyssue_kernels`
//! Python module. Zero-copy: numpy arrays are read as contiguous slices; the
//! caller passes C-contiguous float64 / uint32 arrays.

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use tyssue_core as core;

/// Per-edge Euclidean length. `pos` is (n_vert, dim); `srce`/`trgt` are
/// positional vertex indices. Returns (n_edge,) float64.
#[pyfunction]
fn edge_lengths<'py>(
    py: Python<'py>,
    pos: PyReadonlyArray2<'py, f64>,
    srce: PyReadonlyArray1<'py, u32>,
    trgt: PyReadonlyArray1<'py, u32>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let dim = pos.shape()[1];
    let pos = pos
        .as_slice()
        .map_err(|_| PyValueError::new_err("pos must be C-contiguous"))?;
    let srce = srce
        .as_slice()
        .map_err(|_| PyValueError::new_err("srce must be C-contiguous"))?;
    let trgt = trgt
        .as_slice()
        .map_err(|_| PyValueError::new_err("trgt must be C-contiguous"))?;
    Ok(core::edge_lengths(pos, srce, trgt, dim).into_pyarray_bound(py))
}

/// Scatter-add: `np.add.at(out, index, values)`. `values` is (n_edge, dim);
/// `index` is (n_edge,) positional vertex indices. Returns (n_vert*dim,)
/// float64 (reshape to (n_vert, dim) on the Python side).
#[pyfunction]
fn scatter_add<'py>(
    py: Python<'py>,
    values: PyReadonlyArray2<'py, f64>,
    index: PyReadonlyArray1<'py, u32>,
    n_vert: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let dim = values.shape()[1];
    let values = values
        .as_slice()
        .map_err(|_| PyValueError::new_err("values must be C-contiguous"))?;
    let index = index
        .as_slice()
        .map_err(|_| PyValueError::new_err("index must be C-contiguous"))?;
    Ok(core::scatter_add(values, index, n_vert, dim).into_pyarray_bound(py))
}

/// Stateless core of `SheetGeometry.update_all` in one pass. Returns a dict of
/// flat float64 arrays: `length`/`sub_area` (n_edge,), `area`/`perimeter`
/// (n_face,), and `dcoords`/`rcoords` (n_edge*3,), `normals` (n_edge*3,),
/// `centroid` (n_face*3,) — reshape the flat ones to (-1, 3) on the Python side.
///
/// `face` is the positional face index of each edge; `n_face` the face count.
#[pyfunction]
fn update_geometry<'py>(
    py: Python<'py>,
    pos: PyReadonlyArray2<'py, f64>,
    srce: PyReadonlyArray1<'py, u32>,
    trgt: PyReadonlyArray1<'py, u32>,
    face: PyReadonlyArray1<'py, u32>,
    n_face: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let dim = pos.shape()[1];
    let pos = pos
        .as_slice()
        .map_err(|_| PyValueError::new_err("pos must be C-contiguous"))?;
    let srce = srce
        .as_slice()
        .map_err(|_| PyValueError::new_err("srce must be C-contiguous"))?;
    let trgt = trgt
        .as_slice()
        .map_err(|_| PyValueError::new_err("trgt must be C-contiguous"))?;
    let face = face
        .as_slice()
        .map_err(|_| PyValueError::new_err("face must be C-contiguous"))?;
    let g = core::update_geometry(pos, srce, trgt, face, n_face, dim);

    let out = PyDict::new_bound(py);
    out.set_item("length", g.length.into_pyarray_bound(py))?;
    out.set_item("sub_area", g.sub_area.into_pyarray_bound(py))?;
    out.set_item("area", g.area.into_pyarray_bound(py))?;
    out.set_item("perimeter", g.perimeter.into_pyarray_bound(py))?;
    out.set_item("dcoords", g.dcoords.into_pyarray_bound(py))?;
    out.set_item("rcoords", g.rcoords.into_pyarray_bound(py))?;
    out.set_item("normals", g.normals.into_pyarray_bound(py))?;
    out.set_item("centroid", g.centroid.into_pyarray_bound(py))?;
    Ok(out)
}

/// Bulk / Monolayer `update_all` core in one pass. Like `update_geometry` but
/// with a length-weighted face centroid and per-cell centroids / areas /
/// volumes. `cell` is the positional cell index of each edge. Returns a dict of
/// flat float64 arrays: edge (n_edge,) `length`/`sub_area`/`sub_vol`; edge*3
/// `dcoords`/`rcoords`/`normals`; face (n_face,) `face_area`/`perimeter`; face*3
/// `face_centroid`; cell (n_cell,) `cell_area`/`cell_vol`; cell*3 `cell_centroid`.
#[pyfunction]
fn update_geometry_bulk<'py>(
    py: Python<'py>,
    pos: PyReadonlyArray2<'py, f64>,
    srce: PyReadonlyArray1<'py, u32>,
    trgt: PyReadonlyArray1<'py, u32>,
    face: PyReadonlyArray1<'py, u32>,
    cell: PyReadonlyArray1<'py, u32>,
    n_face: usize,
    n_cell: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let c = |a: &str| PyValueError::new_err(format!("{a} must be C-contiguous"));
    let g = core::update_geometry_bulk(
        pos.as_slice().map_err(|_| c("pos"))?,
        srce.as_slice().map_err(|_| c("srce"))?,
        trgt.as_slice().map_err(|_| c("trgt"))?,
        face.as_slice().map_err(|_| c("face"))?,
        cell.as_slice().map_err(|_| c("cell"))?,
        n_face,
        n_cell,
    );
    let out = PyDict::new_bound(py);
    out.set_item("dcoords", g.dcoords.into_pyarray_bound(py))?;
    out.set_item("length", g.length.into_pyarray_bound(py))?;
    out.set_item("face_centroid", g.face_centroid.into_pyarray_bound(py))?;
    out.set_item("cell_centroid", g.cell_centroid.into_pyarray_bound(py))?;
    out.set_item("rcoords", g.rcoords.into_pyarray_bound(py))?;
    out.set_item("normals", g.normals.into_pyarray_bound(py))?;
    out.set_item("sub_area", g.sub_area.into_pyarray_bound(py))?;
    out.set_item("face_area", g.face_area.into_pyarray_bound(py))?;
    out.set_item("cell_area", g.cell_area.into_pyarray_bound(py))?;
    out.set_item("sub_vol", g.sub_vol.into_pyarray_bound(py))?;
    out.set_item("cell_vol", g.cell_vol.into_pyarray_bound(py))?;
    out.set_item("perimeter", g.perimeter.into_pyarray_bound(py))?;
    Ok(out)
}

/// Fused gradient for the standard 3-effector sheet model + edge->vertex
/// assembly (all of `compute_gradient`). Takes tyssue's geometry columns as-is;
/// returns grad_i as (n_vert*3,) float64 (reshape to (-1, 3) on the Python side).
#[allow(clippy::too_many_arguments)]
#[pyfunction]
fn sheet_gradient<'py>(
    py: Python<'py>,
    ucoords: PyReadonlyArray2<'py, f64>,
    normals: PyReadonlyArray2<'py, f64>,
    sub_area: PyReadonlyArray1<'py, f64>,
    r_ak: PyReadonlyArray2<'py, f64>,
    r_aj: PyReadonlyArray2<'py, f64>,
    srce: PyReadonlyArray1<'py, u32>,
    trgt: PyReadonlyArray1<'py, u32>,
    face: PyReadonlyArray1<'py, u32>,
    line_active: PyReadonlyArray1<'py, f64>,
    gamma_face: PyReadonlyArray1<'py, f64>,
    ka_face: PyReadonlyArray1<'py, f64>,
    boundary: PyReadonlyArray1<'py, u8>,
    n_vert: usize,
    norm_factor: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let c = |a: &str| PyValueError::new_err(format!("{a} must be C-contiguous"));
    let out = core::sheet_gradient(
        ucoords.as_slice().map_err(|_| c("ucoords"))?,
        normals.as_slice().map_err(|_| c("normals"))?,
        sub_area.as_slice().map_err(|_| c("sub_area"))?,
        r_ak.as_slice().map_err(|_| c("r_ak"))?,
        r_aj.as_slice().map_err(|_| c("r_aj"))?,
        srce.as_slice().map_err(|_| c("srce"))?,
        trgt.as_slice().map_err(|_| c("trgt"))?,
        face.as_slice().map_err(|_| c("face"))?,
        line_active.as_slice().map_err(|_| c("line_active"))?,
        gamma_face.as_slice().map_err(|_| c("gamma_face"))?,
        ka_face.as_slice().map_err(|_| c("ka_face"))?,
        boundary.as_slice().map_err(|_| c("boundary"))?,
        n_vert,
        norm_factor,
    );
    Ok(out.into_pyarray_bound(py))
}

/// 2D `PlanarGeometry.update_all` core. Like `update_geometry` but `pos` is
/// (n_vert, 2); the normal is the signed scalar `nz`, and `sub_area = nz/2` is
/// signed. Returns a dict of flat float64 arrays: `length`/`nz`/`sub_area`
/// (n_edge,), `area`/`perimeter` (n_face,), `dcoords`/`rcoords` (n_edge*2,),
/// `centroid` (n_face*2,) — reshape the coord ones to (-1, 2) on the Python side.
#[pyfunction]
fn update_geometry_planar<'py>(
    py: Python<'py>,
    pos: PyReadonlyArray2<'py, f64>,
    srce: PyReadonlyArray1<'py, u32>,
    trgt: PyReadonlyArray1<'py, u32>,
    face: PyReadonlyArray1<'py, u32>,
    n_face: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let c = |a: &str| PyValueError::new_err(format!("{a} must be C-contiguous"));
    let g = core::update_geometry_planar(
        pos.as_slice().map_err(|_| c("pos"))?,
        srce.as_slice().map_err(|_| c("srce"))?,
        trgt.as_slice().map_err(|_| c("trgt"))?,
        face.as_slice().map_err(|_| c("face"))?,
        n_face,
    );
    let out = PyDict::new_bound(py);
    out.set_item("length", g.length.into_pyarray_bound(py))?;
    out.set_item("nz", g.nz.into_pyarray_bound(py))?;
    out.set_item("sub_area", g.sub_area.into_pyarray_bound(py))?;
    out.set_item("area", g.area.into_pyarray_bound(py))?;
    out.set_item("perimeter", g.perimeter.into_pyarray_bound(py))?;
    out.set_item("dcoords", g.dcoords.into_pyarray_bound(py))?;
    out.set_item("rcoords", g.rcoords.into_pyarray_bound(py))?;
    out.set_item("centroid", g.centroid.into_pyarray_bound(py))?;
    Ok(out)
}

/// Unit-edge gradient primitive (length/tension effector family). `ucoords` is
/// (n_edge, dim); `coeff` is (n_edge,). Returns a dict `{"srce", "trgt"}` of flat
/// (n_edge*dim,) float64 arrays — reshape to (-1, dim) on the Python side.
#[pyfunction]
fn unit_edge_gradient<'py>(
    py: Python<'py>,
    ucoords: PyReadonlyArray2<'py, f64>,
    coeff: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyDict>> {
    let dim = ucoords.shape()[1];
    let c = |a: &str| PyValueError::new_err(format!("{a} must be C-contiguous"));
    let (gs, gt) = core::unit_edge_gradient(
        ucoords.as_slice().map_err(|_| c("ucoords"))?,
        coeff.as_slice().map_err(|_| c("coeff"))?,
        dim,
    );
    let out = PyDict::new_bound(py);
    out.set_item("srce", gs.into_pyarray_bound(py))?;
    out.set_item("trgt", gt.into_pyarray_bound(py))?;
    Ok(out)
}

/// Area gradient primitive (3D area effector family). `normals`/`r_ak`/`r_aj` are
/// (n_edge, 3); `sub_area`/`coeff` are (n_edge,). Returns a dict `{"srce","trgt"}`
/// of flat (n_edge*3,) float64 arrays.
#[allow(clippy::too_many_arguments)]
#[pyfunction]
fn area_gradient<'py>(
    py: Python<'py>,
    normals: PyReadonlyArray2<'py, f64>,
    r_ak: PyReadonlyArray2<'py, f64>,
    r_aj: PyReadonlyArray2<'py, f64>,
    sub_area: PyReadonlyArray1<'py, f64>,
    coeff: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyDict>> {
    let c = |a: &str| PyValueError::new_err(format!("{a} must be C-contiguous"));
    let (gs, gt) = core::area_gradient(
        normals.as_slice().map_err(|_| c("normals"))?,
        r_ak.as_slice().map_err(|_| c("r_ak"))?,
        r_aj.as_slice().map_err(|_| c("r_aj"))?,
        sub_area.as_slice().map_err(|_| c("sub_area"))?,
        coeff.as_slice().map_err(|_| c("coeff"))?,
    );
    let out = PyDict::new_bound(py);
    out.set_item("srce", gs.into_pyarray_bound(py))?;
    out.set_item("trgt", gt.into_pyarray_bound(py))?;
    Ok(out)
}

/// Area gradient primitive (2D planar area effector family). `nz`/`sub_area`/
/// `coeff` are (n_edge,); `r_ak`/`r_aj` are (n_edge, 2). Returns a dict
/// `{"srce","trgt"}` of flat (n_edge*2,) float64 arrays.
#[pyfunction]
fn area_gradient_2d<'py>(
    py: Python<'py>,
    nz: PyReadonlyArray1<'py, f64>,
    r_ak: PyReadonlyArray2<'py, f64>,
    r_aj: PyReadonlyArray2<'py, f64>,
    sub_area: PyReadonlyArray1<'py, f64>,
    coeff: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyDict>> {
    let c = |a: &str| PyValueError::new_err(format!("{a} must be C-contiguous"));
    let (gs, gt) = core::area_gradient_2d(
        nz.as_slice().map_err(|_| c("nz"))?,
        r_ak.as_slice().map_err(|_| c("r_ak"))?,
        r_aj.as_slice().map_err(|_| c("r_aj"))?,
        sub_area.as_slice().map_err(|_| c("sub_area"))?,
        coeff.as_slice().map_err(|_| c("coeff"))?,
    );
    let out = PyDict::new_bound(py);
    out.set_item("srce", gs.into_pyarray_bound(py))?;
    out.set_item("trgt", gt.into_pyarray_bound(py))?;
    Ok(out)
}

#[pymodule]
fn tyssue_kernels(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__doc__", "Rust numeric kernels for the tyssue vertex-model hot loop.")?;
    m.add_function(wrap_pyfunction!(edge_lengths, m)?)?;
    m.add_function(wrap_pyfunction!(scatter_add, m)?)?;
    m.add_function(wrap_pyfunction!(update_geometry, m)?)?;
    m.add_function(wrap_pyfunction!(update_geometry_bulk, m)?)?;
    m.add_function(wrap_pyfunction!(update_geometry_planar, m)?)?;
    m.add_function(wrap_pyfunction!(sheet_gradient, m)?)?;
    m.add_function(wrap_pyfunction!(unit_edge_gradient, m)?)?;
    m.add_function(wrap_pyfunction!(area_gradient, m)?)?;
    m.add_function(wrap_pyfunction!(area_gradient_2d, m)?)?;
    Ok(())
}
