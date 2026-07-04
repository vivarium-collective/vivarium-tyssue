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

#[pymodule]
fn tyssue_kernels(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__doc__", "Rust numeric kernels for the tyssue vertex-model hot loop.")?;
    m.add_function(wrap_pyfunction!(edge_lengths, m)?)?;
    m.add_function(wrap_pyfunction!(scatter_add, m)?)?;
    m.add_function(wrap_pyfunction!(update_geometry, m)?)?;
    Ok(())
}
