//! pyo3 bindings exposing the tyssue numeric kernels as the `tyssue_kernels`
//! Python module. Zero-copy: numpy arrays are read as contiguous slices; the
//! caller passes C-contiguous float64 / uint32 arrays.

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
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

#[pymodule]
fn tyssue_kernels(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__doc__", "Rust numeric kernels for the tyssue vertex-model hot loop.")?;
    m.add_function(wrap_pyfunction!(edge_lengths, m)?)?;
    m.add_function(wrap_pyfunction!(scatter_add, m)?)?;
    Ok(())
}
