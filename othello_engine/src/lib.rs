use pyo3::prelude::*;

/// Doubles the number
#[pyfunction]
fn double(x: i32) -> i32 {
    2 * x
}

#[pymodule]
pub fn othello_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(double, m)?)
}