use crate::bridge::state::PyGameState;
use pyo3::prelude::*;

mod othello;
mod bridge;

#[pymodule]
pub fn othello_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyGameState>()?;
    Ok(())
}
