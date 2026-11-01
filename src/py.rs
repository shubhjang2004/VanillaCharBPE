use std::collections::{HashMap, HashSet};

use pyo3::exceptions::{PyKeyError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use crate::script::{
    MARKER_BASE, N_FIXED_TOKENS, N_MARKERS, N_SHARED, SCRIPT_NAMES, SHARED_BASE, SPECIFIC_BASE, VARIANT,
};
use crate::{train, Rank, DEFAULT_PATTERN};

fn decode_err(e: crate::DecodeKeyError) -> PyErr {
    PyKeyError::new_err(e.to_string())
}

#[pyclass(module = "_core", name = "CoreBPE", frozen)]
struct PyCoreBPE {
    inner: crate::CoreBPE,
}

#[pymethods]
impl PyCoreBPE {
    #[new]
    fn new(
        pattern: &str,
        specific_units: Vec<u32>,
        merges: Vec<(Rank, Rank)>,
        special_tokens: HashMap<String, Rank>,
    ) -> PyResult<Self> {
        let special_tokens = special_tokens.into_iter().collect();
        crate::CoreBPE::new(pattern, &specific_units, &merges, special_tokens)
            .map(|inner| Self { inner })
            .map_err(PyValueError::new_err)
    }

    fn encode_ordinary(&self, py: Python<'_>, text: &str) -> Vec<Rank> {
        py.allow_threads(|| self.inner.encode_ordinary(text))
    }

    fn encode(&self, py: Python<'_>, text: &str, allowed_special: HashSet<String>) -> Vec<Rank> {
        py.allow_threads(|| {
            let allowed = allowed_special.iter().map(String::as_str).collect();
            self.inner.encode(text, &allowed)
        })
    }

    fn encode_ordinary_batch(&self, py: Python<'_>, texts: Vec<String>) -> Vec<Vec<Rank>> {
        py.allow_threads(|| self.inner.encode_ordinary_batch(&texts))
    }

    fn encode_batch(
        &self,
        py: Python<'_>,
        texts: Vec<String>,
        allowed_special: HashSet<String>,
    ) -> Vec<Vec<Rank>> {
        py.allow_threads(|| {
            let allowed = allowed_special.iter().map(String::as_str).collect();
            self.inner.encode_batch(&texts, &allowed)
        })
    }

    fn decode_bytes<'py>(&self, py: Python<'py>, tokens: Vec<Rank>) -> PyResult<Bound<'py, PyBytes>> {
        let bytes = py
            .allow_threads(|| self.inner.decode_bytes(&tokens))
            .map_err(decode_err)?;
        Ok(PyBytes::new_bound(py, &bytes))
    }

    fn decode_batch_bytes<'py>(
        &self,
        py: Python<'py>,
        batch: Vec<Vec<Rank>>,
    ) -> PyResult<Vec<Bound<'py, PyBytes>>> {
        let out = py
            .allow_threads(|| self.inner.decode_batch_bytes(&batch))
            .map_err(decode_err)?;
        Ok(out.iter().map(|b| PyBytes::new_bound(py, b)).collect())
    }

    #[pyo3(signature = (token, script = 0))]
    fn decode_single_token_bytes<'py>(
        &self,
        py: Python<'py>,
        token: Rank,
        script: usize,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let bytes = self
            .inner
            .decode_single_token_bytes(token, script)
            .map_err(decode_err)?;
        Ok(PyBytes::new_bound(py, &bytes))
    }

    fn token_units(&self, token: Rank) -> Option<Vec<u32>> {
        self.inner.token_units(token).map(<[u32]>::to_vec)
    }

    fn n_vocab(&self) -> usize {
        self.inner.n_vocab()
    }

    fn special_tokens(&self) -> HashMap<String, Rank> {
        self.inner.special_tokens().iter().map(|(k, &v)| (k.clone(), v)).collect()
    }
}

#[pyclass(module = "_core", name = "Trainer")]
struct PyTrainer {
    inner: train::Trainer,
}

#[pymethods]
impl PyTrainer {
    #[new]
    fn new(pattern: &str) -> PyResult<Self> {
        train::Trainer::new(pattern)
            .map(|inner| Self { inner })
            .map_err(PyValueError::new_err)
    }

    fn feed(&mut self, py: Python<'_>, texts: Vec<String>) {
        py.allow_threads(|| self.inner.feed(&texts))
    }

    fn n_unique_pieces(&self) -> usize {
        self.inner.n_unique_pieces()
    }

    /// Returns `(specific_units, merges)`. `progress(done, total)` is called
    /// roughly every 1000 merges if given.
    #[pyo3(signature = (n_ordinary, min_frequency = 2, min_unit_frequency = 10, progress = None))]
    fn finish(
        &self,
        py: Python<'_>,
        n_ordinary: usize,
        min_frequency: u64,
        min_unit_frequency: u64,
        progress: Option<PyObject>,
    ) -> PyResult<(Vec<u32>, Vec<(Rank, Rank)>)> {
        let mut cb_err: Option<PyErr> = None;
        let res = self.inner.finish(n_ordinary, min_frequency, min_unit_frequency, |done, total| {
            if let Some(cb) = &progress {
                if cb_err.is_none() && (done % 1000 == 0 || done == total) {
                    if let Err(e) = cb.call1(py, (done, total)) {
                        cb_err = Some(e);
                    }
                }
            }
        });
        if let Some(e) = cb_err {
            return Err(e);
        }
        Ok((res.specific_units, res.merges))
    }
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCoreBPE>()?;
    m.add_class::<PyTrainer>()?;
    m.add("DEFAULT_PATTERN", DEFAULT_PATTERN)?;
    m.add("SCRIPT_NAMES", SCRIPT_NAMES.to_vec())?;
    m.add("VARIANT", VARIANT)?;
    m.add("N_SHARED", N_SHARED)?;
    m.add("N_MARKERS", N_MARKERS)?;
    m.add("SHARED_BASE", SHARED_BASE)?;
    m.add("SPECIFIC_BASE", SPECIFIC_BASE)?;
    m.add("MARKER_BASE", MARKER_BASE)?;
    m.add("N_FIXED_TOKENS", N_FIXED_TOKENS)?;
    Ok(())
}
