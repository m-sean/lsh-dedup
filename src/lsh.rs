use indicatif::ProgressIterator;
use pyo3::prelude::*;
use rand::prelude::*;
use rayon::prelude::*;
use rustc_hash::FxHasher;
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};

#[derive(Clone, Debug)]
#[pyclass]
pub struct MinHash {
    pub hash_values: Vec<u32>,
    num_perm: usize,
}

impl MinHash {
    fn new(items: Vec<&str>, permutations: &Vec<(u64, u64)>) -> Self {
        let num_perm = permutations.len();
        let mut hash_values = vec![u32::MAX; num_perm];
        for item in items {
            let item_hash = calculate_hash(&item);
            for (i, &(a, b)) in permutations.iter().enumerate() {
                let hash = permute_hash(item_hash, a, b);
                hash_values[i] = hash_values[i].min(hash);
            }
        }
        MinHash {
            hash_values,
            num_perm,
        }
    }

    pub fn jaccard_similarity(&self, other: &MinHash) -> f64 {
        let equal_count = self
            .hash_values
            .par_iter()
            .zip(&other.hash_values)
            .filter(|&(&a, &b)| a == b)
            .count();
        equal_count as f64 / self.num_perm as f64
    }
}

#[pymethods]
impl MinHash {
    #[pyo3(name = "jaccard_similarity")]
    fn jaccard_similarity_py(&self, other: &MinHash) -> PyResult<f64> {
        Ok(self.jaccard_similarity(other))
    }

    fn __repr__(&self) -> String {
        format!("MinHash({:?})", &self.hash_values)
    }
}

#[derive(Clone)]
#[pyclass]
/// Locality-Sensitive Hashing using MinHash for efficient similarity search.
pub struct MinHashLSH {
    /// A table for looking up full minhashes for jaccard similarity thresholding
    pub minhash_index: HashMap<usize, MinHash>,
    /// Number of times to split the hash singature (number of banded hash tables)
    band_size: usize,
    /// Banded hash tables used to find candidates for similarity
    hash_tables: Vec<HashMap<u64, Vec<usize>>>,
}

#[pymethods]
impl MinHashLSH {
    /// Creates a new MinHashLSH instance.
    ///
    /// ## Arguments
    ///
    /// * `records` - The records to dedupe.
    /// * `num_perm` - Number of permutations to use in the MinHash algorithm.
    /// * `num_bands` - Number of times to split each hash signature in the LSH algorithm
    /// (i.e., number of hash tables).
    #[new]
    pub fn new(records: Vec<String>, num_perm: usize, num_bands: usize) -> Self {
        let mut rng = StdRng::from_entropy();
        let permutations: Vec<(u64, u64)> = (0..num_perm).map(|_| (rng.gen(), rng.gen())).collect();
        let band_size = num_perm / num_bands;
        let mut minhash_index: HashMap<usize, MinHash> = HashMap::with_capacity(records.len());
        let mut hash_tables: Vec<HashMap<u64, Vec<usize>>> = vec![HashMap::new(); num_bands];
        for (id, text) in records.iter().enumerate().progress() {
            let items = text.split_whitespace().collect();
            let minhash = MinHash::new(items, &permutations);
            minhash_index.insert(id, minhash.clone());
            for (i, table) in hash_tables.iter_mut().enumerate() {
                let start = i * band_size;
                let end = start + band_size;
                let band_hash = calculate_band_hash(&minhash.hash_values[start..end]);
                table.entry(band_hash).or_insert_with(Vec::new).push(id);
            }
        }
        MinHashLSH {
            minhash_index,
            band_size,
            hash_tables,
        }
    }

    #[pyo3(name="query", signature=(minhash, threshold=None))]
    fn query_py(&self, minhash: &MinHash, threshold: Option<f64>) -> PyResult<Vec<usize>> {
        let result = self
            .query(minhash, threshold)
            .into_iter()
            .map(|&u| u)
            .collect();
        Ok(result)
    }

    fn get_minhash_index(&self) -> HashMap<usize, MinHash> {
        self.minhash_index.clone()
    }
}

impl MinHashLSH {
    /// Query the LSH index for (potentially) similar items.
    ///
    /// ## Arguments
    ///
    /// * `minhash` - The MinHash instance to query for.
    /// * `threshold` - threshold (inclusive) for jaccard similarity to apply to query result (optional) .
    ///
    pub fn query(&self, minhash: &MinHash, threshold: Option<f64>) -> Vec<&usize> {
        let candidates: HashSet<&usize> =
            self.hash_tables
                .iter()
                .enumerate()
                .fold(HashSet::new(), |mut doc_set, (i, table)| {
                    let start = i * self.band_size;
                    let end = start + self.band_size;
                    let band_hash = calculate_band_hash(&minhash.hash_values[start..end]);
                    if let Some(docs) = table.get(&band_hash) {
                        doc_set.extend(docs);
                    }
                    doc_set
                });
        if let Some(threshold) = threshold {
            candidates
                .into_par_iter()
                .filter_map(|idx| {
                    let candidate_hash = &self.minhash_index[&idx];
                    if minhash.jaccard_similarity(candidate_hash) >= threshold {
                        Some(idx)
                    } else {
                        None
                    }
                })
                .collect()
        } else {
            candidates.into_iter().collect()
        }
    }
}

#[inline]
fn calculate_hash<T: Hash>(t: &T) -> u64 {
    let mut s = FxHasher::default();
    t.hash(&mut s);
    s.finish()
}

#[inline]
fn permute_hash(hash: u64, a: u64, b: u64) -> u32 {
    ((a.wrapping_mul(hash).wrapping_add(b)) >> 32) as u32
}

#[inline]
fn calculate_band_hash(band: &[u32]) -> u64 {
    let mut hasher = FxHasher::default();
    for &value in band {
        hasher.write_u32(value);
    }
    hasher.finish()
}
