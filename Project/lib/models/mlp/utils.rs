extern crate cpython;

use cpython::{PyResult, Python};
use rand::Rng;



pub fn init_sub_w(input_count_per_sample: usize) -> Vec<f64> {
    (0..input_count_per_sample)
        .map(|_| rand::thread_rng().gen_range(-1.0, 1.0))
        .collect::<Vec<f64>>()
}


pub fn create_model(_: Python, npl: Vec<usize>) -> PyResult<Vec<Vec<Vec<f64>>>> {
    let mut w: Vec<Vec<Vec<f64>>> = Vec::with_capacity(npl.len());
    w.push(vec![vec![]]);  // because w[0] will never be used

    for l in 1..npl.len() {
        let prev_neuron_count = npl[l - 1] + 1; // +1 for biais
        let cur_neuron_count = npl[l] + 1;  // +1 for biais

        w.push(Vec::with_capacity(prev_neuron_count));

        for j in 0..prev_neuron_count {
            w[l].push(Vec::with_capacity(cur_neuron_count));
            w[l][j] = init_sub_w(cur_neuron_count);
        }
    }

    Ok(w)
}
