extern crate cpython;

use cpython::{PyResult, Python};

use crate::mlp;


fn init_last_delta(mut deltas: Vec<Vec<f64>>, x: &[Vec<f64>], y_train: &[f64], npl: &[usize]) -> Vec<Vec<f64>> {
    let last_layer_index = npl.len() - 1;

    for j in 1..(npl[last_layer_index] + 1) {
        deltas[last_layer_index][j] = (1_f64 - x[last_layer_index][j].powi(2)) * (x[last_layer_index][j] - y_train[j - 1]);
    }
    return deltas;
}


fn feed_forward(w: &mut [Vec<Vec<f64>>], x_train_element: &[f64], npl: &[usize]) -> Vec<Vec<f64>> {
    let mut x: Vec<Vec<f64>> = mlp::common::init_x(npl);
    x[0] = x_train_element.to_vec();

    for l in 1..npl.len() {
        x[l][0] = 1_f64;

        let prev_neuron_count: usize = npl[l - 1] + 1; // +1 pour le biais
        let cur_neuron_count: usize = npl[l] + 1;

        for j in 1..cur_neuron_count { // +1 pour le biais
            let mut val: f64 = 0_f64;

            for i in 0..prev_neuron_count {
                val += w[l][i][j] * x[l - 1][i];
            }
            x[l][j] = val.tanh();
        }
    }

    return x;
}


pub fn fit(
    _: Python,
    w: Vec<Vec<Vec<f64>>>,
    x_train: Vec<Vec<f64>>,
    y_train: Vec<Vec<f64>>,
    alpha: f64,
    epochs: usize,
    _loss_stop: bool,
    npl: Vec<usize>,
) -> PyResult<Vec<Vec<Vec<f64>>>> {
    mlp::core::fit(
        w,
        x_train,
        y_train,
        alpha,
        epochs,
        _loss_stop,
        npl,
        &feed_forward,
        &init_last_delta,
    )
}


pub fn predict(_: Python, x_train_row: Vec<f64>, w: Vec<Vec<Vec<f64>>>, npl: Vec<usize>) -> PyResult<Vec<f64>> {
    mlp::core::predict(
        x_train_row,
        w,
        npl,
        &feed_forward,
    )
}
