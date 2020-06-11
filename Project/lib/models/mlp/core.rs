use cpython::PyResult;

use crate::linear;
use crate::mlp;
use crate::misc;


pub fn fit(
    mut w: Vec<Vec<Vec<f64>>>,
    mut x_train: Vec<Vec<f64>>,
    y_train: Vec<Vec<f64>>,
    alpha: f64,
    epochs: usize,
    _loss_stop: bool,
    npl: Vec<usize>,
    feed_forward: &dyn Fn(&mut [Vec<Vec<f64>>], &[f64], &[usize]) -> Vec<Vec<f64>>,
    init_last_deltas: &dyn Fn(Vec<Vec<f64>>, &[Vec<f64>], &[f64], &[usize]) -> Vec<Vec<f64>>,
) -> PyResult<Vec<Vec<Vec<f64>>>> {
    linear::utils::add_biais_x_train(&mut x_train);
    let x_train = x_train;

    let mut deltas: Vec<Vec<f64>> = mlp::common::init_deltas(&npl);

    for _ in 0..epochs {
        let shuffled_index_list: Vec<usize> = misc::models::generate_shuffled_index_list(y_train.len());

        for i in shuffled_index_list.iter().map(|x| *x) {
            let x: Vec<Vec<f64>> = feed_forward(&mut w, &x_train[i], &npl);

            deltas = init_last_deltas(deltas, &x, &y_train[i], &npl);
            deltas = mlp::common::init_all_deltas(deltas, &x, &w, &npl);
            mlp::common::update_w(&mut w, &deltas, alpha, &x, &npl);
        }
    }
    Ok(w)
}


pub fn predict(
    mut x_train_row: Vec<f64>,
    mut w: Vec<Vec<Vec<f64>>>,
    npl: Vec<usize>,
    feed_forward: &dyn Fn(&mut [Vec<Vec<f64>>], &[f64], &[usize]) -> Vec<Vec<f64>>
) -> PyResult<Vec<f64>> {
    x_train_row.insert(0, 1_f64);
    let mut predictions: Vec<f64> = feed_forward(&mut w, &x_train_row, &npl).pop().unwrap();
    predictions.remove(0);
    Ok(predictions)
}
