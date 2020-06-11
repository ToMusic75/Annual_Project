extern crate cpython;

use cpython::{PyResult, Python};
use crate::linear;
use crate::misc;


pub fn predict(_: Python, x_train_row: Vec<f64>, w: Vec<f64>) -> PyResult<f64> {
    let mut r: f64 = w[0];

    for i in 0..x_train_row.len() {
        r += x_train_row[i] * w[i + 1];
    }

    Ok(linear::utils::sign(r))
}


fn feed_forward(x_train_row: &[f64], w: &[f64]) -> f64 {
    let mut r: f64 = 0_f64;

    for i in 0..x_train_row.len() {
        r += x_train_row[i] * w[i];
    }

    linear::utils::sign(r)
}


// Rosenblatt
pub fn fit(
    _: Python,
    mut w: Vec<f64>,
    mut x_train: Vec<Vec<f64>>,
    y_train: Vec<f64>,
    alpha: f64,
    epochs: usize,
    loss_stop: bool,
) -> PyResult<Vec<f64>> {
    let shuffled_index_list: Vec<usize> = misc::models::generate_shuffled_index_list(y_train.len());

    linear::utils::add_biais(&mut w, &mut x_train);
    let x_train = x_train;

    for epoch in 0..epochs {
        let mut loss: f64 = 0_f64;

        for (im, k) in shuffled_index_list.iter().map(|x| *x).enumerate() {  // Why deference needed ? --> M.Vidal
            print!("Epoch {:4}/{:4}  => {:4}/{:4}\r", epoch, epochs, im, shuffled_index_list.len());

            let prediction: f64 = feed_forward(&x_train[k], &w);
            let expected_result: f64 = y_train[k];
            loss += (expected_result - prediction).powi(2);

            for i in 0..w.len() {
                let normalized_x_train_value: f64 = x_train[k][i];

                w[i] = linear::utils::bound(w[i] + alpha * (expected_result - prediction) * normalized_x_train_value);
            }
        }
        print!("Epoch {:4}/{:4} => Loss {:.4}\r\n", epoch + 1, epochs, loss / shuffled_index_list.len() as f64);
        if loss_stop && loss == 0_f64 {
            break;
        }
    }
    Ok(w)
}
