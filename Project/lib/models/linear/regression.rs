extern crate cpython;
extern crate nalgebra;

use cpython::{PyResult, Python};
use nalgebra::DMatrix;

use crate::linear::utils;


pub fn predict(_: Python, x_train_row: Vec<f64>, w: Vec<f64>) -> PyResult<f64> {
    let mut r: f64 = w[0];

    for i in 0..x_train_row.len() {
        r += x_train_row[i] * w[i + 1];
    }

    Ok(r)
}


pub fn fit(_: Python, x_train: Vec<Vec<f64>>, y_train: Vec<f64>) -> PyResult<Vec<f64>> {
    let mut x_train = x_train;
    utils::add_biais_x_train(&mut x_train);
    let x_train = x_train;

    let x_train_rows = x_train.len();
    let x_train_cols = x_train[0].len();

    let flatten_x_train = x_train
        .into_iter()
        .flat_map(|x| x.into_iter())
        .collect::<Vec<f64>>();

    /*  Explanation why x_matrix needs to be transposed at creation
        x_matrix =
          ┌         ┐
          │ 1 1 1 1 │
          │ 0 1 2 3 │
          │ 0 1 2 3 │
          └         ┘

        transposed x_matrix =
          ┌       ┐
          │ 1 0 0 │
          │ 1 1 1 │
          │ 1 2 2 │
          │ 1 3 3 │
          └       ┘
    */
    let x_matrix_t = DMatrix::from_vec(x_train_cols, x_train_rows, flatten_x_train);
    let y_matrix = DMatrix::from_vec(y_train.len(), 1, y_train);
    let x_matrix = x_matrix_t.transpose();

    // f64 can have 306 zeros decimals before loosing precision and ending to 0
    let w = (
        (&x_matrix_t * x_matrix).pseudo_inverse(0.0000000000000000000000000000000000000000000000000000000000001_f64).unwrap() * x_matrix_t * y_matrix
    ).into_iter().cloned().collect::<Vec<f64>>();

    Ok(w)
}
