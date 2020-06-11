#[macro_use] extern crate cpython;

use crate::linear::classification::predict as linear_classification_predict;
use crate::linear::classification::fit as linear_classification_fit;
use crate::linear::regression::fit as linear_regression_fit;
use crate::linear::regression::predict as linear_regression_predict;
use crate::linear::utils::create_model as linear_utils_create_model;

use crate::mlp::classification::predict as mlp_classification_predict;
use crate::mlp::classification::fit as mlp_classification_fit;
use crate::mlp::regression::predict as mlp_regression_predict;
use crate::mlp::regression::fit as mlp_regression_fit;
use crate::mlp::utils::create_model as mpl_utils_create_model;

use crate::misc::images::load_from_images_path as misc_images_load_from_images_path;
use crate::misc::models::load as misc_models_load;
use crate::misc::models::save as misc_models_save;

mod misc;
mod linear;
mod mlp;


py_module_initializer!(rustlib, init_rustlib, PyInit__rustlib, |py, m| {
    m.add(py, "__doc__", "This module is implemented in Rust.")?;

    // Linear
    m.add(py, "linear_utils_create_model", py_fn!(py, linear_utils_create_model(
        input_count_per_sample: usize
    )))?;
    //   Classification
    m.add(py, "linear_classification_fit", py_fn!(py, linear_classification_fit(
        w: Vec<f64>,
        x_train: Vec<Vec<f64>>,
        y_train: Vec<f64>,
        alpha: f64,
        epochs: usize,
        loss_stop: bool
    )))?;
    m.add(py, "linear_classification_predict", py_fn!(py, linear_classification_predict(
        x_train: Vec<f64>,
        w: Vec<f64>
    )))?;
    //   Regression
    m.add(py, "linear_regression_fit", py_fn!(py, linear_regression_fit(
        x_train: Vec<Vec<f64>>,
        y_train: Vec<f64>
    )))?;
    m.add(py, "linear_regression_predict", py_fn!(py, linear_regression_predict(
        x_train: Vec<f64>,
        w: Vec<f64>
    )))?;

    // Mlp
    m.add(py, "mpl_utils_create_model", py_fn!(py, mpl_utils_create_model(
        npl: Vec<usize>
    )))?;
    //   Classification
    m.add(py, "mlp_classification_fit", py_fn!(py, mlp_classification_fit(
        w: Vec<Vec<Vec<f64>>>,
        x_train: Vec<Vec<f64>>,
        y_train: Vec<Vec<f64>>,
        alpha: f64,
        epochs: usize,
        loss_stop: bool,
        npl: Vec<usize>
    )))?;
    m.add(py, "mlp_classification_predict", py_fn!(py, mlp_classification_predict(
        x_train: Vec<f64>,
        w: Vec<Vec<Vec<f64>>>,
        npl: Vec<usize>
    )))?;
    //   Regression
    m.add(py, "mlp_regression_fit", py_fn!(py, mlp_regression_fit(
        w: Vec<Vec<Vec<f64>>>,
        x_train: Vec<Vec<f64>>,
        y_train: Vec<Vec<f64>>,
        alpha: f64,
        epochs: usize,
        loss_stop: bool,
        npl: Vec<usize>
    )))?;
    m.add(py, "mlp_regression_predict", py_fn!(py, mlp_regression_predict(
        x_train: Vec<f64>,
        w: Vec<Vec<Vec<f64>>>,
        npl: Vec<usize>
    )))?;

    // Image
    m.add(py, "misc_images_load_from_images_path", py_fn!(py, misc_images_load_from_images_path(
        path: &str,
        dimensions: (usize, usize),
        y_train_value: f64
    )))?;

    // Models
    m.add(py, "misc_models_save", py_fn!(py, misc_models_save(
        path: &str,
        model: Vec<f64>
    )))?;
    m.add(py, "misc_models_load", py_fn!(py, misc_models_load(
        path: &str
    )))?;

    Ok(())
});
