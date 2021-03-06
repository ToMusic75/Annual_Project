use std::slice::from_raw_parts;

use crate::linear_model::LinearModel;
use crate::mlp::MLPModel;

mod linear_model;
mod mlp;

#[no_mangle]
pub extern fn linear_create_model(inputs_size: usize) -> *const LinearModel {
    Box::leak(Box::new(LinearModel::new(inputs_size)))
}

#[no_mangle]
pub extern fn mlp_create_model(npl: Vec<usize>) -> *const MLPModel {
    Box::leak(Box::new( MLPModel::new(npl)));
}

#[no_mangle]
pub extern fn linear_dispose_model(model_ptr: *mut LinearModel) {
    unsafe {
        let model = model_ptr.as_mut().unwrap();
        Box::from_raw(model);
    }
}

#[no_mangle]
pub extern fn mlp_dispose_model(model_ptr: *mut MLPModel) {
    unsafe {
        let model = model_ptr.as_mut().unwrap();
        Box::from_raw(model);
    }
}

#[no_mangle]
pub extern fn linear_predict_regression(model_ptr: *const LinearModel,
                                        inputs_ptr: *mut f64,
                                        inputs_size: usize) -> f64 {
    let model;
    let inputs;
    unsafe {
        model = model_ptr.as_ref().unwrap();
        inputs = from_raw_parts(inputs_ptr, inputs_size);
    }

    model.predict_regression(inputs, inputs_size)
}

#[no_mangle]
pub extern fn mlp_predict_regression(model_ptr: *const MLPModel,
                                        inputs_ptr: *mut f64,
                                        inputs_size: usize) -> f64 {
    let model;
    let inputs;
    unsafe {
        model = model_ptr.as_ref().unwrap();
        inputs = from_raw_parts(inputs_ptr, inputs_size);
    }

    model.predict_common()
}

#[no_mangle]
pub extern fn linear_predict_model_classification(model_ptr: *const LinearModel,
                                            inputs_ptr: *mut f64,
                                            inputs_size: usize) -> f64 {
    let model;
    let inputs;
    unsafe {
        model = model_ptr.as_ref().unwrap();
        inputs = from_raw_parts(inputs_ptr, inputs_size);
    }

    model.predict_classification(inputs, inputs_size)
}

#[no_mangle]
pub extern fn mlp_predict_classification(model_ptr: *const MLPModel,
                                                  inputs_ptr: *mut f64,
                                                  inputs_size: usize) -> f64 {
    let model;
    let inputs;
    unsafe {
        model = model_ptr.as_ref().unwrap();
        inputs = from_raw_parts(inputs_ptr, inputs_size);
    }

    model.predict_common()
}


#[no_mangle]
pub extern fn linear_train_model_classification(model_ptr: *mut LinearModel,
                                          dataset_inputs_ptr: *mut f64,
                                          dataset_expected_outputs_ptr: *mut f64,
                                          dataset_samples_count: usize,
                                          dataset_sample_features_count: usize,
                                          alpha: f64,
                                          iterations_count: usize) {
    let model;
    let dataset_inputs;
    let dataset_expected_outputs;
    unsafe {
        model = model_ptr.as_mut().unwrap();
        dataset_inputs = from_raw_parts(dataset_inputs_ptr,
                                        dataset_samples_count * dataset_sample_features_count);
        dataset_expected_outputs = from_raw_parts(dataset_expected_outputs_ptr,
                                                  dataset_samples_count * 1);
    }

    model.train_classification(dataset_inputs,
                               dataset_expected_outputs,
                               dataset_samples_count,
                               dataset_sample_features_count,
                               alpha,
                               iterations_count)
}

#[no_mangle]
pub extern fn mlp_train_model_classification(model_ptr: *mut MLPModel,
                                                dataset_inputs_ptr: *mut [f64],
                                                dataset_expected_outputs_ptr: *mut [f64],
                                                dataset_samples_count: usize,
                                                dataset_sample_features_count: usize,
                                                alpha: f64,
                                                iterations_count: usize) {
    let model;
    let dataset_inputs;
    let dataset_expected_outputs;
    unsafe {
        model = model_ptr.as_mut().unwrap();
        dataset_inputs = from_raw_parts(dataset_inputs_ptr,
                                        dataset_samples_count * dataset_sample_features_count);
        dataset_expected_outputs = from_raw_parts(dataset_expected_outputs_ptr,
                                                  dataset_samples_count * 1);
    }

    model.train_classification(dataset_inputs,
                               dataset_expected_outputs,
                               dataset_samples_count,
                               dataset_sample_features_count,
                               iterations_count,
                               alpha
                               )
}

#[no_mangle]
pub extern fn linear_train_regression(model_ptr: *mut LinearModel,
                                      dataset_inputs_ptr: *mut f64,
                                      dataset_expected_outputs_ptr: *mut f64,
                                      dataset_samples_count: usize,
                                      dataset_sample_features_count: usize) {
    let model;
    let dataset_inputs;
    let dataset_expected_outputs;
    unsafe {
        model = model_ptr.as_mut().unwrap();
        dataset_inputs = from_raw_parts(dataset_inputs_ptr,
                                        dataset_samples_count * dataset_sample_features_count);
        dataset_expected_outputs = from_raw_parts(dataset_expected_outputs_ptr,
                                                  dataset_samples_count * 1);
    }

    model.train_regression(dataset_inputs,
                           dataset_expected_outputs,
                           dataset_samples_count,
                           dataset_sample_features_count)
}

#[no_mangle]
pub extern fn mlp_train_regression(model_ptr: *mut MLPModel,
                                      dataset_inputs_ptr: *mut [f64],
                                      dataset_expected_outputs_ptr: *mut [f64],
                                      dataset_samples_count: usize,
                                      dataset_sample_features_count: usize,
                                      iteration_count: usize, alpha: f64) {
    let model;
    let dataset_inputs;
    let dataset_expected_outputs;
    unsafe {
        model = model_ptr.as_mut().unwrap();
        dataset_inputs = from_raw_parts(dataset_inputs_ptr,
                                        dataset_samples_count * dataset_sample_features_count);
        dataset_expected_outputs = from_raw_parts(dataset_expected_outputs_ptr,
                                                  dataset_samples_count * 1);
    }

    model.train_regression(dataset_inputs,
                           dataset_expected_outputs,
                           dataset_samples_count,
                           dataset_sample_features_count,
                           iteration_count,
                           alpha)
}

