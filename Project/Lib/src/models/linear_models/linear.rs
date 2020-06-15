use std::slice::{from_raw_parts, from_raw_parts_mut};

use rand::{Rng, thread_rng};

#[no_mangle]
pub extern fn linear_create_model(nb_features: usize) -> *mut f64 {
    let mut model = Vec::new();

    let mut rng = rand::thread_rng();

    for _ in 0..(nb_features + 1) {
        model.push(rng.gen_range(-1.0, 1.0));
    }

    let mut slice = model.into_boxed_slice(); // To Remove Excess Capacity

    let ptr = slice.as_mut_ptr();

    Box::leak(slice); // To prevent memory from being reclaimed

    ptr
}

#[no_mangle]
pub extern "C" fn linear_dispose_model(model: *mut f64, size: usize) {
    unsafe {
        let slice = from_raw_parts_mut(model, size + 1);
        Box::from_raw(slice); // In order to drop content
    }
}

#[no_mangle]
pub extern "C" fn linear_predict_model_regression(model_ptr: *mut f64,
                                                  inputs_ptr: *mut f64, inputs_size: usize) -> f64 {
    let model;
    let inputs;

    unsafe {
        model = from_raw_parts(model_ptr, inputs_size + 1);
        inputs = from_raw_parts(inputs_ptr, inputs_size);
    }

    linear_predict_model_regression_(inputs_size, &model, inputs)
}

fn linear_predict_model_regression_(inputs_size: usize, model: &[f64], inputs: &[f64]) -> f64 {
    let mut sum = model[0];
    for i in 0..inputs_size {
        sum += model[i + 1] * inputs[i]
    }
    sum
}


#[no_mangle]
pub extern "C" fn linear_predict_model_classification(model_ptr: *mut f64,
                                                      inputs_ptr: *mut f64, inputs_size: usize) -> f64 {
    let rslt = linear_predict_model_regression(model_ptr, inputs_ptr, inputs_size);

    if rslt >= 0.0 {
        1.0
    } else {
        -1.0
    }
}

pub extern fn linear_predict_model_classification_(inputs_size: usize, model: &[f64], inputs: &[f64]) -> f64 {
    let rslt = linear_predict_model_regression_(inputs_size, model, inputs);

    if rslt >= 0.0 {
        1.0
    } else {
        -1.0
    }
}

#[no_mangle]
pub extern "C" fn linear_train_model_classification(model_ptr: *mut f64,
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
        model = from_raw_parts_mut(model_ptr, dataset_sample_features_count + 1);
        dataset_inputs = from_raw_parts(dataset_inputs_ptr,
                                        dataset_samples_count * dataset_sample_features_count);
        dataset_expected_outputs = from_raw_parts(dataset_expected_outputs_ptr,
                                                  dataset_samples_count);
    }

    let mut rng = thread_rng();

    for _ in 0..iterations_count {
        let k = rng.gen_range(0, dataset_samples_count);
        let index_k = k * dataset_sample_features_count;

        let inputs_k = &dataset_inputs[index_k..(index_k + dataset_sample_features_count)];
        let output_k = dataset_expected_outputs[k];

        let predicted_output_k = linear_predict_model_classification_(dataset_sample_features_count,
                                                                      &model, inputs_k);

        let semi_grad = alpha * (output_k - predicted_output_k);

        for i in 0..dataset_sample_features_count {
            model[i + 1] += semi_grad * inputs_k[i];
        }
        model[0] += semi_grad * 1.0;
    }
}