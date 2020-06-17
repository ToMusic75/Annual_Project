use rand::{Rng, thread_rng};
use ndarray::Array2;
use ndarray_linalg::solve::Inverse;

pub struct LinearModel {
    pub w: Box<[f64]>
}

impl LinearModel {
    pub fn new(inputs_size: usize) -> Self {
        let mut w = Vec::with_capacity(inputs_size + 1);
        let mut rng = thread_rng();

        for _ in 0..(inputs_size + 1) {
            w.push(rng.gen_range(-1.0, 1.0));
        }

        LinearModel {
            w: w.into_boxed_slice()
        }
    }

    pub fn predict_regression(
        &self,
        inputs: &[f64],
        inputs_size: usize,
    ) -> f64 {
        let weights = &self.w;
        let mut sum = weights[0];
        for i in 0..inputs_size {
            sum += weights[i + 1] * inputs[i];
        }
        sum
    }

    pub fn predict_classification(
        &self,
        inputs: &[f64],
        inputs_size: usize,
    ) -> f64 {
        if self.predict_regression(inputs, inputs_size) >= 0.0 {
            1.0
        } else {
            -1.0
        }
    }

    pub fn train_classification(
        &mut self,
        dataset_inputs: &[f64],
        dataset_expected_outputs: &[f64],
        dataset_samples_count: usize,
        dataset_sample_features_count: usize,
        alpha: f64,
        iterations_count: usize,
    ) {
        let mut rng = thread_rng();

        for _ in 0..iterations_count {
            let k = rng.gen_range(0, dataset_samples_count);
            let inputs_start_index = k * dataset_sample_features_count;
            let inputs_k = &dataset_inputs[inputs_start_index..(inputs_start_index + dataset_sample_features_count)];
            let expected_output_k = dataset_expected_outputs[k];
            let predicted_output_k = self.predict_classification(inputs_k, dataset_sample_features_count);

            let semi_grad = alpha * (expected_output_k - predicted_output_k);
            self.w[0] += semi_grad * 1.0;
            for i in 0..dataset_sample_features_count {
                self.w[i + 1] += semi_grad * inputs_k[i];
            }
        }
    }

    pub fn train_regression(
        &mut self,
        dataset_inputs: &[f64],
        dataset_expected_outputs: &[f64],
        dataset_samples_count: usize,
        dataset_sample_features_count: usize,
    ) {
        let mut x = Array2::<f64>::zeros((dataset_samples_count, dataset_sample_features_count + 1));
        let mut y =     Array2::<f64>::zeros((dataset_samples_count, 1));

        for k in 0..dataset_samples_count {
            x[[k, 0]]= 1.0;
            for j in 0..dataset_sample_features_count {
                x[[k, j + 1]] = dataset_inputs[k * dataset_sample_features_count + j];
            }
            y[[k, 0]] = dataset_expected_outputs[k];
        }

        let xt = x.t();
        let xtx = xt.dot(&x);
        let xtx_inv = xtx.inv().unwrap();
        let w = xtx_inv.dot(&xt).dot(&y);

        for i in 0..dataset_sample_features_count + 1 {
            self.w[i] = w[[i, 0]];
        }
    }
}