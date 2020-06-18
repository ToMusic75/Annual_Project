use rand::{Rng, thread_rng};
use ndarray::Array2;
use ndarray_linalg::solve::Inverse;
use rand::prelude::ThreadRng;
use fastapprox::fast;
use std::borrow::Borrow;

pub struct MLPModel {
    pub w: Box<[Vec<Vec<f64>>]>,
    pub x: Box<[Vec<f64>]>,
    pub nlp: Box<[i64]>,
    pub L: Box<usize>,
    pub deltas: Box<[Vec<f64>]>
}

impl MLPModel {
    pub fn new(npl: Vec<i64>, input_size: usize) -> Self {

        let mut npl = npl;
        let mut L: usize = input_size - 1;
        let mut w: Vec<Vec<Vec<f64>>> = Vec::with_capacity(input_size);
        let mut rng: ThreadRng = thread_rng();

        for layer in 1..(L +1) {
          w.push(Vec::new());
            for i in 0..npl[layer - 1] + 1 {
                w[layer].push(Vec::new());

                for j in 0..npl[layer] + 1 {
                    w[layer][i].push(rng.gen_range(-1.0, 1.0)); // Push -1 or 1
                }
            }
        }

        let mut deltas: Vec<Vec<f64>> = Vec::with_capacity(input_size);
        deltas.push(Vec::new());
        for layer in 1..L + 1 {
            deltas.push(Vec::new());
            for j in 0..npl[layer] + 1 {
                deltas[layer].push(0.0);
            }
        }

        let mut x: Vec<Vec<f64>> = Vec::with_capacity(input_size);
        for layer in 0..L + 1 {
            x.push(Vec::new());
            for j in 0..npl[layer] + 1 {
                if j == 0 {
                    x[layer].push(1.0); //Bias neuron
                }
                else {
                    x[layer].push(0.0);
                }
            }
        }

        MLPModel {
            w: w.into_boxed_slice(),
            x: x.into_boxed_slice(),
            nlp: nlp.into_boxed_slice(),
            L: Box::from(L),
            deltas: deltas.into_boxed_slice()
        }
    }

    fn predict_common(&self, sample_inputs: Vec<f64>, classification_mode: bool) -> &_ {
        let mut x = &self.x;
        let npl = &self.npl;
        for j in 1..npl[0] + 1 {
            x[0][j] = sample_inputs[j - 1];
        }

        for layer in 1..&self.L + 1 {
            for j in 1..&self.nlp[layer] + 1{
                let mut res: f64 = 0.0;
                for i in 0..&self.nlp[layer - 1] + 1 {
                    res += &self.w[layer][i][j] * &self.x[layer - 1][i];
                }
                if layer != **&self.L || classification_mode {
                    res += fastapprox::fast::tanh(res as f32) as f64;
                }
                &self.x[layer][j] = res.borrow();
            }
        }
        &self.x[&self.L]
    }

    pub fn predict_classification(&self, sample_inputs: Vec<f64>) -> &Vec<f64> {
        return &self.predict_common( sample_inputs, true);
    }

    pub fn predict_regression(&self, sample_inputs: Vec<f64>) -> &Vec<f64> {
        return &self.predict_common( sample_inputs, false);
    }

    fn train_common(&self, dataset_inputs: &[[f64]], dataset_expected_outputs: &[[f64]], dataset_samples_count: usize, dataset_sample_features_count: usize, iteration_count: usize, alpha: f64, classification_mode: bool)  {
        for it in 0..iteration_count {
            let k = rng.gen_range(0, dataset_samples_count);
            &self.predict_common(dataset_inputs[k], classification_mode);

            for j in 1..&self.nlp[&self.L] + 1 {
                &self.deltas[&self.L][j] = &self.x[&self.L][j] - dataset_expected_outputs[k][j - 1];
                if classification_mode {
                    &self.deltas[&self.L][j] *= 1 - &self.x[&self.L][j] * &self.x[&self.L][j];
                }
            }

            for layer in &self.L + 1..2 {
                for i in 1..&self.nlp[layer - 1] + 1 {
                    let mut res = 0.0;

                    for j in 1..&self.nlp[layer] + 1 {
                        res += &self.w[layer][i][j] * &self.deltas[layer][j];
                    }
                    res *= (1 - &self.x[layer - 1][i] * &self.x[layer - 1][i]) as f64;
                    &self.deltas[layer - 1][i] = res.borrow();
                }
            }

            for layer in 1..&self.L + 1 {
                for i in 0..&self.nlp[layer - 1] + 1 {
                    for j in 1..&self.nlp[layer] + 1 {
                        &self.x[layer][i][j] -= alpha * *&self.x[layer - 1][i] * *&self.deltas[layer][j];
                    }
                }
            }
        }
    }

    pub fn train_classification(&self,dataset_inputs: &[[f64]], dataset_expected_outputs: &[[f64]], dataset_samples_count: usize,dataset_sample_features_count:usize, iteration_count: usize, alpha: f64) {
        &self.train_common(dataset_inputs, dataset_expected_outputs, dataset_samples_count,dataset_sample_features_count, iteration_count, alpha, true);
    }

    pub fn train_regression(&self,dataset_inputs: &[[f64]], dataset_expected_outputs: &[[f64]], dataset_samples_count: usize, dataset_sample_features_count:usize, iteration_count: usize, alpha: f64) {
        &self.train_common(dataset_inputs, dataset_expected_outputs, dataset_samples_count, dataset_sample_features_count, iteration_count, alpha, false);
    }
}