use rand::{Rng, thread_rng};
use ndarray::Array2;
use ndarray_linalg::solve::Inverse;
use rand::prelude::ThreadRng;

pub struct MLPModel {
    pub w: Box<[Vec<Vec<f64>>]>,
    pub x: Box<[Vec<f64>]>,
    pub nlp: Box<[i64]>
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
            nlp: nlp.into_boxed_slice()
        }
    }

    pub fn predict_common(&self, sample_inputs: Vec<f64>, classification_mode: bool) -> f64 {
        let mut x = &self.x;
        let npl = &self.npl;
        for j in 1..npl[0] + 1 {
            x[0][j] = sample_inputs[j - 1];
        }
        x[0] //to be done
    }
}