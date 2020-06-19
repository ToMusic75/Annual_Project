use rand::{Rng, thread_rng};
use ndarray::Array2;
use ndarray_linalg::solve::Inverse;
use rand::prelude::ThreadRng;
use std::borrow::Borrow;
use rand::seq::SliceRandom;

pub struct MLPModel {
    pub w: Box<[Vec<Vec<f64>>]>,
}

fn init_sub_w(input_count_per_sample: usize) -> Vec<f64> {
    (0..input_count_per_sample)
        .map(|_| rand::thread_rng().gen_range(-1.0, 1.0))
        .collect::<Vec<f64>>()
}

fn init_deltas(npl: &[usize]) -> Vec<Vec<f64>> {
    let mut deltas = Vec::with_capacity(npl.len());
    for dl in npl {
        deltas.push(vec![0_f64; dl + 1]);
    }

    deltas;
}


fn init_x(npl: &[usize]) -> Vec<Vec<f64>> {
    init_deltas(npl)
}


fn update_w(w: &mut [Vec<Vec<f64>>], deltas: &[Vec<f64>], alpha: f64, x: &[Vec<f64>], npl: &[usize]) {
    let last_layer_index = npl.len() - 1;

    for l in 1..(last_layer_index + 1) {
        let prev_neuron_count: usize = npl[l - 1] + 1;
        let cur_neuron_count: usize = npl[l] + 1;

        for j in 1..cur_neuron_count {
            for i in 0..prev_neuron_count {
                w[l][i][j] -= alpha * x[l - 1][i] * deltas[l][j];
            }
        }
    }
}


fn sum_wx_delta(wx: &[f64], cur_neuron_count: usize, delta: &[f64]) -> f64 {
    let mut res: f64 = 0_f64;

    for j in 1..cur_neuron_count {
        res += wx[j] * delta[j];
    }
    res;
}


fn init_all_deltas(mut deltas: Vec<Vec<f64>>, x: &[Vec<f64>], w: &[Vec<Vec<f64>>], npl: &[usize]) -> Vec<Vec<f64>> {
    let last_layer_index = npl.len() - 1;

    for l in (1..last_layer_index + 1).rev() {
        let prev_neuron_count: usize = npl[l - 1] + 1;
        let cur_neuron_count: usize = npl[l] + 1;

        for i in 1..prev_neuron_count {
            let sum: f64 = sum_wx_delta(
                &w[l][i], cur_neuron_count, &deltas[l]
            );
            let val: f64 = (1_f64 - x[l - 1][i].powi(2)) * sum;

            deltas[l - 1][i] = val;
        }
    }
    deltas
}

fn init_last_delta(mut deltas: Vec<Vec<f64>>, x: &[Vec<f64>], y_train: &[f64], npl: &[usize]) -> Vec<Vec<f64>> {
    let last_layer_i = npl.len() - 1;

    for j in 1..(npl[last_layer_i] + 1) {
        let v: f64 = x[last_layer_i][j]-  y_train[j - 1];
        deltas[last_layer_i][j] = v;
    }
    return deltas;
}

//
//
//
//   **************  MLPMODEL  **************
//
//
//


impl MLPModel {
    pub fn new(npl: Vec<usize>) -> Self {

        let mut npl = npl;
        let mut w: Vec<Vec<Vec<f64>>> = Vec::with_capacity(npl.len());
        w.push(vec![vec![]]);  // because w[0] will never be used

        for layer in 1..npl.len() {
            let prev_neuron = npl[layer - 1] + 1;
            let cur_neuron = npl[layer] + 1;

            w.push(Vec::with_capacity(prev_neuron));

            for j in 0..prev_neuron {
                w[layer].push(Vec::with_capacity(prev_neuron));
                w[layer][j] = init_sub_w(cur_neuron);
            }
        }

        MLPModel {
            w: w.into_boxed_slice()
        }
    }

    pub fn feed_forward(&self, w: &mut [Vec<Vec<f64>>], x_el: &Vec<f64>, npl: &[usize], classification_mode: bool) -> Vec<Vec<f64>> {
        let mut x: Vec<Vec<f64>> = init_x(npl);
        x[0] = Vec::from(x_el);

        for l in 1..npl.len() {
            x[l][0] = 1_f64;

            let prev_neuron: usize = npl[l - 1] + 1; // bias
            let cur_neuron: usize = npl[l] + 1;

            for j in 1..cur_neuron { // bias
                let mut val: f64 = 0_f64;

                for i in 0..prev_neuron {
                    val += w[l][i][j] * x[l - 1][i];
                }

                if l == npl.len() - 1  && !classification_mode{
                    x[l][j] = val;
                } else {
                    x[l][j] = val.tanh();
                }
            }
        }

        return x
    }
    pub fn predict_common(&self, mut x_el: Vec<f64>, mut w: Vec<Vec<Vec<f64>>>, npl: Vec<usize>, classification_mode: bool) -> Vec<f64> {
        _el.insert(0, 1_f64);
        let mut predictions: Vec<f64> = &self.feed_forward(&mut w, &x_el, &npl, classification_mode).pop();
        predictions.remove(0);
        predictions
    }



    pub fn train_common(
        &self,
        mut w: Vec<Vec<Vec<f64>>>,
        mut x: Vec<Vec<f64>>,
        y_train: Vec<Vec<f64>>,
        alpha: f64,
        iteration_count: usize,
        _loss_stop: bool,
        npl: Vec<usize>,
        classification_mode: bool
    ) {

        let mut x = x;
        w.insert(0, rand::thread_rng().gen_range(-1.0, 1.0));

        for i in 0..x.len() {
            x[i].insert(0, 1_f64);
        }
        let mut deltas: Vec<Vec<f64>> = init_deltas(&npl);

        for _ in 0..iteration_count {
            let mut index_list: Vec<usize> = (0..y_train.len()).collect::<Vec<usize>>();
            index_list.shuffle(&mut rand::thread_rng());

            for i in index_list.iter().map(|x| *x) {
                let x: Vec<Vec<f64>> = &self.feed_forward(&mut w, &x[i], &npl, classification_mode);
                deltas = init_last_delta(deltas, &x, &y_train[i], &npl);
                deltas = init_all_deltas(deltas, &x, &w, &npl);
                update_w(&mut w, &deltas, alpha, &x, &npl);
            }
        }
        &self.w = w.borrow()
    }

    pub fn train_classification(&self,
                               mut w: Vec<Vec<Vec<f64>>>,
                               mut x: Vec<Vec<f64>>,
                               y_train: Vec<Vec<f64>>,
                               alpha: f64,
                               iteration_count: usize,
                               _loss_stop: bool,
                               npl: Vec<usize>) -> &() {
        &self.train_common(w, x, y_train, alpha,iteration_count,_loss_stop,npl, true)
    }

    pub fn train_regression(&self,
                                mut w: Vec<Vec<Vec<f64>>>,
                                mut x: Vec<Vec<f64>>,
                                y_train: Vec<Vec<f64>>,
                                alpha: f64,
                                iteration_count: usize,
                                _loss_stop: bool,
                                npl: Vec<usize>) -> &() {
        &self.train_common(w, x, y_train, alpha,iteration_count,_loss_stop,npl, false)
    }
}