extern crate cpython;

use cpython::{PyResult, Python};
use rand::Rng;


#[allow(dead_code)]
pub fn normalize_pixel(pixel_value: f64) -> f64 {
    pixel_value / 255.0
}


pub fn sign(f: f64) -> f64 {
    if f >= 0.0 { 1_f64 } else { -1_f64 }
}


pub fn bound(f: f64) -> f64 {
    if f > 1.0 {
        1.0
    } else if f < -1.0 {
        -1.0
    } else {
        f
    }
}


pub fn create_model(_: Python, input_count_per_sample: usize) -> PyResult<Vec<f64>> {
    Ok((0..input_count_per_sample)
        .map(|_| rand::thread_rng().gen_range(-1.0, 1.0))
        .collect::<Vec<f64>>())
}


pub fn add_biais(w: &mut Vec<f64>, x_train: &mut Vec<Vec<f64>>) {
    w.insert(0, rand::thread_rng().gen_range(-1.0, 1.0));

    for i in 0..x_train.len() {
        x_train[i].insert(0, 1_f64);
    }
}


pub fn add_biais_x_train(x_train: &mut Vec<Vec<f64>>) {
    for i in 0..x_train.len() {
        x_train[i].insert(0, 1_f64);
    }
}
