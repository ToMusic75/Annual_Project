extern crate cpython;
extern crate serde_json;

use cpython::{PyResult, Python};
use std::fs;
use std::io::Read;
use rand::seq::SliceRandom;


pub fn load(_: Python, path: &str) -> PyResult<Vec<f64>> {
    let mut file: fs::File = match fs::File::open(path) {
        Ok(v) => v,
        Err(_) => panic!("Can't open \"{}\"", path),
    };
    let mut contents = String::new();
    file.read_to_string(&mut contents).unwrap();
    Ok(serde_json::from_str(&contents).unwrap())
}


pub fn save(_: Python, path: &str, model: Vec<f64>) -> PyResult<i32> {
    let serialized = serde_json::to_string(&model).unwrap();
    fs::write(path, serialized).unwrap();
    Ok(1)
}


pub fn generate_shuffled_index_list(length: usize) -> Vec<usize> {
    let mut index_list: Vec<usize> = (0..length).collect::<Vec<usize>>();
    index_list.shuffle(&mut rand::thread_rng());
    return index_list;
}
