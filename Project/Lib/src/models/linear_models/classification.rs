#[no_mangle]
extern fn predict_linear_classification(w: Vec<f64>, xk: Vec<f64>) -> f64 {
    let mut sum: f64 = w[0];

    for i in 0.len() {
        sum += xk[i] * w[i + 1]
    }
    return 1 if sum >= 0 else -1
}

#[no_mangle]
extern fn train_rosenblatt(mut w: Vec<f64>, mut x: Vec<Vec<f64>>, y: Vec<f64>, alpha: f64, nb_iterations: i64) -> Vec<f64> {
    for i in 0..nb_iterations {
        let mut l: f64 = 0_f64;
        

    }
}