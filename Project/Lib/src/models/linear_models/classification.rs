#[no_mangle]
extern fn predict_linear_classification(w: Vec<f64>, xk: Vec<f64>) -> f64 {
    let mut sum: f64 = w[0];

    for i in 0.len() {
        sum += xk[i] * w[i + 1]
    }
    return 1 if sum >= 0 else -1
}

fn feed_forward(x: &[f64], w: &[f64]) -> f64 {
    let mut r: f64 = 0_f64;

    for i in 0..x.len() {
        r += x[i] * w[i];
    }

    return r
}

pub fn random_index(l: usize) -> Vec<usize> {
    let mut l: Vec<usize> = (0..l).collect::<Vec<usize>>();
    l.shuffle(&mut rand::thread_rng());
    return l;
}

#[no_mangle]
extern fn train_rosenblatt(mut w: Vec<f64>, mut x: Vec<Vec<f64>>, y: Vec<f64>, alpha: f64, nb_iterations: i64) -> Vec<f64> {

    for j in 0..nb_iterations {
        let k: Vec<usize> = random_index(y.len())
        gxk = predict_linear_classification(w, x[k])

        for i in 0..w.len() {
            w[i + 1] += alpha * (y[k] - gxk) * x[k][i]
        }
        w[0] += alpha * (y[k] - gxk)
    }
        
}