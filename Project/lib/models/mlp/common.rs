pub fn init_deltas(npl: &[usize]) -> Vec<Vec<f64>> {
    let mut deltas = Vec::with_capacity(npl.len());
    for dl in npl {
        deltas.push(vec![0_f64; dl + 1]);
    }
    return deltas;
}


pub fn init_x(npl: &[usize]) -> Vec<Vec<f64>> {
    init_deltas(npl)
}


pub fn update_w(w: &mut [Vec<Vec<f64>>], deltas: &[Vec<f64>], alpha: f64, x: &[Vec<f64>], npl: &[usize]) {
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


pub fn sum_wx_delta(wx: &[f64], cur_neuron_count: usize, delta: &[f64]) -> f64 {
    let mut res: f64 = 0_f64;

    for j in 1..cur_neuron_count {
        res += wx[j] * delta[j];
    }
    return res;
}


pub fn init_all_deltas(mut deltas: Vec<Vec<f64>>, x: &[Vec<f64>], w: &[Vec<Vec<f64>>], npl: &[usize]) -> Vec<Vec<f64>> {
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
    return deltas;
}
