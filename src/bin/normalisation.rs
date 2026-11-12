use ndarray::{array, Array1, Array2, Axis};

fn main() {
    // "Don't miss me too much", one embedding row per token (5 x 4)
    let x = array![
        [0.42, 0.28, 0.14, 0.01], // Don't
        [0.31, 0.53, 0.66, 0.41], // miss
        [0.89, 0.52, 0.26, 0.13], // me
        [0.02, 0.95, 0.98, 0.37], // too
        [0.10, 0.16, 0.05, 0.08], // much
    ];

    let layer_norm = LayerNorm::new(4);
    let out = layer_norm.forward(&x);

    println!("Input mean:      {:.3}", x.mean_axis(Axis(1)).unwrap());
    println!("Input variance:  {:.3}\n", x.var_axis(Axis(1), 0.0));
    println!("Output {:?}:\n{:.2}\n", out.dim(), out);
    println!("Output mean:     {:.2}", out.mean_axis(Axis(1)).unwrap());
    println!("Output variance: {:.2}", out.var_axis(Axis(1), 0.0));
}


// Layer normalisation: normalise each row (token) across its features,
// then scale and shift with trainable gamma and beta.
//     out = (x - mean) / sqrt(var + eps) * gamma + beta
struct LayerNorm {
    gamma: Array1<f32>, // (d_model,) scale, starts at 1
    beta: Array1<f32>,  // (d_model,) shift, starts at 0
    eps: f32,
}

impl LayerNorm {
    fn new(d_model: usize) -> Self {
        Self {
            gamma: Array1::ones(d_model),
            beta: Array1::zeros(d_model),
            eps: 1e-5,
        }
    }

    /// x: (seq_len, d_model) -> (seq_len, d_model)
    fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        // One mean and one variance per row; insert_axis makes them (seq, 1) so they broadcast.
        let mean = x.mean_axis(Axis(1)).unwrap().insert_axis(Axis(1));
        let var = x.var_axis(Axis(1), 0.0).insert_axis(Axis(1)); // ddof = 0: divide by N

        let normalized = (x - &mean) / (var + self.eps).mapv(f32::sqrt);
        normalized * &self.gamma + &self.beta
    }
}

