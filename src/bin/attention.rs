use ndarray::{array, concatenate, Array2, Axis};
use rand::{rngs::StdRng, RngExt, SeedableRng};

fn main() {
    // "Don't miss me too much", one embedding row per token (5 x 4)
    let x = array![
        [0.42, 0.28, 0.14, 0.01], // Don't
        [0.31, 0.53, 0.66, 0.41], // miss
        [0.89, 0.52, 0.26, 0.13], // me
        [0.02, 0.95, 0.98, 0.37], // too
        [0.10, 0.16, 0.05, 0.08], // much
    ];

    let w_query = array![
        [-0.5, -3.0, -0.5, 1.0],
        [3.0, 3.0, 2.0, -3.0],
        [-3.0, 1.5, -2.0, 3.0],
        [0.0, -1.5, -3.0, -2.0],
    ];
    let w_key = array![
        [1.5, 1.0, 2.0, 1.0],
        [1.5, 1.0, -2.0, -2.0],
        [2.5, -3.0, 0.5, -3.0],
        [3.0, 3.0, 3.0, -2.5],
    ];
    let w_value = array![
        [1.0, 0.5, 0.0, 0.0],
        [0.0, 1.0, -0.5, 0.0],
        [0.0, 0.0, 1.0, 0.5],
        [-0.5, 0.0, 0.0, 1.0],
    ];

    // causal = false reproduces the diagram (every token sees every token)
    let head = CausalAttention::from_weights(w_query, w_key, w_value, false);
    let context = head.forward(&x);
    println!("Context (diagram weights, no mask) {:?}:\n{:.2}\n", context.dim(), context);

    // Same idea with seeded random weights and the causal mask turned on
    let head = CausalAttention::random(4, 4, 423240, true);
    println!("Context (random weights, causal) {:?}:\n{:.2}\n", head.forward(&x).dim(), head.forward(&x));

    // 2 heads of width 4 -> 8 output columns
    let mha = MultiHeadCausalSelfAttention::random(2, 4, 4, 423240);
    let out = mha.forward(&x);
    println!("Multi-head output shape: {:?} \n {:.2}", out.dim(), out);
}


/// One attention head: context = softmax(Q @ K^T / sqrt(d_k)) @ V
struct CausalAttention {
    w_query: Array2<f32>, // (d_in, d_out)
    w_key: Array2<f32>,   // (d_in, d_out)
    w_value: Array2<f32>, // (d_in, d_out)
    causal: bool,         // true = tokens cannot attend to later tokens
}

impl CausalAttention {
    /// Build a head from weights you already have (e.g. the ones in the diagram).
    fn from_weights(w_query: Array2<f32>, w_key: Array2<f32>, w_value: Array2<f32>, causal: bool) -> Self {
        Self { w_query, w_key, w_value, causal }
    }

    /// Build a head with seeded random weights in [-1, 1).
    fn random(d_in: usize, d_out: usize, seed: u64, causal: bool) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut init = || Array2::from_shape_fn((d_in, d_out), |_| rng.random_range(-1.0..1.0));
        let (w_query, w_key, w_value) = (init(), init(), init());
        Self { w_query, w_key, w_value, causal }
    }

    /// x: (seq_len, d_in) -> context: (seq_len, d_out)
    fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        let q = x.dot(&self.w_query); // (seq, d_out)
        let k = x.dot(&self.w_key);   // (seq, d_out)
        let v = x.dot(&self.w_value); // (seq, d_out)

        let d_k = k.ncols() as f32;
        let mut scores = q.dot(&k.t()) / d_k.sqrt(); // (seq, seq)

        if self.causal { // Causal Mask.
            for ((i, j), s) in scores.indexed_iter_mut() {
                if j > i {
                    *s = f32::NEG_INFINITY; // softmax turns this into 0
                }
            }
        }

        let weights = softmax_rows(&scores); // (seq, seq), each row sums to 1
        weights.dot(&v)                      // (seq, d_out)
    }
}

/// Several heads run on the same input; their outputs are joined side by side.
struct MultiHeadCausalSelfAttention {
    heads: Vec<CausalAttention>,
}

impl MultiHeadCausalSelfAttention {
    fn random(num_heads: usize, d_in: usize, d_out: usize, seed: u64) -> Self {
        let heads = (0..num_heads)
            .map(|h| CausalAttention::random(d_in, d_out, seed + h as u64, true))
            .collect();
        Self { heads }
    }

    /// x: (seq_len, d_in) -> (seq_len, num_heads * d_out)
    fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        let outs: Vec<_> = self.heads.iter().map(|h| h.forward(x)).collect();
        let views: Vec<_> = outs.iter().map(|o| o.view()).collect();
        concatenate(Axis(1), &views).unwrap()
    }
}

/// Row-wise softmax, using the same x - max(x) trick as before.
fn softmax_rows(x: &Array2<f32>) -> Array2<f32> {
    let mut out = x.clone();
    for mut row in out.rows_mut() {
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        row.mapv_inplace(|v| (v - max).exp());
        let sum = row.sum();
        row.mapv_inplace(|v| v / sum);
    }
    out
}
