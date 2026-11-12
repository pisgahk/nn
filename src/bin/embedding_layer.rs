use ndarray::{s, Array2, Axis};
use rand::{rngs::StdRng, RngExt, SeedableRng};


fn main() {
    // "Don't miss me too much"
    let tokens = [4, 28, 432, 332, 75];
    let emb = Embedding::new(500, 4, 513347);
    let embeddings = emb.forward(&tokens);

    let pos = PositionalEncoding::new(512, 4);
    let x = pos.forward(&embeddings);

    println!("Embeddings:\n{embeddings:.4}\n");
    println!("Positional encoding (first 5 rows):\n{:.4}\n", pos.table.slice(s![..5, ..]));
    println!("Embeddings + positions {:?}:\n{x:.4}", x.dim());
}

struct Embedding {
    table: Array2<f32>, // shape: (vocab_size, dim)
}

impl Embedding {
    fn new(vocab_size: usize, dim: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let table = Array2::from_shape_fn((vocab_size, dim), |_| rng.random_range(-0.1..0.1));
        Self { table }
    }

    fn forward(&self, input: &[usize]) -> Array2<f32> {
        self.table.select(Axis(0), input)
    }
}

/// Sinusoidal positional encoding (fixed, nothing to train):
///     PE(pos, 2i)   = sin(pos / 10000^(2i/dim))
///     PE(pos, 2i+1) = cos(pos / 10000^(2i/dim))
struct PositionalEncoding {
    table: Array2<f32>, // shape: (max_len, dim)
}

impl PositionalEncoding {
    fn new(max_len: usize, dim: usize) -> Self {
        let table = Array2::from_shape_fn((max_len, dim), |(pos, col)| {
            let pair = (col / 2) as f32; // columns 0,1 -> 0   columns 2,3 -> 1 ...
            let angle = pos as f32 / 10000f32.powf(2.0 * pair / dim as f32);
            if col % 2 == 0 { angle.sin() } else { angle.cos() }
        });
        Self { table }
    }

    /// x: (seq_len, dim) -> (seq_len, dim), position info added to each row
    fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        let seq_len = x.nrows();
        x + &self.table.slice(s![..seq_len, ..])
    }
}
