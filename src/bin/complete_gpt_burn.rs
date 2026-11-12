
use std::collections::{BTreeSet, HashMap};
use std::error::Error;
use std::fs;
use std::io::{self, Write};
use std::path::Path;

use burn::grad_clipping::GradientClippingConfig;
use burn::module::Param;
use burn::nn::loss::CrossEntropyLossConfig;
use burn::nn::{Embedding, EmbeddingConfig, Initializer, LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::optim::{AdamWConfig, GradientsParams};
use burn::prelude::*;
use burn::tensor::activation::{relu, softmax};
use clap::{Parser, Subcommand};
use rand::distr::Distribution;
use rand::distr::weighted::WeightedIndex;
use rand::{Rng, RngExt};

// Same architecture as the PyTorch script. Two differences come from Burn itself:
//  * Tensor has no backend type parameter in Burn 0.22. The backend (CPU or GPU) is chosen at
//    runtime through a `Device`, so none of the code below is generic.
//  * Burn's checkpoint format is not PyTorch's, so weights are saved next to the old `.pt`
//    file (see `CHECKPOINT_BASE`) instead of over it.

type AppResult<T> = std::result::Result<T, Box<dyn Error>>;

// ---------------------------------------------------------------------------------------
// Paths and training settings
// ---------------------------------------------------------------------------------------

const DATASET_PATH: &str = "datasets/tinyshakespeare.txt";
// Burn writes three files from this base: `.bpk` (weights), `.optim.bpk` (optimizer state)
// and `.json` (step, vocabulary and model shape). Your PyTorch `.pt` file is left untouched.
const CHECKPOINT_BASE: &str = "data/tinyshakespeare";

const BATCH_SIZE: usize = 32;
const BLOCK_SIZE: usize = 128; // context length, also the model's max_len
const BLOCKS: usize = 2;
const DIM_MODEL: usize = 64;
const N_HEADS: usize = 4;
const LEARNING_RATE: f64 = 3e-4;

fn weights_path() -> String {
    format!("{CHECKPOINT_BASE}.bpk")
}
fn optimizer_path() -> String {
    format!("{CHECKPOINT_BASE}.optim.bpk")
}
fn meta_path() -> String {
    format!("{CHECKPOINT_BASE}.json")
}

// ---------------------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------------------

/// Fixed sinusoidal encodings (no learnable parameters), shape (len, dim).
fn positional_encodings(len: usize, dim: usize, device: &Device) -> Tensor<2> {
    let mut values = vec![0.0_f32; len * dim];
    for position in 0..len {
        for i in (0..dim).step_by(2) {
            let angle = position as f32 * (-(10000.0_f32).ln() * i as f32 / dim as f32).exp();
            values[position * dim + i] = angle.sin(); // even dims
            if i + 1 < dim {
                values[position * dim + i + 1] = angle.cos(); // odd dims
            }
        }
    }
    Tensor::from_data(TensorData::new(values, [len, dim]), device)
}

/// A linear layer with the PyTorch script's init: small random weights, zero bias.
fn linear(d_input: usize, d_output: usize, bias: bool, device: &Device) -> Linear {
    let mut layer = LinearConfig::new(d_input, d_output)
        .with_bias(bias)
        .with_initializer(Initializer::Normal { mean: 0.0, std: 0.02 })
        .init(device);
    if bias {
        // Burn uses the weight initializer for the bias too, so reset it to zeros.
        layer.bias = Some(Param::from_tensor(Tensor::zeros([d_output], device)));
    }
    layer
}

#[derive(Module, Debug)]
struct MultiHeadAttention {
    query: Linear,
    key: Linear,
    value: Linear,
    proj: Linear,
    n_heads: usize,
}

impl MultiHeadAttention {
    fn new(dim_model: usize, n_heads: usize, device: &Device) -> Self {
        assert!(dim_model.is_multiple_of(n_heads), "dim_model must divide by n_heads");
        Self {
            query: linear(dim_model, dim_model, false, device),
            key: linear(dim_model, dim_model, false, device),
            value: linear(dim_model, dim_model, false, device),
            proj: linear(dim_model, dim_model, false, device),
            n_heads,
        }
    }

    fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let [batch, time, dim] = x.dims();
        let head_dim = dim / self.n_heads;

        // (B, T, D) -> (B, T, heads, head_dim) -> (B, heads, T, head_dim)
        let split_heads = |t: Tensor<3>| t.reshape([batch, time, self.n_heads, head_dim]).swap_dims(1, 2);
        let q = split_heads(self.query.forward(x.clone()));
        let k = split_heads(self.key.forward(x.clone()));
        let v = split_heads(self.value.forward(x.clone()));

        // How much each token should look at every other token: (B, heads, T, T)
        let scores = q.matmul(k.swap_dims(2, 3)).div_scalar((head_dim as f32).sqrt());

        // Causal mask: -1e9 above the diagonal, so softmax gives ~0 weight to future tokens.
        let mask = Tensor::<2, Bool>::triu_mask([time, time], 1, &x.device())
            .unsqueeze::<4>()
            .expand([batch, self.n_heads, time, time]);
        let weights = softmax(scores.mask_fill(mask, -1e9), 3);

        let out = weights.matmul(v); // (B, heads, T, head_dim)
        let out = out.swap_dims(1, 2).reshape([batch, time, dim]); // glue the heads back together
        self.proj.forward(out)
    }
}

#[derive(Module, Debug)]
struct FeedForwardNetwork {
    expand: Linear,
    contract: Linear,
}

impl FeedForwardNetwork {
    fn new(dim_model: usize, device: &Device) -> Self {
        Self {
            expand: linear(dim_model, 4 * dim_model, true, device),
            contract: linear(4 * dim_model, dim_model, true, device),
        }
    }

    fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        self.contract.forward(relu(self.expand.forward(x)))
    }
}

/// Where the layer norms sit relative to the residual connections.
#[derive(Config, Debug, Copy, PartialEq)]
enum NormPosition {
    Pre,
    Post,
}

#[derive(Module, Debug)]
struct TransformerBlock {
    attention: MultiHeadAttention,
    layernorm1: LayerNorm,
    ffn: FeedForwardNetwork,
    layernorm2: LayerNorm,
    // Plain `bool` rather than the enum, because Module fields must be tensors or primitives.
    pre_norm: bool,
}

impl TransformerBlock {
    fn new(dim_model: usize, n_heads: usize, position: NormPosition, device: &Device) -> Self {
        Self {
            attention: MultiHeadAttention::new(dim_model, n_heads, device),
            layernorm1: LayerNormConfig::new(dim_model).init(device),
            ffn: FeedForwardNetwork::new(dim_model, device),
            layernorm2: LayerNormConfig::new(dim_model).init(device),
            pre_norm: position == NormPosition::Pre,
        }
    }

    fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        if self.pre_norm {
            let x = x.clone() + self.attention.forward(self.layernorm1.forward(x)); // normalize, attend, add back
            x.clone() + self.ffn.forward(self.layernorm2.forward(x)) // normalize, think, add back
        } else {
            let x = self.layernorm1.forward(x.clone() + self.attention.forward(x));
            self.layernorm2.forward(x.clone() + self.ffn.forward(x))
        }
    }
}

/// Everything needed to build a `Gpt`. Saved next to the weights so a checkpoint can be reloaded.
#[derive(Config, Debug)]
struct GptConfig {
    vocab_size: usize,
    #[config(default = 2)]
    blocks: usize,
    #[config(default = 64)]
    dim_model: usize,
    #[config(default = 4)]
    n_heads: usize,
    #[config(default = "NormPosition::Pre")]
    ln_position: NormPosition,
    #[config(default = 256)]
    max_len: usize, // longest context the model is used with
}

impl GptConfig {
    fn init(&self, device: &Device) -> Gpt {
        let normal = Initializer::Normal { mean: 0.0, std: 0.02 };
        Gpt {
            embedding: EmbeddingConfig::new(self.vocab_size, self.dim_model)
                .with_initializer(normal)
                .init(device),
            blocks: (0..self.blocks)
                .map(|_| TransformerBlock::new(self.dim_model, self.n_heads, self.ln_position, device))
                .collect(),
            final_layernorm: LayerNormConfig::new(self.dim_model).init(device),
            final_linear: linear(self.dim_model, self.vocab_size, true, device),
            dim_model: self.dim_model,
            max_len: self.max_len,
        }
    }
}

#[derive(Module, Debug)]
struct Gpt {
    embedding: Embedding,
    blocks: Vec<TransformerBlock>,
    final_layernorm: LayerNorm,
    final_linear: Linear,
    dim_model: usize,
    max_len: usize,
}

impl Gpt {
    /// token ids (B, T) -> logits (B, T, vocab)
    fn forward(&self, tokens: Tensor<2, Int>) -> Tensor<3> {
        let [_, time] = tokens.dims();
        let device = tokens.device();

        let x = self.embedding.forward(tokens); // (B, T, D)
        let x = x + positional_encodings(time, self.dim_model, &device).unsqueeze::<3>();
        let x = self.blocks.iter().fold(x, |x, block| block.forward(x));
        self.final_linear.forward(self.final_layernorm.forward(x))
    }

    /// Average cross-entropy of predicting `targets` from `tokens`.
    fn loss(&self, tokens: Tensor<2, Int>, targets: Tensor<2, Int>) -> Tensor<1> {
        let [batch, time] = targets.dims();
        let logits = self.forward(tokens);
        let vocab = logits.dims()[2];

        // cross entropy applies the stable softmax + log + average for us
        let loss_fn = CrossEntropyLossConfig::new().init(&logits.device());
        loss_fn.forward(logits.reshape([batch * time, vocab]), targets.reshape([batch * time]))
    }

    /// Pick the next token id after `context`, sampling from the softmax of the logits.
    fn sample_next(&self, context: &[usize], temperature: f64, rng: &mut impl Rng) -> usize {
        let device = self.embedding.weight.device();
        let ids: Vec<i64> = context.iter().map(|&id| id as i64).collect();
        let tokens = Tensor::<2, Int>::from_data(TensorData::new(ids, [1, context.len()]), &device);

        let logits = self.forward(tokens);
        let [_, time, vocab] = logits.dims();
        // Only the last position predicts the next token.
        let last = logits.narrow(1, time - 1, 1).reshape([vocab]);
        let probabilities: Vec<f32> = softmax(last.div_scalar(temperature), 0)
            .into_data()
            .try_to_vec::<f32>()
            .expect("probabilities are f32");

        WeightedIndex::new(&probabilities)
            .expect("probabilities are positive")
            .sample(rng)
    }
}

// ---------------------------------------------------------------------------------------
// Tokenizer and data
// ---------------------------------------------------------------------------------------

/// Maps each character in the training text to a small integer ID.
struct CharacterTokenizer {
    chars: Vec<char>, // sorted, so the id of a character is its index
    stoi: HashMap<char, usize>,
}

impl CharacterTokenizer {
    fn from_text(text: &str) -> Self {
        let unique: BTreeSet<char> = text.chars().collect(); // a BTreeSet iterates in sorted order
        Self::from_chars(unique.into_iter().collect())
    }

    fn from_chars(chars: Vec<char>) -> Self {
        let stoi = chars.iter().enumerate().map(|(i, &c)| (c, i)).collect();
        Self { chars, stoi }
    }

    fn encode(&self, text: &str) -> AppResult<Vec<usize>> {
        let unknown: BTreeSet<char> = text.chars().filter(|c| !self.stoi.contains_key(c)).collect();
        if !unknown.is_empty() {
            return Err(format!("Prompt contains characters not in the dataset: {unknown:?}").into());
        }
        Ok(text.chars().map(|c| self.stoi[&c]).collect())
    }

    fn decode_one(&self, token: usize) -> char {
        self.chars[token]
    }
}

/// Pick random chunks and their one-character-shifted targets.
fn get_batch(data: &[usize], rng: &mut impl Rng, device: &Device) -> (Tensor<2, Int>, Tensor<2, Int>) {
    let mut inputs = Vec::with_capacity(BATCH_SIZE * BLOCK_SIZE);
    let mut targets = Vec::with_capacity(BATCH_SIZE * BLOCK_SIZE);
    for _ in 0..BATCH_SIZE {
        let start = rng.random_range(0..data.len() - BLOCK_SIZE);
        inputs.extend(data[start..start + BLOCK_SIZE].iter().map(|&t| t as i64));
        targets.extend(data[start + 1..start + BLOCK_SIZE + 1].iter().map(|&t| t as i64));
    }
    let shape = [BATCH_SIZE, BLOCK_SIZE];
    (
        Tensor::from_data(TensorData::new(inputs, shape), device),
        Tensor::from_data(TensorData::new(targets, shape), device),
    )
}

// ---------------------------------------------------------------------------------------
// Checkpoint metadata
// ---------------------------------------------------------------------------------------

/// What the PyTorch checkpoint stored besides tensors: step, vocabulary and model config.
#[derive(Config, Debug)]
struct CheckpointMeta {
    step: usize,
    chars: String,
    model: GptConfig,
}

// ---------------------------------------------------------------------------------------
// Training
// ---------------------------------------------------------------------------------------

/// Train for `steps` more steps, continuing a saved checkpoint if there is one.
fn train_tiny_shakespeare(steps: usize) -> AppResult<()> {
    // `Device::default()` picks a GPU when built with `--features gpu`, otherwise the CPU.
    // `.autodiff()` makes tensors on it record the graph that `backward()` walks.
    let device = Device::default().autodiff();

    let text = fs::read_to_string(DATASET_PATH)?;
    let tokenizer = CharacterTokenizer::from_text(&text);
    let encoded = tokenizer.encode(&text)?;
    let (train_data, validation_data) = encoded.split_at((0.9 * encoded.len() as f64) as usize);

    let model_config = GptConfig::new(tokenizer.chars.len())
        .with_blocks(BLOCKS)
        .with_dim_model(DIM_MODEL)
        .with_n_heads(N_HEADS)
        .with_max_len(BLOCK_SIZE);
    let mut model = model_config.init(&device);
    // AdamW's weight decay (0.01) and epsilon (1e-8) set to PyTorch's defaults. Burn clips
    // the norm of each parameter's gradient separately; PyTorch's clip_grad_norm_ clips the
    // norm over all gradients together.
    let mut optimizer = AdamWConfig::new()
        .with_weight_decay(0.01)
        .with_epsilon(1e-8)
        .with_grad_clipping(Some(GradientClippingConfig::Norm(1.0)))
        .init();
    let mut start_step = 0;

    if Path::new(&weights_path()).exists() {
        let saved = CheckpointMeta::load(meta_path())?;
        if saved.chars != tokenizer.chars.iter().collect::<String>() {
            return Err("checkpoint vocabulary does not match the dataset".into());
        }
        model = model.try_load_file(weights_path())?;
        if Path::new(&optimizer_path()).exists() {
            optimizer = optimizer.load(optimizer_path())?;
        }
        start_step = saved.step;
        println!("resuming from step {start_step}");
    }

    let mut rng = rand::rng();
    println!(
        "training {steps} steps on {:?}; {} characters",
        device,
        tokenizer.chars.len()
    );
    for step in 0..steps {
        let (inputs, targets) = get_batch(train_data, &mut rng, &device);
        let loss = model.loss(inputs, targets);
        let train_loss = loss.clone().into_scalar::<f32>();

        let grads = GradientsParams::from_grads(loss.backward(), &model);
        model = optimizer.step(LEARNING_RATE, model, grads);

        let completed_step = start_step + step + 1;
        if step == 0 || (step + 1) % 250 == 0 {
            // `valid()` gives a copy of the model that records no gradients.
            let (val_inputs, val_targets) = get_batch(validation_data, &mut rng, &device.clone().without_autodiff());
            let validation_loss = model.valid().loss(val_inputs, val_targets).into_scalar::<f32>();
            println!("step {completed_step:5} train loss {train_loss:.4} validation loss {validation_loss:.4}");
        }
    }

    fs::create_dir_all(Path::new(CHECKPOINT_BASE).parent().unwrap_or(Path::new(".")))?;
    model.save_file(weights_path())?;
    optimizer.save(optimizer_path())?;
    CheckpointMeta::new(start_step + steps, tokenizer.chars.iter().collect(), model_config).save(meta_path())?;
    println!("saved model to {}", weights_path());
    Ok(())
}

// ---------------------------------------------------------------------------------------
// Generation
// ---------------------------------------------------------------------------------------

/// Continue `prompt` with Shakespeare-like text, printing each completed word as it is generated.
fn generate_shakespeare(prompt: &str, max_words: usize, temperature: f64) -> AppResult<()> {
    if max_words < 1 {
        return Err("max_words must be at least 1".into());
    }
    if temperature <= 0.0 {
        return Err("temperature must be greater than zero".into());
    }
    if prompt.is_empty() {
        return Err("prompt must not be empty".into());
    }

    // No `.autodiff()` here: generating needs no gradients.
    let device = Device::default();
    let meta = CheckpointMeta::load(meta_path())?;
    let tokenizer = CharacterTokenizer::from_chars(meta.chars.chars().collect());
    let model = meta.model.init(&device).try_load_file(weights_path())?;

    let mut tokens = tokenizer.encode(prompt)?;
    let mut result = prompt.to_string();
    let mut rng = rand::rng();
    let mut pending_word = String::new();
    let mut pending_space = false;

    print!("{prompt}");
    io::stdout().flush()?;

    while result.split_whitespace().count() < max_words {
        // never feed more than max_len tokens
        let context = &tokens[tokens.len().saturating_sub(model.max_len)..];
        let next = model.sample_next(context, temperature, &mut rng);
        tokens.push(next);
        let next_char = tokenizer.decode_one(next);
        result.push(next_char);

        if next_char.is_whitespace() {
            if !pending_word.is_empty() {
                print!("{pending_word} ");
                io::stdout().flush()?;
                pending_word.clear();
            }
            pending_space = true;
        } else {
            pending_word.push(next_char);
            pending_space = false;
        }
    }

    if !pending_word.is_empty() {
        print!("{pending_word}");
    } else if pending_space {
        print!(" ");
    }
    println!();
    Ok(())
}

// ---------------------------------------------------------------------------------------
// Command line
// ---------------------------------------------------------------------------------------

#[derive(Parser)]
#[command(about = "Train or use the Shakespeare GPT.")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Train, or continue training from the saved checkpoint
    Train {
        #[arg(long, default_value_t = 2000)]
        steps: usize,
    },
    /// Generate text from a prompt
    Generate {
        /// Asked for interactively when left out
        #[arg(long)]
        prompt: Option<String>,
    },
}

fn main() -> AppResult<()> {
    match Cli::parse().command {
        Command::Train { steps } => train_tiny_shakespeare(steps),
        Command::Generate { prompt } => {
            let prompt = match prompt {
                Some(prompt) => prompt,
                None => {
                    print!("Prompt: ");
                    io::stdout().flush()?;
                    let mut line = String::new();
                    io::stdin().read_line(&mut line)?;
                    line.trim_end_matches(['\r', '\n']).to_string()
                }
            };
            generate_shakespeare(&prompt, 200, 0.8)
        }
    }
}

// to train or continue training: cargo run --release -- train --steps 2000
//
// to generate text: cargo run --release -- generate --prompt "Behold, thine, I am of sovereign"
//
// Or omit --prompt to enter it interactively: cargo run --release -- generate