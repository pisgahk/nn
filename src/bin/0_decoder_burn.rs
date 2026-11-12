
use burn::{
    config::Config,
    module::Module,
    nn::{
        attention::{
            generate_autoregressive_mask, MhaInput, MultiHeadAttention, MultiHeadAttentionConfig,
        },
        Dropout, DropoutConfig, Embedding, EmbeddingConfig, Gelu, LayerNorm, LayerNormConfig,
        Linear, LinearConfig, PositionalEncoding, PositionalEncodingConfig,
    },
    prelude::*,
};

// ------------------------------------------------------------------
// 1. Position-Wise Feed-Forward Network (FFN)
// ------------------------------------------------------------------
#[derive(Module, Debug)]
pub struct FeedForward {
    linear1: Linear,
    linear2: Linear,
    dropout: Dropout,
    activation: Gelu,
}

impl FeedForward {
    pub fn new(d_model: usize, d_ff: usize, dropout: f64, device: &Device) -> Self {
        Self {
            linear1: LinearConfig::new(d_model, d_ff).init(device),
            linear2: LinearConfig::new(d_ff, d_model).init(device),
            dropout: DropoutConfig::new(dropout).init(),
            activation: Gelu::new(),
        }
    }

    pub fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let x = self.linear1.forward(x);
        let x = self.activation.forward(x);
        let x = self.dropout.forward(x);
        self.linear2.forward(x)
    }
}

// ------------------------------------------------------------------
// 2. Single Transformer Decoder Layer
// ------------------------------------------------------------------
#[derive(Module, Debug)]
pub struct CustomDecoderLayer {
    self_attn: MultiHeadAttention,
    cross_attn: MultiHeadAttention,
    ffn: FeedForward,
    norm1: LayerNorm,
    norm2: LayerNorm,
    norm3: LayerNorm,
    dropout: Dropout,
}

impl CustomDecoderLayer {
    pub fn new(
        d_model: usize,
        n_heads: usize,
        d_ff: usize,
        dropout: f64,
        device: &Device,
    ) -> Self {
        Self {
            self_attn: MultiHeadAttentionConfig::new(d_model, n_heads)
                .with_dropout(dropout)
                .init(device),
            cross_attn: MultiHeadAttentionConfig::new(d_model, n_heads)
                .with_dropout(dropout)
                .init(device),
            ffn: FeedForward::new(d_model, d_ff, dropout, device),
            norm1: LayerNormConfig::new(d_model).init(device),
            norm2: LayerNormConfig::new(d_model).init(device),
            norm3: LayerNormConfig::new(d_model).init(device),
            dropout: DropoutConfig::new(dropout).init(),
        }
    }

    /// `tgt`: [batch_size, tgt_seq_len, d_model]
    /// `memory`: [batch_size, src_seq_len, d_model] (output from the Encoder)
    pub fn forward(&self, mut tgt: Tensor<3>, memory: Tensor<3>) -> Tensor<3> {
        let [batch_size, tgt_seq_len, _] = tgt.dims();
        let device = tgt.device();

        // A. Masked Multi-Head Self-Attention
        let causal_mask = generate_autoregressive_mask(batch_size, tgt_seq_len, &device);
        let self_attn_input = MhaInput::self_attn(tgt.clone()).mask_attn(causal_mask);
        let self_attn_out = self.self_attn.forward(self_attn_input).context;

        // Residual connection + LayerNorm
        tgt = self.norm1.forward(tgt + self.dropout.forward(self_attn_out));

        // B. Multi-Head Cross-Attention (Query = Decoder, Key & Value = Encoder memory)
        let cross_attn_input = MhaInput::new(tgt.clone(), memory.clone(), memory);
        let cross_attn_out = self.cross_attn.forward(cross_attn_input).context;

        // Residual connection + LayerNorm
        tgt = self.norm2.forward(tgt + self.dropout.forward(cross_attn_out));

        // C. Feed-Forward Network + Residual connection + LayerNorm
        let ffn_out = self.ffn.forward(tgt.clone());
        self.norm3.forward(tgt + self.dropout.forward(ffn_out))
    }
}

// ------------------------------------------------------------------
// 3. Full Stacked Transformer Decoder + Output Projection
// ------------------------------------------------------------------
#[derive(Config, Debug)]
pub struct CustomTransformerDecoderConfig {
    pub vocab_size: usize,
    pub d_model: usize,
    pub n_heads: usize,
    pub d_ff: usize,
    pub num_layers: usize,
    pub max_seq_len: usize,
    #[config(default = 0.1)]
    pub dropout: f64,
}



impl CustomTransformerDecoderConfig {
    pub fn init(&self, device: &Device) -> CustomTransformerDecoder {
        let layers = (0..self.num_layers)
            .map(|_| {
                CustomDecoderLayer::new(
                    self.d_model,
                    self.n_heads,
                    self.d_ff,
                    self.dropout,
                    device,
                )
            })
            .collect();

        CustomTransformerDecoder {
            embedding: EmbeddingConfig::new(self.vocab_size, self.d_model).init(device),
            pos_encoding: PositionalEncodingConfig::new(self.d_model)
                .with_max_sequence_size(self.max_seq_len)
                .init(device),
            layers,
            output_projection: LinearConfig::new(self.d_model, self.vocab_size).init(device),
            dropout: DropoutConfig::new(self.dropout).init(),
            d_model: self.d_model,
        }
    }
}

#[derive(Module, Debug)]
pub struct CustomTransformerDecoder {
    embedding: Embedding,
    pos_encoding: PositionalEncoding,
    layers: Vec<CustomDecoderLayer>,
    output_projection: Linear,
    dropout: Dropout,
    d_model: usize,
}

impl CustomTransformerDecoder {
    /// `tgt_tokens`: [batch_size, tgt_seq_len] (Integer token IDs)
    /// `encoder_memory`: [batch_size, src_seq_len, d_model]
    /// Returns logits of shape: [batch_size, tgt_seq_len, vocab_size]
    pub fn forward(&self, tgt_tokens: Tensor<2, Int>, encoder_memory: Tensor<3>) -> Tensor<3> {
        // 1. Embed tokens and scale by sqrt(d_model)
        let scale = (self.d_model as f32).sqrt();
        let mut x = self.embedding.forward(tgt_tokens) * scale;

        // 2. Add sinusoidal positional encodings and apply dropout
        x = self.pos_encoding.forward(x);
        x = self.dropout.forward(x);

        // 3. Pass sequentially through all N decoder layers
        for layer in &self.layers {
            x = layer.forward(x, encoder_memory.clone());
        }

        // 4. Project hidden states to vocabulary logits
        self.output_projection.forward(x)
    }
}

fn main() {
    todo!("Do something here.")
}