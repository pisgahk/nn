fn main() {
    // For the input phrase "Don't miss me too ___". Potential words are ["really", "vaguely", "I", "much", "dash"]
    // Potential words are represented by the softmax_predictions below.
    // Calculate Cross Entropy Loss.

    let softmax_predictions = [0.10, 0.06, 0.32, 0.14, 0.17, 0.21];
    let target_probabilities =  [0., 0., 0., 1., 0.];

    let loss = cross_entropy_loss_naive(&target_probabilities, &softmax_predictions);
    println!("Cross entroy loss is: {}", loss)
}

fn cross_entropy_loss_naive( t: &[f32], s: &[f32]) -> f32 {
    -t.iter().zip(s).map(|(t_i, s_i)| t_i * s_i.ln()).sum::<f32>()
}

fn _cross_entropy_loss_idiomatic(input_tensor: &[f32]) -> f32 { // uses inputs of softmax layer.
    let max = input_tensor.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let log_sum = input_tensor.iter().map(|&xj| (xj - max).exp()).sum::<f32>().ln();
    let out = input_tensor.iter().map(|&xi| xi - max - log_sum).collect::<Vec<f32>>();
    out.into_iter().sum()
}
