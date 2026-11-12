fn main() {
    // Implementing Softmax.
    let input_tensor = vec![2.1, 3.2, 4.0, 2.6, 1.0, 4.7];
    let probs = softmax(&input_tensor);
    println!("The probabilities are: {probs:.4?}")
    // The probabilities are: [0.0383, 0.1149, 0.2558, 0.0631, 0.0127, 0.5152]
}

pub fn softmax(input: &[f32]) -> Vec<f32> {
    let max = input.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    let exponentials = input
        .iter()
        .map(|&x| (x - max).exp())
        .collect::<Vec<f32>>();

    let sum:f32 = exponentials.iter().sum();

    exponentials.into_iter().map(|x| x / sum).collect()
}