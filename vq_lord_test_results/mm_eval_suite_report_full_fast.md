# Multimodal Eval Suite Report

## Special Benchmarks

Overall accuracy: 0.6582 (3678/5588)

| Benchmark | Accuracy | Correct | Total |
|---|---:|---:|---:|
| ai2d | 0.6846 | 2114 | 3088 |
| chartqa | 0.6256 | 1564 | 2500 |

## ScienceQA Controls

| Control | Accuracy | Delta vs Baseline |
|---|---:|---:|
| baseline | 0.8875 | +0.0000 |
| text_only_blank | 0.7134 | -0.1740 |
| hint_ablation | 0.8404 | -0.0471 |
| option_shuffle | 0.8949 | +0.0074 |
| random_image_swap | 0.7010 | -0.1864 |
| image_blur | 0.8245 | -0.0630 |
| image_downsample | 0.8453 | -0.0421 |

## Failure Samples

### baseline

- sample_id: `test_9`
  question: What is the expected ratio of offspring with a woolly fleece to offspring with a hairy fleece? Choose the most likely ratio.
  output: [logits_batch_mode]
- sample_id: `test_17`
  question: Think about the magnetic force between the magnets in each pair. Which of the following statements is true?
  output: [logits_batch_mode]
- sample_id: `test_79`
  question: Which country is highlighted?
  output: [logits_batch_mode]

### text_only_blank

- sample_id: `test_2`
  question: What is the name of the colony shown?
  output: [logits_batch_mode]
- sample_id: `test_9`
  question: What is the expected ratio of offspring with a woolly fleece to offspring with a hairy fleece? Choose the most likely ratio.
  output: [logits_batch_mode]
- sample_id: `test_17`
  question: Think about the magnetic force between the magnets in each pair. Which of the following statements is true?
  output: [logits_batch_mode]

### hint_ablation

- sample_id: `test_9`
  question: What is the expected ratio of offspring with a woolly fleece to offspring with a hairy fleece? Choose the most likely ratio.
  output: [logits_batch_mode]
- sample_id: `test_17`
  question: Think about the magnetic force between the magnets in each pair. Which of the following statements is true?
  output: [logits_batch_mode]
- sample_id: `test_36`
  question: Select the bird below.
  output: [logits_batch_mode]

### option_shuffle

- sample_id: `test_9`
  question: What is the expected ratio of offspring with a woolly fleece to offspring with a hairy fleece? Choose the most likely ratio.
  output: [logits_batch_mode]
- sample_id: `test_10`
  question: Which property do these three objects have in common?
  output: [logits_batch_mode]
- sample_id: `test_17`
  question: Think about the magnetic force between the magnets in each pair. Which of the following statements is true?
  output: [logits_batch_mode]

### random_image_swap

- sample_id: `test_2`
  question: What is the name of the colony shown?
  output: [logits_batch_mode]
- sample_id: `test_9`
  question: What is the expected ratio of offspring with a woolly fleece to offspring with a hairy fleece? Choose the most likely ratio.
  output: [logits_batch_mode]
- sample_id: `test_17`
  question: Think about the magnetic force between the magnets in each pair. Which of the following statements is true?
  output: [logits_batch_mode]

### image_blur

- sample_id: `test_9`
  question: What is the expected ratio of offspring with a woolly fleece to offspring with a hairy fleece? Choose the most likely ratio.
  output: [logits_batch_mode]
- sample_id: `test_17`
  question: Think about the magnetic force between the magnets in each pair. Which of the following statements is true?
  output: [logits_batch_mode]
- sample_id: `test_56`
  question: Will these magnets attract or repel each other?
  output: [logits_batch_mode]

### image_downsample

- sample_id: `test_5`
  question: Which of these organisms contains matter that was once part of the lichen?
  output: [logits_batch_mode]
- sample_id: `test_9`
  question: What is the expected ratio of offspring with a woolly fleece to offspring with a hairy fleece? Choose the most likely ratio.
  output: [logits_batch_mode]
- sample_id: `test_17`
  question: Think about the magnetic force between the magnets in each pair. Which of the following statements is true?
  output: [logits_batch_mode]
