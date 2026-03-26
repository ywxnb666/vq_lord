# Multimodal Eval Suite Report

## Special Benchmarks

Overall accuracy: 0.7500 (6/8)

| Benchmark | Accuracy | Correct | Total |
|---|---:|---:|---:|
| ai2d | 1.0000 | 4 | 4 |
| chartqa | 0.5000 | 2 | 4 |

## ScienceQA Controls

| Control | Accuracy | Delta vs Baseline |
|---|---:|---:|
| baseline | 0.7500 | +0.0000 |
| text_only_blank | 0.5000 | -0.2500 |
| hint_ablation | 0.7500 | +0.0000 |
| option_shuffle | 0.7500 | +0.0000 |
| random_image_swap | 0.5000 | -0.2500 |
| image_blur | 0.7500 | +0.0000 |
| image_downsample | 0.5000 | -0.2500 |

## Failure Samples

### baseline

- sample_id: `test_9`
  question: What is the expected ratio of offspring with a woolly fleece to offspring with a hairy fleece? Choose the most likely ratio.
  output: C         B    A M  S : 4:0 | 2:2

### text_only_blank

- sample_id: `test_2`
  question: What is the name of the colony shown?
  output: C   C  B A D            D  C A B
- sample_id: `test_9`
  question: What is the expected ratio of offspring with a woolly fleece to offspring with a hairy fleece? Choose the most likely ratio.
  output: A   D  B  C  A    Answer: D B: 4 C: 2

### hint_ablation

- sample_id: `test_9`
  question: What is the expected ratio of offspring with a woolly fleece to offspring with a hairy fleece? Choose the most likely ratio.
  output: C   B  C D  A    Answer: C  B C D A Answer:

### option_shuffle

- sample_id: `test_9`
  question: What is the expected ratio of offspring with a woolly fleece to offspring with a hairy fleece? Choose the most likely ratio.
  output: A   B  C  D   E         A     B

### random_image_swap

- sample_id: `test_2`
  question: What is the name of the colony shown?
  output: D   D  D  D  D  D  D  D D D D D
- sample_id: `test_9`
  question: What is the expected ratio of offspring with a woolly fleece to offspring with a hairy fleece? Choose the most likely ratio.
  output: A   D  B  C  A    Answer: D  B Answer: D \_0:

### image_blur

- sample_id: `test_9`
  question: What is the expected ratio of offspring with a woolly fleece to offspring with a hairy fleece? Choose the most likely ratio.
  output: C        B    A M  : 4:0 D : 2:2

### image_downsample

- sample_id: `test_5`
  question: Which of these organisms contains matter that was once part of the lichen?
  output: A   B   C  B   D A    Answer: B     B
- sample_id: `test_9`
  question: What is the expected ratio of offspring with a woolly fleece to offspring with a hairy fleece? Choose the most likely ratio.
  output: C        B    A M  S : 4:0 D : 2
