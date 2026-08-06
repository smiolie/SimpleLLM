# SimpleLLM

A simple LLM built using Python, based on the bigram language model. Implements a character-level Generative Pre-trained Transformer (GPT), starting from a freeCodeCamp [tutorial](https://www.youtube.com/watch?v=UU1WVnMk4E8) and then extended with independently researched architectural and training changes, trained on Jane Austen's *Pride and Prejudice*.

## Background
 
This project started as a follow-along of freeCodeCamp's LLM [tutorial](https://www.youtube.com/watch?v=UU1WVnMk4E8) to learn the fundamentals of the Transformer architecture. After getting the base model training, I began fine-tuning it based on my own research into training stability and activation function choice, documented below.

## Dependencies
(assuming windows): `pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126`

NOTE: built on my laptop and trained on Microsoft Azure VM (without an NVIDIA GPU), `device` parameter defaulting to `'cpu'`, experiencing slower runtimes.


## Architectural & Regularization Updates

To resolve initial overfitting issues caused by training a 3.2M parameter model on a lightweight text corpus (~700k characters), several core regularization and stability fixes were introduced:

### 1. Activation Function: Switching from ReLU to GELU
The original tutorial implementation used ReLU in the feedforward blocks. After researching activation function choice, I switched to GELU (Gaussian Error Linear Unit).
 
**Why GELU:** ReLU makes a hard, deterministic cutoff based only on an input's sign (negative inputs are zeroed, positive inputs pass through unchanged). GELU instead weights each input by roughly how likely it is to be "kept," based on the standard Gaussian CDF, so the gating is smooth and probabilistic rather than a hard switch. Hendrycks & Gimpel (2016) show that this smoother, non-monotonic behavior lets GELU match or outperform ReLU and ELU across a range of vision, NLP, and speech tasks, and GELU has since become the standard activation in Transformer models such as GPT and BERT.
 
Reference: Hendrycks, D., & Gimpel, K. (2016). *Gaussian Error Linear Units (GELUs)*. [arXiv:1606.08415](https://arxiv.org/pdf/1606.08415)


### 2. Summary of Regularization & Fixes

| Optimization | Implementation | Purpose |
| :--- | :--- | :--- |
| **Increased Dropout** | Increased dropout probability from `0.05` to `0.20` across attention heads, feedforward, and projection layers. | Prevents internal neurons from co-adapting and memorizing exact phrase sequences. |
| **Embedding Dropout** | Added `self.emb_dropout = nn.Dropout(dropout)` right after token + positional embedding summation. | Regularizes input representations before entering deep Transformer blocks. |
| **Weight Decay** | Configured `weight_decay = 1e-2` in the `AdamW` optimizer. | Applies L2 penalty to large weights to prevent overconfidence. |
| **Cosine Annealing LR** | Added `CosineAnnealingLR(optimizer, T_max=max_iters, eta_min=1e-5)`. | Smoothly decays learning rate over training, helping the model settle into sharper local minima. |
| **Generation Context Fix** | Corrected generation step to pass windowed context `index[:, -block_size:]`. | Prevents positional embedding out-of-bounds errors when auto-regressively generating long sequences. |

---

## Training Experiments & Metrics Comparison

### Comparative Training Loss Progression

| Step | Baseline Run (Overfitting) | Regularized Run (Delayed Overfitting) | Final Run (With Checkpointing) |
| :--- | :--- | :--- | :--- |
| **0** | Train: 4.550 \| Test: 4.551 | Train: 4.552 \| Test: 4.555 | *Pending* |
| **1,000** | Train: 1.114 \| Test: 1.213 | Train: 1.206 \| Test: 1.259 | *Pending* |
| **2,000** | Train: 0.915 \| Test: **1.184** (Min) | Train: 1.044 \| Test: 1.160 | *Pending* |
| **3,000** | Train: 0.747 \| Test: 1.264 | Train: 0.951 \| Test: 1.132 | *Pending* |
| **4,000** | Train: 0.592 \| Test: 1.384 | Train: 0.868 \| Test: **1.128** (Min) | *Pending* |
| **5,000** | — | Train: 0.796 \| Test: 1.139 | *Pending* |
| **6,000** | — | Train: 0.737 \| Test: 1.154 | *Pending* |

### Experiment Findings
1. **Baseline Run:** Severe overfitting occurred quickly. The lowest test loss was reached early at Step 2,000 (**1.184**), after which test loss worsened to **1.384** by step 4,000 as the model memorized training text.
2. **Regularized Run:** Higher dropout and weight decay effectively doubled the useful learning window. Lowest test loss dropped to **1.128** at Step 4,000. Mild validation drift began around Step 5,000.
3. **Model Checkpointing Strategy:** To capture optimal parameters, automated checkpointing saves `best_model.pt` whenever validation loss reaches a new minimum during training.
