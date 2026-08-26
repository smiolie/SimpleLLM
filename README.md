# SimpleLLM

A simple LLM built using Python, based on the bigram language model. Implements a character-level Generative Pre-trained Transformer (GPT), starting from a freeCodeCamp [tutorial](https://www.youtube.com/watch?v=UU1WVnMk4E8) and then extended with independently researched architectural and training changes, trained on Jane Austen's *Pride and Prejudice*.

## Background
 
This project started as a follow-along of freeCodeCamp's LLM [tutorial](https://www.youtube.com/watch?v=UU1WVnMk4E8) to learn the fundamentals of the Transformer architecture. After getting the base model training, I began fine-tuning it based on my own research into training stability and activation function choice, documented below.

## Dependencies
(assuming windows): `pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126`

NOTE: built on my laptop and trained on Microsoft Azure VM (without an NVIDIA GPU), `device` parameter defaulting to `'cpu'`, experiencing slower runtimes.
2025/08/25 UPDATE: went home and got to use my PC with an NVIDIA, experiencing much faster training times

## GPU Migration
 
Moving training from the CPU-only Azure VM to a local NVIDIA GPU (`device = 'cuda'`) exposed a device-mismatch bug in `get_batch()`. The original implementation generated `ix` and `offsets` directly on `device`, then used them to index `train_data`/`test_data`, which had never been moved off the CPU. This worked by coincidence on CPU-only runs (everything was already on the same device) but raised an indexing error once CUDA was introduced.
 
**Fix:** `ix` and `offsets` are now created on CPU (matching where the dataset tensors live), and only the resulting batch (`x`, `y`) is moved to `device` after indexing:
 
```python
def get_batch(split):
    data_split = train_data if split == 'train' else test_data
    ix = torch.randint(len(data_split) - block_size, (batch_size,))
    offsets = torch.arange(block_size)
    x = data_split[ix.unsqueeze(1) + offsets]
    y = data_split[ix.unsqueeze(1) + offsets + 1]
    x, y = x.to(device), y.to(device)
    return x, y
```
 
This is also the more standard/efficient pattern in general, since it avoids keeping the full dataset resident in GPU memory and only transfers the small `(batch_size, block_size)` batch each step.

## Architectural & Regularization Updates

To resolve initial overfitting issues caused by training a 3.2M parameter model on a lightweight text corpus (~700k characters), several core regularization and stability fixes were introduced:

### 1. Fused FlashAttention & Single-Matrix KQV Projections
Replaced individual head loops and manual matrix multiplications with a single linear layer (`nn.Linear(n_embd, 3 * n_embd)`) to project Keys, Queries, and Values simultaneously. Combined this with PyTorch's scaled dot-product attention (`F.scaled_dot_product_attention`), leveraging fused FlashAttention kernels for faster compute and lower memory overhead during training.

### 2. Vectorized Batch Indexing
Optimized dataset sampling in `get_batch()` by replacing list comprehensions and iterative slice stacking with vectorized 2D indexing using matrix offsets (`ix.unsqueeze(1) + offsets`) directly on device memory.

### 3. Activation Function: Switching from ReLU to GELU
The original tutorial implementation used ReLU in the feedforward blocks. After researching activation function choice, I switched to GELU (Gaussian Error Linear Unit).
 
**Why GELU:** ReLU makes a hard, deterministic cutoff based only on an input's sign (negative inputs are zeroed, positive inputs pass through unchanged). GELU instead weights each input by roughly how likely it is to be "kept," based on the standard Gaussian CDF, so the gating is smooth and probabilistic rather than a hard switch. Hendrycks & Gimpel (2016) show that this smoother, non-monotonic behavior lets GELU match or outperform ReLU and ELU across a range of vision, NLP, and speech tasks, and GELU has since become the standard activation in Transformer models such as GPT and BERT.
 
Reference: Hendrycks, D., & Gimpel, K. (2016). *Gaussian Error Linear Units (GELUs)*. [arXiv:1606.08415](https://arxiv.org/pdf/1606.08415)

### 4. Gradient Clipping
Added `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` to the training loop to cap gradient magnitude and guard against destabilizing loss spikes.
 
### 5. Increased Dropout (0.20 → 0.25)
Dropout was raised further from `0.20` to `0.25` across attention, feedforward, and embedding layers to further reduce overfitting on the small (~700k character) corpus.
 

### Summary of Regularization & Fixes

| Optimization | Implementation | Purpose |
| :--- | :--- | :--- |
| **Increased Dropout** | Increased dropout probability from `0.05` to `0.20` to `0.25` across attention heads, feedforward, and projection layers. | Prevents internal neurons from co-adapting and memorizing exact phrase sequences. |
| **Embedding Dropout** | Added `self.emb_dropout = nn.Dropout(dropout)` right after token + positional embedding summation. | Regularizes input representations before entering deep Transformer blocks. |
| **Weight Decay** | Configured `weight_decay = 1e-2` in the `AdamW` optimizer. | Applies L2 penalty to large weights to prevent overconfidence. |
| **Cosine Annealing LR** | Added `CosineAnnealingLR(optimizer, T_max=max_iters, eta_min=1e-5)`. | Smoothly decays learning rate over training, helping the model settle into sharper local minima. |
| **Generation Context Fix** | Corrected generation step to pass windowed context `index[:, -block_size:]`. | Prevents positional embedding out-of-bounds errors when auto-regressively generating long sequences. |
| **FlashAttention Kernel** | Utilizes `F.scaled_dot_product_attention(..., is_causal=True)`. | Accelerates causal attention computation while reducing memory footprint. |
| **Gradient Clipping** | Added `clip_grad_norm_(model.parameters(), max_norm=1.0)`. | Bounds gradient magnitude to prevent destabilizing updates. |
| **GPU Batch Indexing Fix** | Build `ix`/`offsets` on CPU, move only the sampled batch to `device`. | Resolves CPU/GPU device-mismatch indexing error when training on CUDA. |

---

## Training Experiments & Metrics Comparison

### Comparative Training Loss Progression

| Step | Baseline Run (Overfitting) | Regularized Run (Delayed Overfitting) | Optimized FlashAttention Run | GPU Run (dropout 0.25 + grad clipping) |
| :--- | :--- | :--- | :--- | :--- |
| **0** | Train: 4.550 \| Test: 4.551 | Train: 4.552 \| Test: 4.555 | Train: 4.622 \| Test: 4.621 | Train: 4.584 \| Test: 4.586 |
| **500** | — | — | Train: 2.068 \| Test: 2.070 | Train: 1.557 \| Test: 1.571 |
| **1,000** | Train: 1.114 \| Test: 1.213 | Train: 1.206 \| Test: 1.259 | Train: 1.272 \| Test: 1.315 | Train: 1.188 \| Test: 1.241 |
| **1,500** | — | — | Train: 1.100 \| Test: 1.187 | Train: 1.067 \| Test: 1.164 |
| **2,000** | Train: 0.915 \| Test: **1.184** (Min) | Train: 1.044 \| Test: 1.160 | Train: 1.009 \| Test: 1.150 |Train: 0.991 \| Test: 1.133 |
| **2,500** | — | — | Train: 0.934 \| Test: 1.131 |Train: 0.928 \| Test: 1.117 |
| **3,000** | Train: 0.747 \| Test: 1.264 | Train: 0.951 \| Test: 1.132 | Train: 0.871 \| Test: 1.128 | Train: 0.878 \| Test: 1.115 |
| **3,500** | — | — | Train: 0.838 \| Test: **1.125** (Min) | Train: 0.849 \| Test: **1.114** (Min) |
| **4,000** | Train: 0.592 \| Test: 1.384 | Train: 0.868 \| Test: **1.128** (Min) | Final Batch Loss: **0.906** | Final Batch Loss: **0.967** |

### Experiment Findings
1. **Baseline Run:** Severe overfitting occurred quickly. The lowest test loss was reached early at Step 2,000 (**1.184**), after which test loss worsened to **1.384** by step 4,000 as the model memorized training text.
2. **Regularized Run:** Higher dropout and weight decay effectively doubled the useful learning window. Lowest test loss dropped to **1.128** at Step 4,000. Mild validation drift began around Step 5,000.
3. **Optimized FlashAttention Run:** Integrating fused attention, 8 attention heads, and vectorized sampling achieved a lower validation minimum of **1.125** at step 3,500 with significantly improved throughput.
3. **Model Checkpointing Strategy:** To capture optimal parameters, automated checkpointing saves `best_model.pt` whenever validation loss reaches a new minimum during training.
5. **GPU Run (dropout 0.25 + grad clipping):** First full run on the migrated CUDA setup, after fixing the batch-indexing device mismatch. Test loss reached a new low of **1.114** at step 3,500, the best validation result so far. The train/test gap narrowed to ~0.237 at step 3,500, down from ~0.287 in the FlashAttention run, indicating the extra dropout (0.20 → 0.25) reduced overfitting on this small corpus, at the cost of a higher final batch loss (0.967 vs. 0.906).


## Sample Text Generation

Given the prompt `"Hello! Can you see me?"`, the trained model loaded from `best_model.pt` generates coherent character-level text in Jane Austen's prose style:

```text
Hello! Can you see me? I should
      wish to believe myself and mistaken, in deprive of your being
engaged in that match,
```

**GPU run (dropout 0.25 + grad clipping):**
```text
Hello! Can you see me? And child Elizabeth
insist upon his delay, and how can help more of another. After the
point, I hop
```
