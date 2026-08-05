# SimpleLLM

A simple LLM built using Python, based on the bigram language model. Implements a character-level Generative Pre-trained Transformer (GPT), starting from a freeCodeCamp [tutorial](https://www.youtube.com/watch?v=UU1WVnMk4E8) and then extended with independently researched architectural and training changes, trained on Jane Austen's *Pride and Prejudice*.

## Background
 
This project started as a follow-along of freeCodeCamp's LLM [tutorial](https://www.youtube.com/watch?v=UU1WVnMk4E8) to learn the fundamentals of the Transformer architecture. After getting the base model training, I began fine-tuning it based on my own research into training stability and activation function choice, documented below.

## Dependencies
(assuming windows): `pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126`

NOTE: built on my laptop and trained on Microsoft Azure VM (without an NVIDIA GPU), `device` parameter defaulting to `'cpu'`, experiencing slower runtimes.


## Recent Architectural Updates & Optimization Improvements (After second run)
 
To resolve overfitting issues caused by training a 3.2M parameter model on a lightweight text corpus (~700k characters), several core regularization and stability fixes were introduced:
 
### Summary of Changes
 
| Optimization | Change Implemented | Expected Impact on Performance |
| :--- | :--- | :--- |
| **Increased Dropout** | Increased dropout probability from `0.05` to `0.20` across attention heads, feedforward layers, and projection layers. | Prevents internal neurons from co-adapting and memorizing exact phrase sequences in *Pride and Prejudice*. |
| **Embedding Dropout** | Introduced `self.emb_dropout = nn.Dropout(dropout)` immediately after summing token and positional embeddings. | Regularizes input representations before entering deeper Transformer blocks, preventing early sequence memorization. |
| **Weight Decay** | Enabled `weight_decay = 1e-2` in the `AdamW` optimizer. | Applies L2 penalty to large network weights, maintaining smooth decision boundaries and preventing overconfidence. |
| **Cosine Annealing LR** | Added `CosineAnnealingLR(optimizer, T_max=max_iters, eta_min=1e-5)` updated per step. | Gradually lowers the learning rate as training progresses, enabling the model to settle into sharper local minima without exploding loss. |
| **Context Window Fix in Generation** | Corrected auto-regressive generation to pass `index_cond` (`index[:, -block_size:]`) instead of full context length. | Prevents positional embedding index out-of-bounds errors when generating sequences longer than 128 characters. |
 
### Expected Outcomes for the Next Run
1. **Lower Test Loss Gap:** The gap between training loss and test/validation loss will shrink significantly.
2. **Stable Validation Curves:** Validation loss should remain flat or continue decreasing for longer, delaying the onset of overfitting past step 2000.
3. **Better Quality Text Generation:** Generated outputs will sound more coherent and grammatically structured rather than echoing memorized text fragments.


## Activation Function: Switching from ReLU to GELU
 
The original tutorial implementation used ReLU in the feedforward blocks. After researching activation function choice, I switched to GELU (Gaussian Error Linear Unit).
 
**Why GELU:** ReLU makes a hard, deterministic cutoff based only on an input's sign (negative inputs are zeroed, positive inputs pass through unchanged). GELU instead weights each input by roughly how likely it is to be "kept," based on the standard Gaussian CDF, so the gating is smooth and probabilistic rather than a hard switch. Hendrycks & Gimpel (2016) show that this smoother, non-monotonic behavior lets GELU match or outperform ReLU and ELU across a range of vision, NLP, and speech tasks, and GELU has since become the standard activation in Transformer models such as GPT and BERT.
 
Reference: Hendrycks, D., & Gimpel, K. (2016). *Gaussian Error Linear Units (GELUs)*. [arXiv:1606.08415](https://arxiv.org/pdf/1606.08415)
