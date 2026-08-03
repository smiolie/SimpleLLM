# SimpleLLM

A simple LLM built using Python, based on the bigram language model. Uses a GPT instead of the full transformer architecture for simplicity.

## Dependencies
(assuming windows): `pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126`

NOTE: built on my laptop without an NVIDIA GPU, `device` parametering default to `'cpu'`, may experience slower runtimes.