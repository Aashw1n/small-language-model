# Small Language Model
An end-to-end implementation of a small language model + a gpt2-style tokenizer. Training runs and experimentation/ablations included.  



## The Tokenizer

A run-of-the-mill BPE tokenizer in Python, with a training script. 

The tokenizer has been trained on two datasets: 
    - TinyStories up to a 10k vocab size
    - OpenWebText up to a 30k vocab size


## The Langauge Model
THe model cards include information around hyperparameter decisions, prompt responses, modifications made under compute constraints etc. 

**TinyStories**
Find the model card here: https://huggingface.co/Aashw1n/tinystories-22M

**OpenWebText**
Find the model card here: https://huggingface.co/Aashw1n/openwebtext-103M

## Hardware

All training on Google Cloud, single-GPU throughout.

**Tokenizer training and dataset encoding**
- `c2d-standard-32` — 32 vCPU (16 core, AMD), 128 GB RAM, 300 GB balanced PD
- BPE training parallelized across 30 processes for pretokenization;
  merge loop is single-threaded
- TinyStories (2.1 GB, 10k vocab): under 1 minute
- OpenWebText (11.9 GB, 30k vocab): ~5 hours — the merge loop's
  O(merges x distinct_pairs) max-scan dominates at this corpus size
- Encoding parallelized across 30 workers on special-token chunk boundaries

**Model training**
- `a2-ultragpu-1g` — 1x NVIDIA A100 80GB SXM4, 12 vCPU, 170 GB RAM
- Ubuntu 22.04, CUDA 12.9, driver 580, PyTorch with TF32 matmuls enabled
- TinyStories (22.7M params, 327.7M tokens): ~40 min, ~$4.50
- OpenWebText (102.7M params, 819.2M tokens): ~3.5 hours
- Peak memory 52.9 GB at batch 64 / context 512 / vocab 30k

**Storage**
- Datasets, encoded arrays, and checkpoints in GCS; code and tokenizer
  artifacts in git

## Data
wget off huggingface

Tinystories

https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

OpenWebText

https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz



## All of the code here is hand-written, with minimal LLM assistance.
**Why?**
To put it simply, it's for the sake of avoiding thinking atrophy. The code here can easily be implemented one shot by any frontier model. The goal here is to form a deeper, first principles understanding of the concepts involved. The process of writing code is just a means of thinking through the concepts. The code itself is trivial.

