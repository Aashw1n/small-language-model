# Small Language Model
An end-to-end implementation of a small language model + a gpt2-style tokenizer. Training runs and experimentation/ablations included.  



## The Tokenizer

A run-of-the-mill BPE tokenizer in Python, with a training script. 

The tokenizer has been trained on two datasets: 
    - TinyStories up to a 10k vocab size
    - OpenWebText up to a 30k vocab size


## The Langauge Model
TinyStories
Find the model card here: https://huggingface.co/Aashw1n/tinystories-22M



## Data
wget off huggingface

Tinystories

https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

OpenWebText

https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz



## All of the code here is hand-written, with minimal LLM assistance. Why?
To put it simply, it's for the sake of avoiding thinking atrophy. The code here can easily be implemented one shot by any frontier model. The goal here is to form a deeper, first principles understanding of the concepts involved. The process of writing code is just a means of thinking through the concepts. The code itself is trivial.

