# Agentic LLMs Course -- Assignment 1: Build LLMs from Scratch

Repository forked from [karpathy/nanochat](https://github.com/karpathy/nanochat).


## Setup

Clone the nanochat repository, install uv. The base directory of nanochat is `~/.cache/nanochat/`, set env `$NANOCHAT_BASE_DIR`. 
```bash
uv sync --extra gpu
source .venv/bin/activate
export NANOCHAT_BASE_DIR=~/.cache/nanochat/
```

Download ClimbMix dataset.The dataset will be stored in `$NANOCHAT_BASE_DIR/dataset/`.
```bash
python -m nanochat.dataset -n 10
```

Check the data amount, iterate all documents, each doc contribute at most `doc_cap` characters. Obtain approximately 2233M characters from the dataset.
```bash
python -m group28.verify_data_amount
# Output:
# Documents available: 846,848
# Usable characters: 2,233,005,799
# Approx. usable data: 2233.0M chars
# Verification passed: enough training text is available.
```


## Task 1: Tokenization

Train a BPE tokenizer using nanochat’s Rust tokenizer trainer on a 500 MB sample of the CLIMBMix dataset. Run two experiments: one with a vocabulary size of 8,192 tokens and one with 32,768 tokens.

We assume approximately 500M characters as the 500 MB tokenizer-training sample.
```bash
# Train tokenizer with 8192 tokens and 32768 tokens, output to separate directories
python -m scripts.tok_train_group28 --max-chars 500_000_000 --vocab-size 8192 --outdir $NANOCHAT_BASE_DIR/tokenizer_8192
python -m scripts.tok_train_group28 --max-chars 500_000_000 --vocab-size 32768 --outdir $NANOCHAT_BASE_DIR/tokenizer_32768
```

To test how vocabulary size affect compression ratio and sequence length for a fixed text sample, run the following script. This script also include numbers, source code, and non-English text for answering subtask 1.4.
```bash
python -m group28.task1_evaluate_tokenizer --output group28_results/task1_tokenizer_evaluation_results.tsv
```




## Task 2: Pre-training
