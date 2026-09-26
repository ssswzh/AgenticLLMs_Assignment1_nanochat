# Agentic LLMs Course -- Assignment 1: Build LLMs from Scratch

Repository forked from [karpathy/nanochat](https://github.com/karpathy/nanochat).


## Setup

Clone the nanochat repository, install uv. Change the base directory of nanochat to `/data/s4683226/AgenticLLMs/nanochat`, set env `$NANOCHAT_BASE_DIR`. 
```bash
uv sync --extra gpu
source .venv/bin/activate
export NANOCHAT_BASE_DIR=/data/s4683226/AgenticLLMs/nanochat
```

Download ClimbMix dataset.The dataset will be stored in `$NANOCHAT_BASE_DIR/base_data_climbmix/`.
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
python -m scripts.tok_train --max-chars 500_000_000 --vocab-size 8192 --outdir $NANOCHAT_BASE_DIR/tokenizer_8192
python -m scripts.tok_train --max-chars 500_000_000 --vocab-size 32768 --outdir $NANOCHAT_BASE_DIR/tokenizer_32768
```

To test how vocabulary size affect compression ratio and sequence length for a fixed text sample, run the following script. This script also include numbers, source code, and non-English text for answering subtask 1.4.
```bash
python -m group28.task1_evaluate_tokenizer --output group28_results/task1_tokenizer_evaluation_results.tsv
```


## Task 2: Pre-training

We used 10 shards download in *Task 1* (approx. 2B characters), rerun a tokenizer. 
```bash
python -m scripts.tok_train
```

Use slightly modified script `scripts.base_train` to log the training process. Set `--depth=2` for pre-training. The pre-trained model is stored in `$NANOCHAT_BASE_DIR/models/`. Set `--device-batch-size 16` to avoid OOM. Set `--sample-every=999999` to run test sample only at the end of training. Set `--eval-tokens 2097152` to evaluate on 2M (2^{21}) tokens for computational efficiency.
```bash
python -m scripts.base_train --depth 2 --eval-every 50 --save-every 100 --target-param-data-ratio 12 --model-tag depth2_eval50 --device-batch-size 16 --sample-every=999999 --eval-tokens 2097152 
# python -m scripts.base_train --depth 4 --eval-every 50 --save-every 100 --target-param-data-ratio 12 --model-tag depth4_eval50 --device-batch-size 16 --sample-every=999999 --eval-tokens 2097152
```

Draw BPB plot for the pre-trained model. The BPB plot is stored in `$NANOCHAT_BASE_DIR/base_checkpoints/depth2_eval50/metrics.png`.
```bash
python -m group28.task2_bpb_eval "$NANOCHAT_BASE_DIR/base_checkpoints/depth2_eval50/metrics.jsonl" 0.01
```


## Task 3: Mid-Training and Supervised Fine-Tuning

Add `--stage` to isolate mid-training and supervised fine-tuning. Use `--input-source` to specify the input data source. Use `--input-model-tag` to specify the input model tag. Use `--output-model-tag` to specify the output model tag.
```bash
# 3.1 Mid-Training
python -m scripts.chat_sft --stage midtrain --input-source base --input-model-tag depth2_eval50 --output-model-tag depth2_midtrain --inspect-data --chatcore-every 999999
# 3.2 Supervised Fine-Tuning
python -m scripts.chat_sft --stage sft --input-source sft --input-model-tag depth2_midtrain --output-model-tag depth2_sft --chatcore-every 999999
```

Evaluate 3 models separately.
```bash
python -m scripts.chat_eval -i base -g depth2_eval50 -a 'ARC-Easy|ARC-Challenge|GSM8K' -o group28/task3_eval_base.json
python -m scripts.chat_eval -i sft -g depth2_midtrain -a 'ARC-Easy|ARC-Challenge|GSM8K' -o group28/task3_eval_midtrain.json
python -m scripts.chat_eval -i sft -g depth2_sft -a 'ARC-Easy|ARC-Challenge|GSM8K' -o group28/task3_eval_sft.json
```
