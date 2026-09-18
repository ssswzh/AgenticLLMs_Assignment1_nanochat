"""
Task 1: Tokenization
1. how vocabulary size affects compression ratio and sequence length
2. how vocabulary size matters for downstream language models
3. how tokenizer handles numbers, source code, and non-English text
"""

import os
import pickle
import argparse
from nanochat.common import get_base_dir
base_dir = get_base_dir()


# Hardcode sample texts to test the tokenizer
sample_texts = {
    # English text, copied from Wikipedia (https://en.wikipedia.org/wiki/Large_language_model#Tokenization)
    "english": """
As machine learning algorithms process numbers rather than text, the text must be converted to numbers. In the first step, a vocabulary is decided upon, then integer indices are arbitrarily but uniquely assigned to each vocabulary entry, and finally, an embedding is associated with the integer index. Algorithms include byte-pair encoding (BPE) and WordPiece. There are also special tokens serving as control characters, such as [MASK] for masked-out token (as used in BERT), and [UNK] ("unknown") for characters not appearing in the vocabulary. Also, some special symbols are used to denote special text formatting. For example, "G" denotes a preceding whitespace in RoBERTa and GPT and "##" denotes continuation of a preceding word in BERT.
As an example, consider a tokenizer based on byte-pair encoding. In the first step, all unique characters (including blanks and punctuation marks) are treated as an initial set of n-grams (i.e. initial set of uni-grams). Successively the most frequent pair of adjacent characters is merged into a bi-gram and all instances of the pair are replaced by it. All occurrences of adjacent pairs of (previously merged) n-grams that most frequently occur together are then again merged into even lengthier n-gram, until a vocabulary of prescribed size is obtained. After a tokenizer is trained, any text can be tokenized by it, as long as it does not contain characters not appearing in the initial-set of uni-grams.""",

    # Numbers
    "numbers": """20260918, 19930521, 19971103, 3.1415926""",

    # Source code, add from 1 to 100
    "source_code": """
total = 0
for i in range(1, 101):
    total += i
print(total)
""",

    # Non-English text, Chinese text translated by Google Translate, source: https://ssswzh.github.io/
    "non_english": """我是一名数据科学家，拥有7年以上的生物信息学工程经验，目前专注于应用人工智能和机器学习。我的工作一直围绕着数据噪声大、维度高且具有重要意义的领域展开：临床基因组学、转化研究、诊断和医疗保健分析。我投身人工智能和机器学习领域，既源于长久以来的好奇心，也源于明确的职业方向：智能系统正日益成为科学发现、医疗保健和现实世界决策支持的核心。我尤其热衷于构建稳健、易解释且超越基准测试的实用模型。"""
}


def GetArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--small', type=str, default=f'{base_dir}/tokenizer_8192/tokenizer.pkl', help='Path to the tokenizer file with 8192 vocabulary size')
    parser.add_argument('--large', type=str, default=f'{base_dir}/tokenizer_32768/tokenizer.pkl', help='Path to the tokenizer file with 32768 vocabulary size')
    parser.add_argument('--output', type=str, default=f'tokenizer_evaluation_results.tsv', help='Path to save the evaluation results')
    args = parser.parse_args()
    if not os.path.exists(os.path.dirname(args.output)):
        os.makedirs(os.path.dirname(args.output))
    return args

def load_tokenizer(tokenizer_path):
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)
    return tokenizer


def evaluate_tokenizer(tokenizer, text):
    tokens = tokenizer.encode(text)
    # decode the token id back to byte/token for inspection
    pieces = [tokenizer.decode_single_token_bytes(i) for i in tokens]
    return {
        "text_length": len(text),
        "token_length": len(tokens),
        "compression_ratio": len(tokens) / len(text),
        "tokens": tokens[:100],  # Show only the first 100 tokens for brevity
        "pieces": pieces[:100]
    }
    

def main():
    args = GetArgs()
    if not os.path.exists(args.small):
        raise FileNotFoundError(f"Tokenizer file not found: {args.small}")
    else:
        print(f"Loading small tokenizer from: {args.small}")
    small_tokenizer = load_tokenizer(args.small)
    if not os.path.exists(args.large):
        raise FileNotFoundError(f"Tokenizer file not found: {args.large}")
    else:
        print(f"Loading large tokenizer from: {args.large}")
    large_tokenizer = load_tokenizer(args.large)

    with open(args.output, 'w') as f:
        f.write("Text Type\tVocabulary Size\tText Length\tToken Length\tCompression Ratio\n")
        for text_type, text in sample_texts.items():
            print(f"===== Evaluating text type: {text_type} =====")
            for vocab_size, tokenizer in [(8192, small_tokenizer), (32768, large_tokenizer)]:
                f.write(text_type + "\t")
                result = evaluate_tokenizer(tokenizer, text)
                print(f"Vocabulary size: {vocab_size}")
                print(f"Text length: {result['text_length']}")
                print(f"Token length: {result['token_length']}")
                print(f"Compression ratio: {result['compression_ratio']:.4f}")
                print(f"Tokens (first 100): {result['tokens']}")
                print(f"Pieces (first 100): {result['pieces']}")
                print("-" * 20)
                f.write(f"{vocab_size}\t{result['text_length']}\t{result['token_length']}\t{result['compression_ratio']:.4f}\n")
    f.close()


if __name__ == "__main__":
    main()
