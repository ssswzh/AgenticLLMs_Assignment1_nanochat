from nanochat.dataset import parquets_iter_batched

TARGET_CHARS = 500_000_000
DOC_CAP = 10_000


def main():
    total_chars = 0
    total_docs = 0

    for batch in parquets_iter_batched(split="train"):
        for doc in batch:
            # Match tok_train.py behavior:
            # each document contributes at most DOC_CAP characters
            usable_text = doc[:DOC_CAP]

            total_chars += len(usable_text)
            total_docs += 1

    print(f"Documents available: {total_docs:,}")
    print(f"Usable characters: {total_chars:,}")
    print(f"Approx. usable data: {total_chars / 1e6:.1f}M chars")

    if total_chars >= TARGET_CHARS:
        print("Verification passed: enough training text is available.")
    else:
        missing = TARGET_CHARS - total_chars
        print("Verification failed: not enough training text.")
        print(f"Additional characters required: {missing:,}")


if __name__ == "__main__":
    main()