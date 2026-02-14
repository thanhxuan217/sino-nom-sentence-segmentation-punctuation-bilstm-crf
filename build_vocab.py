# build_vocab.py
"""
Script để build vocabulary từ parquet files.
Chạy 1 lần trước khi train.

Usage:
    python build_vocab.py --data_dir data/ --split train --output vocab.json
"""

import argparse
import json
import glob
import os
from datasets import load_dataset


def build_vocab(data_dir: str, split: str = 'train') -> dict:
    """
    Scan toàn bộ parquet files trong split folder và build vocab.
    """
    split_dir = os.path.join(data_dir, split)
    parquet_files = sorted(glob.glob(os.path.join(split_dir, '*.parquet')))

    if not parquet_files:
        raise FileNotFoundError(f"Không tìm thấy file parquet trong {split_dir}")

    print(f"Tìm thấy {len(parquet_files)} parquet files trong {split_dir}")

    # Load streaming để không tốn RAM
    dataset = load_dataset(
        'parquet',
        data_files=parquet_files,
        split='train',
        streaming=True
    )

    chars = set()
    count = 0

    for example in dataset:
        text = example['text']
        chars.update(text)
        count += 1
        if count % 100000 == 0:
            print(f"  Đã xử lý {count:,} examples, vocab size: {len(chars):,}")

    print(f"Tổng: {count:,} examples, {len(chars):,} unique chars")

    # Build char2id
    char2id = {
        '[PAD]': 0,
        '[UNK]': 1,
    }

    for idx, char in enumerate(sorted(chars), start=2):
        char2id[char] = idx

    id2char = {idx: char for char, idx in char2id.items()}

    return {
        'char2id': char2id,
        'id2char': id2char,
        'vocab_size': len(char2id),
        'num_examples': count,
    }


def main():
    parser = argparse.ArgumentParser(description='Build vocabulary từ parquet files')

    parser.add_argument('--data_dir', type=str, required=True,
                        help='Thư mục gốc chứa data (có sub-folders train/, val/, test/)')
    parser.add_argument('--split', type=str, default='train',
                        help='Split để scan (default: train)')
    parser.add_argument('--output', type=str, default='vocab.json',
                        help='Đường dẫn output vocab file')

    args = parser.parse_args()

    print("=" * 60)
    print("Building vocabulary...")
    print("=" * 60)

    vocab = build_vocab(args.data_dir, args.split)

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    print(f"\nĐã lưu vocab ({vocab['vocab_size']} chars) vào {args.output}")


if __name__ == '__main__':
    main()
