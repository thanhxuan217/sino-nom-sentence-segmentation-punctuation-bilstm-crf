# src/dataset.py
"""
Dataset cho token-level sequence labeling với JSONL format
"""

import json
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from typing import List, Dict, Tuple
from pathlib import Path


class ChineseTextDataset(Dataset):
    """Dataset cho văn bản Hán cổ"""
    
    def __init__(
        self,
        data_path: str,
        label_config,
        max_length: int = 512,
        vocab: Dict = None
    ):
        """
        Args:
            data_path: Đường dẫn đến file JSONL
            label_config: LabelConfig object
            max_length: Độ dài tối đa của sequence
            vocab: Vocabulary dict (nếu None sẽ build từ data)
        """
        self.label_config = label_config
        self.max_length = max_length
        
        # Load data từ JSONL
        self.examples = self._load_jsonl(data_path)
        
        # Build hoặc sử dụng vocab có sẵn
        if vocab is None:
            self.char2id, self.id2char = self._build_vocab()
        else:
            self.char2id = vocab['char2id']
            self.id2char = vocab['id2char']
        
        self.vocab_size = len(self.char2id)
    
    def _load_jsonl(self, data_path: str) -> List[Dict]:
        """
        Load dữ liệu từ JSONL file
        Format: {"text": "君不見君有疾...", "labels": ["M", "M", "E", ...]}
        """
        examples = []
        
        with open(data_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    text = data['text']
                    labels = data['labels']
                    
                    # Kiểm tra độ dài khớp
                    if len(text) != len(labels):
                        print(f"Warning: Line {line_num} - Text length ({len(text)}) "
                              f"!= Labels length ({len(labels)}). Skipping.")
                        continue
                    
                    # Kiểm tra max_length
                    if len(text) > self.max_length:
                        # Có thể cắt hoặc bỏ qua
                        print(f"Warning: Line {line_num} - Sequence too long ({len(text)}). "
                              f"Truncating to {self.max_length}.")
                        text = text[:self.max_length]
                        labels = labels[:self.max_length]
                    
                    examples.append({
                        'chars': list(text),
                        'labels': labels
                    })
                    
                except json.JSONDecodeError as e:
                    print(f"Error parsing line {line_num}: {e}")
                    continue
                except KeyError as e:
                    print(f"Missing key in line {line_num}: {e}")
                    continue
        
        print(f"Loaded {len(examples)} examples from {data_path}")
        return examples
    
    def _build_vocab(self) -> Tuple[Dict, Dict]:
        """Xây dựng vocabulary từ dữ liệu"""
        chars = set()
        for example in self.examples:
            chars.update(example['chars'])
        
        # Special tokens
        char2id = {
            '[PAD]': 0,
            '[UNK]': 1,
        }
        
        # Add characters
        for idx, char in enumerate(sorted(chars), start=2):
            char2id[char] = idx
        
        id2char = {idx: char for char, idx in char2id.items()}
        
        return char2id, id2char
    
    def get_vocab(self) -> Dict:
        """Trả về vocabulary để share với các dataset khác"""
        return {
            'char2id': self.char2id,
            'id2char': self.id2char
        }
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        """
        Returns:
            Dict với input_ids, label_ids, length
        """
        example = self.examples[idx]
        chars = example['chars']
        labels = example['labels']
        
        # Convert chars to ids
        input_ids = [self.char2id.get(c, self.char2id['[UNK]']) for c in chars]
        
        # Convert labels to ids
        label_ids = [self.label_config.label2id[label] for label in labels]
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'label_ids': torch.tensor(label_ids, dtype=torch.long),
            'length': len(input_ids)
        }


def collate_fn(batch):
    """Collate function với padding"""
    input_ids = [item['input_ids'] for item in batch]
    label_ids = [item['label_ids'] for item in batch]
    lengths = [item['length'] for item in batch]
    
    # Padding
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=0)
    label_ids_padded = pad_sequence(label_ids, batch_first=True, padding_value=-100)
    
    return {
        'input_ids': input_ids_padded,
        'label_ids': label_ids_padded,
        'lengths': torch.tensor(lengths, dtype=torch.long)
    }


def create_dataloaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Dataset,
    batch_size: int,
    num_workers: int,
    rank: int = 0,
    world_size: int = 1
):
    """
    Tạo DataLoader cho distributed training
    """
    if world_size > 1:
        from torch.utils.data.distributed import DistributedSampler
        
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True
        )
    
    # Val và test không cần distributed
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader
