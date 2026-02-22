# src/dataset.py
"""
Dataset cho token-level sequence labeling với JSONL format và Parquet streaming
"""

import os
import json
import glob
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
from torch.nn.utils.rnn import pad_sequence
from typing import List, Dict, Tuple, Optional
from pathlib import Path


class ChineseTextDataset(Dataset):
    """Dataset cho văn bản Hán cổ"""
    
    def __init__(
        self,
        data_path: str,
        label_config,
        max_length: int = 256,
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


# =============================================================================
# Parquet Streaming Dataset
# =============================================================================

def load_vocab(vocab_path: str) -> Dict:
    """Load vocabulary từ file JSON"""
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    # Ensure id2char keys are ints
    vocab['id2char'] = {int(k): v for k, v in vocab['id2char'].items()}
    return vocab


class ParquetStreamingDataset(IterableDataset):
    """
    Streaming dataset đọc từ Parquet files qua HuggingFace datasets.
    Không load toàn bộ dữ liệu vào RAM.
    """

    def __init__(
        self,
        data_dir: str,
        split: str,
        label_config,
        vocab: Dict,
        max_length: int = 256,
        shuffle_buffer: int = 0,
        seed: int = 42,
        epoch: int = 0,
        max_samples: int = 0,
        no_shard: bool = False,
    ):
        """
        Args:
            data_dir: Thư mục gốc chứa các sub-folder (train/, val/, test/)
            split: Tên split (train, val, test)
            label_config: LabelConfig object
            vocab: Dict với 'char2id' và 'id2char'
            max_length: Độ dài tối đa của sequence
            shuffle_buffer: Kích thước buffer để shuffle (0 = không shuffle)
            seed: Random seed
            epoch: Epoch hiện tại (dùng để thay đổi shuffle order)
            max_samples: Số mẫu tối đa (0 = không giới hạn)
        """
        super().__init__()
        self.label_config = label_config
        self.char2id = vocab['char2id']
        self.id2char = vocab['id2char']
        self.vocab_size = len(self.char2id)
        self.max_length = max_length
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed
        self.epoch = epoch
        self.max_samples = max_samples
        self.no_shard = no_shard

        # Tìm tất cả parquet files trong split directory
        split_dir = os.path.join(data_dir, split)
        self.parquet_files = sorted(glob.glob(os.path.join(split_dir, '*.parquet')))

        if not self.parquet_files:
            raise FileNotFoundError(
                f"Không tìm thấy file parquet nào trong {split_dir}"
            )

    def set_epoch(self, epoch: int):
        """Cập nhật epoch để thay đổi shuffle order"""
        self.epoch = epoch

    def _process_example(self, example):
        """Chuyển đổi 1 example thành tensor"""
        text = example['text']
        labels = example['labels']

        # Xử lý labels nếu lưu dưới dạng string trong parquet
        if isinstance(labels, str):
            try:
                labels = json.loads(labels)
            except (json.JSONDecodeError, ValueError):
                # Thử split nếu không phải JSON
                labels = labels.split()

        # Kiểm tra độ dài khớp
        if len(text) != len(labels):
            return 'length_mismatch'

        # Truncate nếu cần
        if len(text) > self.max_length:
            text = text[:self.max_length]
            labels = labels[:self.max_length]

        chars = list(text)

        # Convert chars to ids
        input_ids = [self.char2id.get(c, self.char2id.get('[UNK]', 1)) for c in chars]

        # Convert labels to ids
        try:
            label_ids = [self.label_config.label2id[label] for label in labels]
        except KeyError:
            return 'label_key_error'

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'label_ids': torch.tensor(label_ids, dtype=torch.long),
            'length': len(input_ids)
        }

    def __iter__(self):
        import torch.distributed as dist
        from datasets import load_dataset

        # Lấy thông tin worker và rank
        # QUAN TRỌNG: Dùng dist.is_initialized() thay vì đọc env var RANK/WORLD_SIZE.
        # Env var có thể vẫn tồn tại sau khi torchrun kết thúc, khiến evaluate.py
        # standalone bị nhầm world_size > 1 và sharding sai (data bị nhân lên).
        worker_info = torch.utils.data.get_worker_info()
        is_distributed = dist.is_available() and dist.is_initialized()
        rank = dist.get_rank() if is_distributed else 0
        world_size = dist.get_world_size() if is_distributed else 1

        # Load dataset streaming
        dataset = load_dataset(
            'parquet',
            data_files=self.parquet_files,
            split='train',  # HF datasets dùng 'train' cho single split
            streaming=True
        )

        # Shuffle nếu cần
        if self.shuffle_buffer > 0:
            dataset = dataset.shuffle(
                seed=self.seed + self.epoch,
                buffer_size=self.shuffle_buffer
            )

        # Tính toán shard params để chia dữ liệu thủ công
        # thay vì dùng dataset.filter(with_indices=True) vì nó crash
        # khi num_workers > 0 (features=None trong worker subprocess)
        # no_shard=True: rank=0 đọc toàn bộ data (dùng trong evaluate để tránh mất data)
        if self.no_shard:
            total_shards = 1
            shard_index = 0
        else:
            total_shards = world_size
            shard_index = rank
            if worker_info is not None:
                num_workers = worker_info.num_workers
                worker_id = worker_info.id
                total_shards = world_size * num_workers
                shard_index = rank * num_workers + worker_id

        # Yield processed examples với diagnostic logging
        count = 0
        total_seen = 0
        skip_length_mismatch = 0
        skip_label_error = 0
        logged_sample = False
        for global_idx, example in enumerate(dataset):
            # Skip examples không thuộc shard này
            if total_shards > 1 and global_idx % total_shards != shard_index:
                continue
            total_seen += 1

            # Log mẫu đầu tiên để debug
            if not logged_sample and rank == 0:
                logged_sample = True
                print(f"[DEBUG] First example keys: {list(example.keys())}")
                print(f"[DEBUG] text type: {type(example['text'])}, len: {len(example['text'])}")
                print(f"[DEBUG] labels type: {type(example['labels'])}, "
                      f"value[:100]: {str(example['labels'])[:100]}")

            result = self._process_example(example)
            if isinstance(result, str):
                if result == 'length_mismatch':
                    skip_length_mismatch += 1
                elif result == 'label_key_error':
                    skip_label_error += 1
                    if skip_label_error <= 3 and rank == 0:
                        labels = example['labels']
                        if isinstance(labels, str):
                            try:
                                labels = json.loads(labels)
                            except Exception:
                                labels = labels.split()
                        sample_labels = labels[:5] if isinstance(labels, list) else str(labels)[:50]
                        print(f"[DEBUG] Label KeyError - sample labels: {sample_labels}")
                continue
            if result is not None:
                yield result
                count += 1
                if self.max_samples > 0 and count >= self.max_samples:
                    break

            # Log tiến độ mỗi 100000 examples
            if total_seen % 100000 == 0 and rank == 0:
                print(f"[DEBUG] Seen {total_seen}, yielded {count}, "
                      f"skip_length={skip_length_mismatch}, skip_label={skip_label_error}")

        if rank == 0:
            print(f"[Dataset Stats] Total seen: {total_seen}, Yielded: {count}, "
                  f"Skip(length): {skip_length_mismatch}, Skip(label): {skip_label_error}")


def create_streaming_dataloaders(
    data_dir: str,
    label_config,
    vocab: Dict,
    batch_size: int,
    num_workers: int,
    max_length: int = 256,
    shuffle_buffer: int = 10000,
    seed: int = 42,
    train_split: str = 'train',
    val_split: str = 'val',
    test_split: str = 'test',
    val_max_samples: int = 0,
):
    """
    Tạo DataLoader cho streaming datasets.
    Trả về (train_loader, val_loader, full_val_loader, test_loader, train_dataset, vocab_size).
    train_dataset được trả về để gọi set_epoch().
    val_loader: giới hạn val_max_samples mẫu (dùng khi train).
    full_val_loader: toàn bộ mẫu (dùng sau khi train xong).
    """
    train_dataset = ParquetStreamingDataset(
        data_dir=data_dir,
        split=train_split,
        label_config=label_config,
        vocab=vocab,
        max_length=max_length,
        shuffle_buffer=shuffle_buffer,
        seed=seed,
    )

    val_dataset = ParquetStreamingDataset(
        data_dir=data_dir,
        split=val_split,
        label_config=label_config,
        vocab=vocab,
        max_length=max_length,
        shuffle_buffer=0,  # Không shuffle val
        max_samples=val_max_samples,  # Giới hạn mẫu khi train
    )

    full_val_dataset = ParquetStreamingDataset(
        data_dir=data_dir,
        split=val_split,
        label_config=label_config,
        vocab=vocab,
        max_length=max_length,
        shuffle_buffer=0,
        max_samples=0,  # Không giới hạn - toàn bộ mẫu
    )

    test_dataset = ParquetStreamingDataset(
        data_dir=data_dir,
        split=test_split,
        label_config=label_config,
        vocab=vocab,
        max_length=max_length,
        shuffle_buffer=0,  # Không shuffle test
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    full_val_loader = DataLoader(
        full_val_dataset,
        batch_size=batch_size * 2,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size * 2,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    return train_loader, val_loader, full_val_loader, test_loader, train_dataset, train_dataset.vocab_size
