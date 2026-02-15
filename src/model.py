# src/model.py
"""
BiLSTM + Linear/CRF model cho token classification
"""

import torch
import torch.nn as nn
from torchcrf import CRF
from typing import Optional, Dict


class BiLSTMLinear(nn.Module):
    """BiLSTM + Linear Head"""
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_labels: int,
        dropout: float = 0.3,
        bidirectional: bool = True
    ):
        super().__init__()
        
        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=0
        )
        
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        self.layer_norm = nn.LayerNorm(lstm_output_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(lstm_output_dim, num_labels)
        
    def forward(
        self,
        input_ids: torch.Tensor,
        lengths: torch.Tensor,
        label_ids: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            input_ids: [batch_size, seq_len]
            lengths: [batch_size]
            label_ids: [batch_size, seq_len], optional
        """
        batch_size, seq_len = input_ids.shape
        
        # Embedding
        embedded = self.embedding(input_ids)
        
        # Pack sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )
        
        # BiLSTM
        packed_output, _ = self.lstm(packed_embedded)
        
        # Unpack
        lstm_output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output,
            batch_first=True,
            total_length=seq_len
        )
        
        lstm_output = self.layer_norm(lstm_output)
        lstm_output = self.dropout(lstm_output)
        
        # Classification
        logits = self.classifier(lstm_output)
        
        output = {'logits': logits}
        
        # Compute loss
        if label_ids is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                logits.view(-1, logits.size(-1)),
                label_ids.view(-1)
            )
            output['loss'] = loss
        
        return output


class BiLSTMCRF(nn.Module):
    """BiLSTM + CRF Head"""
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_labels: int,
        dropout: float = 0.3,
        bidirectional: bool = True
    ):
        super().__init__()
        
        self.num_labels = num_labels
        
        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=0
        )
        
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        self.layer_norm = nn.LayerNorm(lstm_output_dim)
        self.dropout = nn.Dropout(dropout)
        self.hidden2tag = nn.Linear(lstm_output_dim, num_labels)
        self.crf = CRF(num_labels, batch_first=True)
        
    def forward(
        self,
        input_ids: torch.Tensor,
        lengths: torch.Tensor,
        label_ids: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            input_ids: [batch_size, seq_len]
            lengths: [batch_size]
            label_ids: [batch_size, seq_len], optional
        """
        batch_size, seq_len = input_ids.shape
        
        # Embedding
        embedded = self.embedding(input_ids)
        
        # Pack sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )
        
        # BiLSTM
        packed_output, _ = self.lstm(packed_embedded)
        
        # Unpack
        lstm_output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output,
            batch_first=True,
            total_length=seq_len
        )
        
        lstm_output = self.layer_norm(lstm_output)
        lstm_output = self.dropout(lstm_output)
        
        # Emissions
        emissions = self.hidden2tag(lstm_output)
        
        # Mask (True cho valid positions)
        mask = (input_ids != 0)
        
        output = {'emissions': emissions}
        
        # Compute loss và decode
        if label_ids is not None:
            # Chuyển -100 thành 0 tạm thời cho CRF
            label_ids_masked = label_ids.clone()
            label_ids_masked[label_ids == -100] = 0
            
            # CRF loss
            loss = -self.crf(emissions, label_ids_masked, mask=mask, reduction='mean')
            output['loss'] = loss
        
        # Decode predictions (chỉ khi inference, không cần khi training)
        if label_ids is None:
            predictions = self.crf.decode(emissions, mask=mask)
            output['predictions'] = predictions
        
        return output


def create_model(
    vocab_size: int,
    num_labels: int,
    model_config,
) -> nn.Module:
    """Factory function để tạo model"""
    
    if model_config.use_crf:
        model = BiLSTMCRF(
            vocab_size=vocab_size,
            embedding_dim=model_config.embedding_dim,
            hidden_dim=model_config.hidden_dim,
            num_layers=model_config.num_layers,
            num_labels=num_labels,
            dropout=model_config.dropout,
            bidirectional=model_config.bidirectional
        )
    else:
        model = BiLSTMLinear(
            vocab_size=vocab_size,
            embedding_dim=model_config.embedding_dim,
            hidden_dim=model_config.hidden_dim,
            num_layers=model_config.num_layers,
            num_labels=num_labels,
            dropout=model_config.dropout,
            bidirectional=model_config.bidirectional
        )
    
    return model
