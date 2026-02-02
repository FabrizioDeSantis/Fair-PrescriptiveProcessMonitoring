import pandas as pd
from torch.utils.data import Dataset
import torch
from dataclasses import dataclass

@dataclass
class ModelConfig:
    hidden_size: int
    num_layers: int
    sequence_length: int
    dataset: str
    dropout_rate: float = 0.1
    learning_rate: float = 0.001
    num_features: int = 4
    num_epochs: int = 100
    seed: int = 42
    
class CounterfactualDataset(Dataset):
    def __init__(self, sequences, sequences_cf, labels, case_ids):
        
        self.sequences = []
        self.sequences_cf = []
        self.case_ids = torch.tensor(case_ids, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.float)
        for seq in sequences:
            for i in range(len(seq)):
                seq[i] = torch.tensor(seq[i], dtype=torch.float32)
            self.sequences.append(seq)
        
        self.sequences = [
            torch.cat(seq, dim=0) for seq in self.sequences
        ]

        for seq in sequences_cf:
            for i in range(len(seq)):
                seq[i] = torch.tensor(seq[i], dtype=torch.float32)
            self.sequences_cf.append(seq)

        self.sequences_cf = [
            torch.cat(seq, dim=0) for seq in self.sequences_cf
        ]

    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return (
            self.sequences[idx],
            self.sequences_cf[idx],
            self.labels[idx],
            self.case_ids[idx]
        )