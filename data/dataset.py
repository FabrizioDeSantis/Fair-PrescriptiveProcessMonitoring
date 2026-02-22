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

class StandardDataset(Dataset):
    def __init__(self, sequences, labels, case_ids):
        
        self.sequences = []
        self.case_ids = torch.tensor(case_ids, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.float)
        for seq in sequences:
            for i in range(len(seq)):
                seq[i] = torch.tensor(seq[i], dtype=torch.float32)
            self.sequences.append(seq)
        
        self.sequences = [
            torch.cat(seq, dim=0) for seq in self.sequences
        ]

    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return (
            self.sequences[idx],
            self.labels[idx],
            self.case_ids[idx]
        )

class CounterfactualDataset(Dataset):
    def __init__(self, sequences, labels, case_ids):
        self.sequences_og = []
        self.sequences_cf = []
        self.case_ids = torch.tensor(case_ids, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.float)

        for inner_list in sequences:
            # Process original sequence (inner_list[0])
            for i in range(len(inner_list[0])):
                inner_list[0][i] = torch.tensor(inner_list[0][i], dtype=torch.float32)
            self.sequences_og.append(inner_list[0])

            # Process counterfactual sequences (inner_list[1] is a list of lists)
            cf_tensors = []
            for cf_seq in inner_list[1]:
                for i in range(len(cf_seq)):
                    cf_seq[i] = torch.tensor(cf_seq[i], dtype=torch.float32)
                cf_tensors.append(torch.cat(cf_seq, dim=0))
            self.sequences_cf.append(torch.stack(cf_tensors))  # [num_counterfactual, seq_len*num_features]

        self.sequences_og = [
            torch.cat(seq, dim=0) for seq in self.sequences_og
        ]

    def __len__(self):
        return len(self.sequences_og)
    
    def __getitem__(self, idx):
        return (
            self.sequences_og[idx],
            self.sequences_cf[idx],
            self.labels[idx],
            self.case_ids[idx]
        )