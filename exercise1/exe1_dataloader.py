from torch.utils.data import DataLoader, Dataset
import torch
from transformers import BertTokenizer, BertModel
from pathlib import Path
from pyfaidx import Fasta
from typing import Dict, Tuple, List


class TokenSeqDataset(Dataset):
    def __init__(self, fasta_path: Path, labels_dict: Dict[str, int], max_num_residues: int, protbert_cache: Path, device: torch.device):
        self.protbert = None
        self.tokenizer = None
        self.data = None

    def load_tokenizer(self) -> BertTokenizer:
        return None

    def load_model(self) -> BertModel:
        return None

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        return None

    def __len__(self) -> int:
        return 0

    def tokenize(self, seq: str) -> Tuple[torch.Tensor, torch.Tensor]:
        return None

    def embedd(self, tokens: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        print("hey")
        return None

    def parse_fasta_input(self, input_file: Path) -> Dict[str, str]:
        return None


def collate_paired_sequences(data: List[Tuple[torch.Tensor, int]]) -> Tuple[torch.Tensor, torch.Tensor]:
    tensors=[x[0] for x in data]
    ints =[x[1] for x in data]
    stacked_tensors=torch.stack(tensors)
    int_tensor = torch.as_tensor(ints)
    return (stacked_tensors,int_tensor)


def get_dataloader(fasta_path: Path, labels_dict: Dict[str, int], batch_size: int, max_num_residues: int,
                   protbert_cache: Path, device: torch.device, seed: int) -> DataLoader:
    torch.manual_seed(seed)
    tokenDataset = TokenSeqDataset(fasta_path,labels_dict,max_num_residues,protbert_cache, device)
    dataloader =DataLoader(dataset=tokenDataset,batch_size=batch_size,shuffle=True,collate_fn=collate_paired_sequences)
    return dataloader

