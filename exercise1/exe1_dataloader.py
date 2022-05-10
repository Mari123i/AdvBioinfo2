from torch.utils.data import DataLoader, Dataset
import torch
from transformers import BertTokenizer, BertModel
from pathlib import Path
from pyfaidx import Fasta
from typing import Dict, Tuple, List


class TokenSeqDataset(Dataset):
    def __init__(self, fasta_path: Path, labels_dict: Dict[str, int], max_num_residues: int, protbert_cache: Path, device: torch.device):
        self.protbert = self.load_model()
        self.tokenizer = self.load_tokenizer()
        self.data = self.parse_fasta_input(fasta_path)

    def load_tokenizer(self) -> BertTokenizer:
        loaded_tokenizer = BertTokenizer.from_pretrained('Rostlab/prot_bert_bfd', do_lower_case=False )
        return loaded_tokenizer

    def load_model(self) -> BertModel:
        loaded_model = BertModel.from_pretrained("Rostlab/prot_bert_bfd")
        loaded_model = loaded_model.to(self.device)
        loaded_model = loaded_model.eval()
        return loaded_model

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        key_list = list(self.data.keys())
        key = key_list[index]
        sequence = self.data[key]

        return tensor, 0

    def __len__(self) -> int:
        return len(self.data)

    def tokenize(self, seq: str) -> Tuple[torch.Tensor, torch.Tensor]:
        return None

    def embedd(self, tokens: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        print("hey")
        return None

    def parse_fasta_input(self, input_file: Path) -> Dict[str, str]:
        fasta_dict = {}
        inputs = Fasta(input_file)
        for key in inputs.keys():
            fasta_dict[key] = inputs[key]

        return fasta_dict


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

