from torch.utils.data import DataLoader, Dataset
import torch
from transformers import BertTokenizer, BertModel
from pathlib import Path
from pyfaidx import Fasta
from typing import Dict, Tuple, List


class TokenSeqDataset(Dataset):
    def __init__(self, fasta_path: Path, labels_dict: Dict[str, int], max_num_residues: int, protbert_cache: Path, device: torch.device):
        self.device = device
        self.labels_dict = labels_dict
        self.max_num_residues = max_num_residues
        self.protbert_cache = protbert_cache
        self.model = self.load_model()
        self.tokenizer = self.load_tokenizer()
        self.data = self.parse_fasta_input(fasta_path)



    def load_tokenizer(self) -> BertTokenizer:
        loaded_tokenizer = BertTokenizer.from_pretrained('Rostlab/prot_bert_bfd', do_lower_case=False, cache_dir=self.protbert_cache)
        return loaded_tokenizer

    def load_model(self) -> BertModel:
        loaded_model = BertModel.from_pretrained("Rostlab/prot_bert_bfd", cache_dir=self.protbert_cache)
        loaded_model = loaded_model.to(self.device)
        loaded_model = loaded_model.eval()
        return loaded_model

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        key_list = list(self.data.keys())
        key = key_list[index]
        sequence = self.data[key]
        if len(sequence) > self.max_num_residues:
            sequence = sequence[:self.max_num_residues]
        token_ids, att_mask = self.tokenize(sequence)
        embedding = self.embedd(token_ids,att_mask)
        return embedding, self.labels_dict[key]

    def __len__(self) -> int:
        return len(self.data)

    def tokenize(self, seq: str) -> Tuple[torch.Tensor, torch.Tensor]:
        ids = self.tokenizer.batch_encode_plus(seq, add_special_tokens=True, max_length=max_num_residues+2,pad_to_max_length=True)
        input_ids = torch.tensor(ids['input_ids']).to(self.device)
        attention_mask = torch.tensor(ids['attention_mask']).to(self.device)
        return input_ids, attention_mask

    def embedd(self, tokens: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            embedding = self.model(input_ids=tokens, attention_mask=attention_mask)[0]
        embedding = torch.mean(embedding,1)
        return embedding

    def parse_fasta_input(self, input_file: Path) -> Dict[str, str]:
        fasta_dict = {}
        inputs = Fasta(str(input_file))
        for key in inputs.keys():
            fasta_dict[key] = inputs[key]

        return fasta_dict


def collate_paired_sequences(data: List[Tuple[torch.Tensor, int]]) -> Tuple[torch.Tensor, torch.Tensor]:
    tensors=[x[0] for x in data]
    ints =[x[1] for x in data]
    stacked_tensors=torch.stack(tensors)
    int_tensor = torch.as_tensor(ints)
    int_tensor= int_tensor.view([-1,1])
    print((stacked_tensors,int_tensor))
    return (stacked_tensors,int_tensor)


def get_dataloader(fasta_path: Path, labels_dict: Dict[str, int], batch_size: int, max_num_residues: int,
                   protbert_cache: Path, device: torch.device, seed: int) -> DataLoader:
    torch.manual_seed(seed)
    tokenDataset = TokenSeqDataset(fasta_path,labels_dict,max_num_residues,protbert_cache, device)
    dataloader =DataLoader(dataset=tokenDataset,batch_size=batch_size,shuffle=True,collate_fn=collate_paired_sequences)
    return dataloader

collate_paired_sequences([[torch.Tensor([2, 1, 3.5]), 1], [torch.Tensor([4, 4.5, 2.1]), 0], [torch.tensor([5, 5, 9]), 1]])