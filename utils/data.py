import os
import torch
from utils.utils import *
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import random
from sentence_transformers import SentenceTransformer
import numpy as np


def build_interests_cache(interests_dict, device='cuda'):
    """
    Build a cache for interests embeddings to avoid repeated encoding during training.
    
    Args:
        interests_dict: Dictionary containing interests for train/valid/test sets
        device: Device to load the sentence transformer on
        
    Returns:
        Dictionary mapping interests strings to their embeddings
    """
    print("Building interests cache...")
    
    # Collect all unique interests
    all_interests = set()
    for split in interests_dict.values():
        for interest in split:
            if interest and interest.strip():  # Skip empty interests
                all_interests.add(interest)
    
    all_interests = list(all_interests)
    print(f"Found {len(all_interests)} unique interests")
    
    # Initialize sentence transformer
    interest_encoder = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    interest_encoder.eval()
    
    # Encode all unique interests
    interests_embeddings = interest_encoder.encode(all_interests, show_progress_bar=True, batch_size=512)
    
    # Build cache dictionary
    interests_cache = {}
    for interest, embedding in zip(all_interests, interests_embeddings):
        interests_cache[interest] = torch.tensor(embedding, dtype=torch.float32)
    
    # Add empty string mapping
    empty_embedding = torch.zeros(384, dtype=torch.float32)  # 384 is the embedding dimension for all-MiniLM-L6-v2
    interests_cache[""] = empty_embedding
    
    print(f"Interests cache built with {len(interests_cache)} entries")
    return interests_cache


def load_split_data(config):
    def transform_token2id_seq(token_seqs, item2id):
        id_seqs = []
        for one_piece in token_seqs:
            item_token_seq = one_piece["inter_history"]
            item_id_seq = [item2id[token] for token in item_token_seq]
            target_id = item2id[one_piece["target_id"]]
            id_seqs.append(item_id_seq + [target_id])

        return id_seqs
            
    data_path = config["data_path"]
    dataset = config["dataset"]
    dataset_path = os.path.join(data_path, f"{dataset}/{dataset}")
    map_path = dataset_path + config["map_path"]

    if config["interest_fusion"]:
        processed = ".processed"
    else:
        processed = ""
    
    train_inter = load_jsonl(dataset_path + f".train{processed}.jsonl")
    valid_inter = load_jsonl(dataset_path + f".valid{processed}.jsonl")
    test_inter = load_jsonl(dataset_path + f".test{processed}.jsonl")

    item2id = load_json(map_path) # id start from 1, 2, ...
    
    train_seq = transform_token2id_seq(train_inter, item2id)
    valid_seq = transform_token2id_seq(valid_inter, item2id)
    test_seq = transform_token2id_seq(test_inter, item2id)

    interests_dict = None
    interests_cache = None

    if config["interest_fusion"]:
        interests_dict = {
            "train": [i["interests"] for i in train_inter],
            "valid": [i["interests"] for i in valid_inter],
            "test": [i["interests"] for i in test_inter],
        }
        
        # Build interests cache
        interests_cache = build_interests_cache(interests_dict, config.get('device', 'cpu'))
    
    n_items = len(item2id)

    return item2id, n_items, train_seq, valid_seq, test_seq, interests_dict, interests_cache
    
    
class SequentialSplitDataset(Dataset):
    def __init__(self, config, n_items, inter_seq, data_ratio=1, interests=None):
        self.n_items = n_items
        self.config = config

        if data_ratio < 1:
            # random sampling
            n_sample = int(len(inter_seq)*data_ratio)
            inter_seq = random.sample(inter_seq, n_sample)
            
        self.data = self.__map_inter__(inter_seq)

        self.interests = interests

    def __map_inter__(self, inter_seq):
        data = []

        for seq in inter_seq:
            target = seq[-1]
            dict_data = {"id_seq": seq[:-1], "target": [target]}
            data.append(dict_data)

        return data
            
    def __getitem__(self, idx):
        data = self.data[idx]
        id_seq = data['id_seq']
        target = data['target']
        interests = self.interests[idx] if self.interests else None
            
        return id_seq, target, interests

    def __len__(self):
        return len(self.data)
    
    
class Collator(object):
    def __init__(self, eos_token_id, pad_token_id, max_length):
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.max_length = max_length
    
    def __pad_seq__(self, seq):
        if len(seq) > self.max_length:
            return seq[-self.max_length+1:]
        return seq
    
    def __call__(self, batch):
        # Update to unpack 3 values from each item in batch
        id_seqs, targets, interests_list = zip(*batch)
        
        input_ids = [torch.tensor(self.__pad_seq__(id_seq)) for id_seq in id_seqs]
        input_ids = pad_sequence(input_ids).transpose(0, 1)
        input_ids = input_ids.to(torch.long)

        attention_mask = (input_ids != self.pad_token_id).bool()
        
                              
        targets = torch.tensor(targets)

        targets = targets.to(torch.long).contiguous()
        
        result_dict = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'targets': targets
        }
        
        # Add interests to result dict if they are not None and not all empty strings
        if any(interest is not None and interest != "" for interest in interests_list):
            # Since interests are string data, we don't need to convert to tensor
            # Just pass them through as a list
            result_dict['interests'] = interests_list
            
        return result_dict



