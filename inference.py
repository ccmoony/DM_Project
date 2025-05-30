import torch
import json
from typing import List
from accelerate import PartialState
from model import Model
from utils.utils import load_json, safe_load
import os
from transformers import T5Config, T5ForConditionalGeneration
import numpy as np
from vq import RQVAE
import yaml

class Recommender:
    def __init__(self, config_path, model_path: str, code_path: str, rqvae_path: str, device: str = None):
        """
        Initialize the predictor with model and code mappings
        
        Args:
            model_path: Path to the trained model checkpoint
            code_path: Path to the code json file
            device: Target device (cuda/cpu), auto-detect if None
        """
        config = yaml.safe_load(open(config_path, 'r'))
        self.state = PartialState()
        self.device = device
        
        # Load model components
        model_config = T5Config(
            num_layers=config['encoder_layers'], 
            num_decoder_layers=config['decoder_layers'],
            d_model=config['d_model'],
            d_ff=config['d_ff'],
            num_heads=config['num_heads'],
            d_kv=config['d_kv'],
            dropout_rate=config['dropout_rate'],
            activation_function=config['activation_function'],
            vocab_size=1,
            pad_token_id=0,
            eos_token_id=300,
            decoder_start_token_id=0,
            feed_forward_proj=config['feed_forward_proj'],
            n_positions=config['max_length'],
        )

        code_num = config['code_num']
        code_length = config['code_length']
        
        data_path = config["data_path"]
        dataset = config["dataset"]
        dataset_path = os.path.join(data_path, f"{dataset}/{dataset}")
        semantic_emb_path = os.path.join(data_path, dataset, config["semantic_emb_path"])
        map_path = dataset_path + config["map_path"]
        self.item2id = load_json(map_path)
        num_items = len(self.item2id)
        self.id2item = {v: k for k, v in self.item2id.items()}

        t5 = T5ForConditionalGeneration(config=model_config).to(self.device)
        self.model_rec = Model(config=config, model=t5, n_items=num_items,
                  code_length=code_length, code_number=code_num).to(self.device)
        
        semantic_emb = np.load(semantic_emb_path)
            
        self.model_rec.semantic_embedding.weight.data[1:] = torch.tensor(semantic_emb).to(self.device)
        self.model_id = RQVAE(config=config, in_dim=self.model_rec.semantic_hidden_size)
        
        # Load weights
        # rqvae_path = config.get('rqvae_path', None)
        safe_load(self.model_id, rqvae_path, verbose=True)
        safe_load(self.model_rec, model_path, verbose=True)
        
        # Load item codes
        with open(code_path, 'r') as f:
            self.item_codes = torch.tensor(json.load(f)).to(self.device)
        
        self.model_rec.eval()
        self.model_id.eval()

    def predict_next_item(self, item_sequence: List[str], interests = None, top_k: int = 8) -> List[int]:
        """
        Predict next items given a sequence of item IDs
        
        Args:
            item_sequence: List of item IDs (e.g., [123, 456, 789])
            top_k: Number of top predictions to return
            
        Returns:
            List of predicted item IDs ordered by likelihood
        """
        # Convert item IDs to codes
        item_sequence = [self.item2id[token] for token in item_sequence]
        input_codes = self.item_codes[item_sequence].unsqueeze(0).to(self.device) # [1, seq_len, code_len]
        input_codes = input_codes.contiguous().clone().view(1, -1)
        attention_mask = (input_codes != -1).bool().to(self.device)

        # Generate predictions
        with torch.no_grad():
            output_codes = self.model_rec.generate(
                input_ids=input_codes,
                attention_mask=attention_mask,
                interests=interests,
                n_return_sequences=top_k,
            )[0]  # [top_k, seq_len+1]
        
        # Convert predicted codes back to item IDs
        output_codes = output_codes.unsqueeze(1)  
        expanded_large = self.item_codes.unsqueeze(0)  

        matches = torch.all(output_codes == expanded_large, dim=2)

        # 找到每行第一个匹配的索引
        indices = torch.where(matches.any(dim=1), matches.int().argmax(dim=1), torch.tensor(-1)).tolist()
        pred_items = [self.id2item[idx] if idx != -1 else None for idx in indices]
        return pred_items  # Return at most top_k unique items


# if __name__ == "__main__":
    # 初始化预测器
    # predictor = Recommender(
    #     config_path = "/home/yjchen/workspace/homework/ETEGRec/config/scientific.yaml",
    #     model_path="/home/yjchen/workspace/homework/ETEGRec/myckpt/scientific/May-28-2025_15-05-20270c/83.pt",
    #     code_path="/home/yjchen/workspace/homework/ETEGRec/myckpt/scientific/May-28-2025_15-05-20270c/83.code.json",
    #     rqvae_path="/home/yjchen/workspace/homework/ETEGRec/myckpt/scientific/May-28-2025_15-05-20270c/83.pt.rqvae",
    #     device="cpu",
    # )

    # # 单个序列预测
    # sequence = ["B00DX7KEP8", "B098BR67DJ", "B08Y97BK7N", "B0B6YT7HNF", "B0883CDD2Z"]  # 历史商品ID序列
    # predictions = predictor.predict_next_item(sequence, top_k=10)
    # print(f"Top 10 predicted next items: {predictions}")