import torch
import torch.nn as nn
import numpy as np
from typing import Optional
from transformers import GenerationMixin
from torch import nn
from typing import Optional
from dataclasses import dataclass
from transformers.modeling_outputs import ModelOutput
from vq import RQVAE
from layers import *
from fusion_layers import MultiLayerInterestFusion, AdaptiveInterestFusion
from sentence_transformers import SentenceTransformer
from utils.utils import get_sentence_transformer_config


@dataclass
class QuantizeOutput(ModelOutput):
    logits: Optional[torch.FloatTensor] = None
    rank_logits: Optional[torch.FloatTensor] = None
    seq_latents: Optional[torch.FloatTensor] = None
    seq_project_latents: Optional[torch.FloatTensor] = None
    dec_latents: Optional[torch.FloatTensor] = None
        
        
class Model(nn.Module, GenerationMixin):
    def __init__(self, config, model, n_items, code_length=1, code_number=256):
        super().__init__()
        self.model = model
        self._supports_cache_class = model._supports_cache_class
        self.config = model.config
        self.base_model_prefix = "model"
        self.generation_config = model.generation_config
        self.main_input_name = model.main_input_name
        self.get_encoder = model.get_encoder
        self.device = model.device
        self.can_generate = lambda: True

        self.hidden_size = model.config.hidden_size
        self.semantic_hidden_size = config.get('semantic_hidden_size')
        self.n_items = n_items
        self.code_length = code_length
        self.code_number = code_number
        self.num_beams = config['num_beams']
        
        self.semantic_embedding = nn.Embedding(self.n_items, self.semantic_hidden_size)
        self.semantic_embedding.requires_grad_(False)
        
        self.token_embeddings = nn.ModuleList([nn.Embedding(self.code_number, self.hidden_size) for i in range(self.code_length)])
        self.token_embeddings.requires_grad_(True)
        
        enc_adapter_layers = config['layers']
        enc_adapter_layers = [self.hidden_size] + [config['e_dim']]
        self.enc_adapter = MLPLayers(layers=enc_adapter_layers)

        dec_adapter_layers = config['layers'][::-1]
        dec_adapter_layers = [self.hidden_size] + [self.semantic_hidden_size]
        self.dec_adapter = MLPLayers(layers=dec_adapter_layers)

        # Get SentenceTransformer configuration
        model_name, embedding_dim, batch_size, max_seq_length = get_sentence_transformer_config(config)
        
        self.interest_encoder = SentenceTransformer(model_name, device=self.device)
        self.interest_encoder.eval()
        self.interest_proj = nn.Linear(embedding_dim, config['e_dim'])
        
        
        fusion_type = config.get('fusion_type', 'simple')  # 'simple', 'transformer', 'adaptive'
        self.num_fusion_layers = config.get('num_fusion_layers', 3)
        
        if fusion_type == 'simple':
            # Multiple layers of MultiheadAttention
            self.interest_fusion_layers = nn.ModuleList([
                nn.MultiheadAttention(
                    embed_dim=config['e_dim'], 
                    num_heads=4,
                    batch_first=True,
                    add_bias_kv=False
                ) for _ in range(self.num_fusion_layers)
            ])
            self.fusion_layer_norms = nn.ModuleList([
                nn.LayerNorm(config['e_dim']) for _ in range(self.num_fusion_layers)
            ])
            self.fusion_type = 'simple'
            
        elif fusion_type == 'transformer':
            # Transformer Encoder block for interest fusion
            self.interest_fusion = MultiLayerInterestFusion(
                embed_dim=config['e_dim'],
                num_layers=self.num_fusion_layers,
                num_heads=8,
                dropout=config.get('fusion_dropout', 0.1),
                fusion_strategy=config.get('fusion_strategy', 'weighted')  # 'last', 'weighted', 'concat'
            )
            self.fusion_type = 'transformer'
            
        elif fusion_type == 'adaptive':
            # Adaptive Interest Fusion
            self.interest_fusion = AdaptiveInterestFusion(
                embed_dim=config['e_dim'],
                num_layers=self.num_fusion_layers,
                num_heads=8,
                dropout=config.get('fusion_dropout', 0.1)
            )
            self.fusion_type = 'adaptive'
            
        else:
            # Single layer MultiheadAttention as default
            self.interest_fusion = nn.MultiheadAttention(
                embed_dim=config['e_dim'], 
                num_heads=8,
                batch_first=True,
                add_bias_kv=False
            )
            self.fusion_type = 'single'
        
        # Interest cache for faster encoding
        self.interests_cache = None
        
        # parameters initialization
        self.apply(self._init_weights)
    
    def set_interests_cache(self, interests_cache):
        """Set the interests cache for faster encoding during training."""
        self.interests_cache = interests_cache
        print(f"Interests cache set with {len(interests_cache)} entries")

    def _init_weights(self, module):
        """Initialize the weights"""

        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, encoder_outputs=None, **kwargs):
        return {"decoder_input_ids": input_ids, "encoder_outputs": encoder_outputs, "attention_mask": attention_mask}

    def _shift_right(self, input_ids):
        pad_token_id = self.config.pad_token_id

        shifted_input_ids = torch.full(input_ids.shape[:-1] + (1,), pad_token_id, device=input_ids.device)
        shifted_input_ids = torch.cat([shifted_input_ids, input_ids], dim=-1)

        return shifted_input_ids
    
    def get_input_embeddings(self, input_ids, attention_mask):
        attention_mask_flatten = attention_mask.reshape(-1)

        inputs_embeds = torch.zeros(*input_ids.shape, self.hidden_size, device=self.device)
        input_ids[input_ids==-1] = 0
        for i in range(self.code_length):
            inputs_embeds[:, i::self.code_length] = self.token_embeddings[i](input_ids[:, i::self.code_length])
        
        inputs_embeds = inputs_embeds.view(-1, self.hidden_size)
        inputs_embeds[~attention_mask_flatten] = self.model.shared.weight[0]
        inputs_embeds = inputs_embeds.view(input_ids.shape[0], -1, self.hidden_size)

        return inputs_embeds
    
    def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None, labels=None, decoder_input_ids=None,
                decoder_inputs_embeds=None, encoder_outputs=None, interests=None, **kwargs):
        
        if input_ids is not None:
            inputs_embeds = self.get_input_embeddings(input_ids, attention_mask)

        if decoder_input_ids is None and labels is None:
            decoder_input_ids = torch.zeros(input_ids.size(0), self.code_length).long().to(input_ids.device)
        elif decoder_input_ids is None and labels is not None:
            decoder_input_ids = self._shift_right(labels)

        if decoder_inputs_embeds is None and decoder_input_ids is not None:
            decoder_inputs_embeds = []
            for i in range(min(decoder_input_ids.shape[1], self.code_length)):
                if i==0:
                    code_embedding = self.model.shared
                else:
                    code_embedding = self.token_embeddings[i-1]  # 0~255
                decoder_inputs_embeds.append(code_embedding(decoder_input_ids[:, i]))
            decoder_inputs_embeds = torch.stack(decoder_inputs_embeds, dim=1)


        model_outputs = self.model(
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            output_hidden_states=True,
            encoder_outputs=encoder_outputs
        )

        decoder_outputs = model_outputs.decoder_hidden_states[-1]

        code_logits = []
        for i in range(min(decoder_inputs_embeds.shape[1], self.code_length)):
            centroid = self.token_embeddings[i].weight.t()
            code_logits.append(torch.matmul(decoder_outputs[:, i], centroid))
        
        code_logits = torch.stack(code_logits, dim=1) # (batch, code_len, code_num)
        
        seq_latents = model_outputs.encoder_last_hidden_state.clone()
        # mean pooling
        seq_latents[~attention_mask] = 0
        seq_last_latents = torch.sum(seq_latents, dim=1) / attention_mask.sum(dim=1).unsqueeze(1)
        # seq_project_latents = self.enc_adapter(seq_last_latents)

        # fusion with interests
        if interests is None:
            interests = [""] * input_ids.shape[0]

        # if all(interest.strip() == "" for interest in interests):
        #     print("Warning: All interests are empty strings. Using empty interests embedding.")
        
        # Use cached embeddings if available
        if self.interests_cache is not None:
            interests_embed = []
            for interest in interests:
                if interest in self.interests_cache:
                    interests_embed.append(self.interests_cache[interest])
                else:
                    # Fallback to real-time encoding for unknown interests
                    fallback_embed = self.interest_encoder.encode([interest], show_progress_bar=False)
                    interests_embed.append(torch.tensor(fallback_embed[0], device=self.device, dtype=torch.float32))
            interests_embed = torch.stack(interests_embed, dim=0).to(self.device)
        else:
            # Original real-time encoding
            interests_embed = self.interest_encoder.encode(interests, show_progress_bar=False)
            interests_embed = torch.tensor(interests_embed, device=self.device, dtype=torch.float32)
        
        interests_embed_proj = self.interest_proj(interests_embed)
        
        # 根据融合类型选择不同的融合策略
        if self.fusion_type == 'simple':
            # 方案1：简单多层融合
            query = interests_embed_proj
            key_value = seq_last_latents
            
            for i, (attn_layer, norm_layer) in enumerate(zip(self.interest_fusion_layers, self.fusion_layer_norms)):
                attention_output, _ = attn_layer(
                    query=query,
                    key=key_value,
                    value=key_value,
                )
                query = norm_layer(query + attention_output)
            
            seq_last_latents = seq_last_latents + query
            
        elif self.fusion_type == 'transformer':
            # 方案2：Transformer 编码器块式融合
            interests_expanded = interests_embed_proj.unsqueeze(1)  # (batch_size, 1, embed_dim)
            seq_expanded = seq_last_latents.unsqueeze(1)  # (batch_size, 1, embed_dim)
            
            fused_interests = self.interest_fusion(interests_expanded, seq_expanded)
            fused_interests = fused_interests.squeeze(1)  # (batch_size, embed_dim)
            
            seq_last_latents = seq_last_latents + fused_interests
            
        elif self.fusion_type == 'adaptive':
            # 方案3：自适应融合
            interests_expanded = interests_embed_proj.unsqueeze(1)  # (batch_size, 1, embed_dim)
            seq_last_latents = self.interest_fusion(interests_expanded, seq_last_latents)
            
        else:
            # 原始单层融合
            attention_output, _ = self.interest_fusion(
                query=interests_embed_proj,
                key=seq_last_latents,
                value=seq_last_latents,
            )
            seq_last_latents = seq_last_latents + attention_output

        seq_project_latents = self.enc_adapter(seq_last_latents)
        
        dec_latents = model_outputs.decoder_hidden_states[-1].clone()
        dec_latents = dec_latents[:,0,:]
        dec_latents = self.dec_adapter(dec_latents)
        
        outputs = QuantizeOutput(
            logits=code_logits,
            seq_latents=seq_last_latents,
            seq_project_latents=seq_project_latents,
            dec_latents=dec_latents
        )
        return outputs
    
    def generate(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, n_return_sequences: int = 1,
                 prefix_allowed_tokens_fn=None, interests=None) -> torch.Tensor:
        """
        Generates sequences using beam search algorithm.

        Args:
            batch (dict): A dictionary containing input_ids and attention_mask.
            n_return_sequences (int): The number of sequences to generate.
            interests: User interests to be passed to the model.

        Returns:
            torch.Tensor: The generated sequences.
        """
        # Store interests temporarily for use in forward pass
        self._current_interests = interests
        
        if prefix_allowed_tokens_fn is not None:
            inputs_embeds = self.get_input_embeddings(input_ids, attention_mask)
            outputs = super().generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_length=self.code_length+1,
                num_beams=self.num_beams,
                num_return_sequences=n_return_sequences,
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn
            )
        else:
            outputs = self.my_beam_search(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=self.code_length+1,
                num_beams=self.num_beams,
                num_return_sequences=n_return_sequences,
                return_score=False,
                interests=interests
            )
        
        # Clear stored interests
        self._current_interests = None
        
        outputs = outputs[:, 1:].reshape(-1, n_return_sequences, self.code_length)
        return outputs

    def my_beam_search(
        self,
        input_ids,
        attention_mask,
        max_length=6,
        num_beams=1,
        num_return_sequences=1,
        return_score=False,
        interests=None
    ):
        """
        Adpated from huggingface's implementation
        https://github.com/huggingface/transformers/blob/v4.39.3/src/transformers/generation/utils.py#L2823

        Perform beam search to generate sequences using the specified model. 

        *** This implementation does not include stopping conditions based on end-of-sequence (EOS) tokens. Instead, the
        sequence generation is controlled solely by the `max_length` parameter. ***

        Note: In scenarios where the generation should explicitly detect and respond to EOS tokens 
        to terminate the sequence early, this function would need modifications. In the current setup,
        setting `max_length` to a suitable fixed value (e.g., 6) can serve the purpose by limiting
        the maximum sequence length.

        Parameters:
        - input_ids (torch.Tensor): Tensor of input ids.
        - attention_mask (torch.Tensor): Tensor representing the attention mask.
        - max_length (int): Maximum length of the sequence to be generated; controls when to stop extending the sequence.
        - num_beams (int): Number of beams for beam search.
        - num_return_sequences (int): Number of sequences to return.
        - return_score (bool): If True, returns a tuple of (sequences, scores) where 'scores' are the average log likelihood of the returned sequences.

        Returns:
        - torch.Tensor: The final decoder input ids from the beam search, or a tuple of (decoder_input_ids, scores) if 'return_score' is True.

        Example usage:
        # Assuming the model, input_ids, and attention_mask are predefined:
        sequences = beam_search(model, input_ids, attention_mask, max_length=6, num_beams=5, num_return_sequences=5)
        """

        batch_size = input_ids.shape[0]

        # Prepare beam search inputs
        input_ids, attention_mask, decoder_input_ids, beam_scores, beam_idx_offset = \
            self.prepare_beam_search_inputs(
                input_ids, attention_mask, batch_size, num_beams
            )
        
        # Expand interests to match beam search expansion
        if interests is not None:
            if isinstance(interests, list):
                # For list of strings, repeat each interest for each beam
                expanded_interests = []
                for interest in interests:
                    expanded_interests.extend([interest] * num_beams)
                interests = expanded_interests
            else:
                # For tensor, use repeat_interleave
                interests = interests.repeat_interleave(num_beams, dim=0)
        
        inputs_embeds = self.get_input_embeddings(input_ids, attention_mask)

        # Store encoder_outputs to prevent running full forward path repeatedly
        with torch.no_grad():
            encoder_outputs = self.get_encoder()(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True
            )

        # Beam search loop
        while decoder_input_ids.shape[1] < max_length:
            with torch.no_grad():
                outputs = self.forward(
                    encoder_outputs=encoder_outputs,
                    attention_mask=attention_mask,
                    decoder_input_ids=decoder_input_ids,
                    interests=interests
                )

            decoder_input_ids, beam_scores = self.beam_search_step(
                outputs.logits,
                decoder_input_ids,
                beam_scores,
                beam_idx_offset,
                batch_size,
                num_beams
            )

        # (batch_size * num_beams, ) -> (batch_size * num_return_sequences, )
        selection_mask = torch.zeros(batch_size, num_beams, dtype=torch.bool)
        selection_mask[:, :num_return_sequences] = True

        if return_score:
            return decoder_input_ids[selection_mask.view(-1), :], \
                beam_scores[selection_mask.view(-1)] / (decoder_input_ids.shape[1] - 1)

        return decoder_input_ids[selection_mask.view(-1), :]

    def prepare_beam_search_inputs(self, input_ids, attention_mask, batch_size, num_beams):
        """
        Adpated from huggingface's implementation
        https://github.com/huggingface/transformers/blob/v4.39.3/src/transformers/generation/utils.py#L2823

        Prepares and duplicates the input data for beam search decoding.

        This function initializes decoder input IDs and beam scores, creates an offset for beam indices, 
        and expands the input_ids and attention_mask tensors to accommodate the specified number of beams for each instance in the batch.

        Parameters:
        - input_ids (torch.Tensor): The input IDs tensor of shape (batch_size, sequence_length) used for the encoder part of the model.
        - attention_mask (torch.Tensor): The attention mask tensor of shape (batch_size, sequence_length) indicating to the model which tokens should be attended to.
        - batch_size (int): The number of instances per batch in the input data.
        - num_beams (int): The number of beams to use in beam search. This expands the input data and scores accordingly.

        Returns:
        - input_ids (torch.Tensor): The expanded input IDs tensor to match the number of beams, shape (batch_size * num_beams, sequence_length).
        - attention_mask (torch.Tensor): The expanded attention mask tensor to match the number of beams, shape (batch_size * num_beams, sequence_length).
        - initial_decoder_input_ids (torch.Tensor): The initialized decoder input IDs for each beam, shape (batch_size * num_beams, 1).
        - initial_beam_scores (torch.Tensor): The initialized scores for each beam, flattened to a single dimension, shape (batch_size * num_beams,).
        - beam_idx_offset (torch.Tensor): An offset for each beam index to assist in reordering beams during the search, shape (batch_size * num_beams,).

        Each input sequence is replicated 'num_beams' times to provide separate candidate paths in beam search. Beam scores are initialized with 0 for the first beam and a very low number (-1e9) for others to ensure the first token of each sequence is chosen from the first beam.
        """

        decoder_input_ids = torch.ones((batch_size * num_beams, 1), device=self.device, dtype=torch.long)
        initial_decoder_input_ids = decoder_input_ids * self.config.decoder_start_token_id

        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
        beam_scores[:, 1:] = -1e9  # Set a low score for all but the first beam to ensure the first beam is selected initially
        initial_beam_scores = beam_scores.view((batch_size * num_beams,))

        beam_idx_offset = torch.arange(batch_size, device=self.device).repeat_interleave(num_beams) * num_beams

        input_ids = input_ids.repeat_interleave(num_beams, dim=0)
        attention_mask = attention_mask.repeat_interleave(num_beams, dim=0)

        return input_ids, attention_mask, initial_decoder_input_ids, initial_beam_scores, beam_idx_offset


    def beam_search_step(self, logits, decoder_input_ids, beam_scores, beam_idx_offset, batch_size, num_beams):
        """
        Adpated from huggingface's implementation
        https://github.com/huggingface/transformers/blob/v4.39.3/src/transformers/generation/utils.py#L2823

        Executes one step of beam search, calculating the next set of input IDs based on logits from a model.

        This function expands the current beam, calculates scores for all possible next tokens, selects the top tokens for each beam, and prepares the input IDs for the next iteration of the model. It utilizes logits output by the model to determine the most likely next tokens and updates the beam scores.

        Parameters:
        - logits (torch.Tensor): Logits returned from the model, shape (batch_size * num_beams, sequence_length, vocab_size).
        - decoder_input_ids (torch.Tensor): Current decoder input IDs, shape (batch_size * num_beams, current_sequence_length).
        - beam_scores (torch.Tensor): Current scores for each beam, shape (batch_size * num_beams,).
        - beam_idx_offset (torch.Tensor): Index offsets for each beam to handle batches correctly, shape (batch_size * num_beams,).
        - batch_size (int): Number of sequences being processed in a batch.
        - num_beams (int): Number of beams used in the beam search.

        Returns:
        - decoder_input_ids (torch.Tensor): Updated decoder input IDs after adding the next tokens, shape (batch_size * num_beams, current_sequence_length + 1).
        - beam_scores (torch.Tensor): Updated scores for each beam, shape (batch_size * num_beams,).

        The function selects the top `2 * num_beams` tokens from the logits based on their scores, reshapes and adjusts them based on the existing beam scores, and determines the next tokens to add to each beam path. The updated paths are then returned for use in the next iteration of the beam search.
        """
        assert batch_size * num_beams == logits.shape[0]

        vocab_size = logits.shape[-1]
        next_token_logits = logits[:, -1, :]
        next_token_scores = torch.log_softmax(next_token_logits, dim=-1)  # Calculate log softmax over the last dimension

        next_token_scores = next_token_scores + beam_scores[:, None].expand_as(next_token_scores)
        next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)
        next_token_scores, next_tokens = torch.topk(next_token_scores, 2 * num_beams, dim=1, largest=True, sorted=True)

        next_indices = torch.div(next_tokens, vocab_size, rounding_mode="floor")
        next_tokens = next_tokens % vocab_size

        beam_scores = next_token_scores[:, :num_beams].reshape(-1)
        beam_next_tokens = next_tokens[:, :num_beams].reshape(-1)
        beam_idx = next_indices[:, :num_beams].reshape(-1)

        # beam_idx_offset: beam_idx contains sequence indicies relative to each individual batch. We need to offset the indicies to retrieve the correct sequence in the corresponding batch
        # for example, when batch_size = 2, beam_size = 3, beam_idx_offset = [0, 0, 0, 3, 3, 3]
        decoder_input_ids = torch.cat([decoder_input_ids[beam_idx + beam_idx_offset, :], beam_next_tokens.unsqueeze(-1)], dim=-1)

        return decoder_input_ids, beam_scores

    