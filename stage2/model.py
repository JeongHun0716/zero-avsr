import sys,logging
import os
import stage1

import contextlib
from argparse import Namespace


import torch
import torch.nn as nn
from dataclasses import dataclass, field
from fairseq import checkpoint_utils, tasks
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.models import FairseqEncoder, FairseqEncoderModel, register_model, BaseFairseqModel, FairseqEncoderDecoderModel
from typing import Any, Optional
from fairseq import utils
from fairseq.modules import LayerNorm
import math
import numpy as np
from fairseq.dataclass import FairseqDataclass
from omegaconf import II, MISSING

from pathlib import Path
from transformers import  AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model


import torch.nn.functional as F
from avhubert.hubert_asr import AVHubertAsrConfig, HubertEncoderWrapper
from transformers import WhisperProcessor, WhisperForConditionalGeneration, WhisperModel
import stage1
from torch.nn.utils.rnn import pad_sequence


logger = logging.getLogger(__name__)

@dataclass
class Zero_AVSR_ModelConfig(AVHubertAsrConfig):
    llm_path: str = field(
        default='meta-llama/Llama-3.2-3B'
    )
    av_romanizer_path: str = field(
        default='default'
    )
    target_modules: str = field(
        default='q_proj.v_proj.k_proj.o_proj'
    )
    av_romanizer_embed_dim: int = field(
        default=1024, metadata={"help": "avhubert embedding dimension"}
    )
    llama_embed_dim: int = field(
        default=3072, metadata={"help": "llama embedding dimension"}
    )
    lora_rank: int = field(
        default=16, metadata={"help": "lora_rank"}
    )
    lora_alpha: int = field(
        default=32, metadata={"help": "lora_alpha"}
    )
    modality_fuse: str = field(
        default='concat', metadata={'help': 'fusing two modalities: concat, add, cross-att'}
    )
    use_speech_emb: bool = field(
        default=False, metadata={'help': 'Use Audio-Visual Speech Embedding as LLM input'}
    )
    use_roman_tok: bool = field(
        default=True, metadata={'help': 'Use Roman tokens as LLM inputs'}
    )
    back_trans_prob: float = field(
        default=0.3,
    )


class Projector(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Projector, self).__init__()
        # create a list of layers
        self.layers = nn.ModuleList([
            nn.Linear(in_features=input_dim, out_features=hidden_dim),
            nn.Linear(in_features=hidden_dim, out_features=output_dim)
        ])
    
    def forward(self, x):
        # iterate through all the layers
        for layer in self.layers:
            x = layer(x)
        return x         
          
          
@register_model("zero-avsr", dataclass=Zero_AVSR_ModelConfig)
class Zero_AVSR(BaseFairseqModel):    
    def __init__(self, av_romanizer, llm, tokenizer, cfg):
        super().__init__() 
        self.cfg = cfg
        self.av_romanizer = av_romanizer
        self.llama = llm   
        self.tokenizer = tokenizer
        self.blank = 0

        self.use_roman_tok = cfg.use_roman_tok
        self.use_speech_emb = cfg.use_speech_emb
        self.back_trans_prob = cfg.back_trans_prob


        for param in self.av_romanizer.parameters():
            param.requires_grad = False

        self.av_feat_1d_conv = nn.Conv1d(in_channels=cfg.av_romanizer_embed_dim, out_channels=cfg.av_romanizer_embed_dim, kernel_size=2, stride=2, padding=0)
        self.avfeat_to_llm = Projector(input_dim=cfg.av_romanizer_embed_dim,
                                        hidden_dim=math.floor((cfg.av_romanizer_embed_dim + cfg.llama_embed_dim)/2),
                                        output_dim=cfg.llama_embed_dim)
    

        self.freeze_finetune_updates = cfg.freeze_finetune_updates
        self.freeze_params = [n for n,p in self.named_parameters() if p.requires_grad == False]
        
        
    @classmethod
    def build_model(cls, cfg, task):
        ## load av-romanizer ##
        models, _, _ = checkpoint_utils.load_model_ensemble_and_task([cfg.av_romanizer_path])
        av_romanizer = models[0]
        ## load llm ##
        llm_model_id = cfg.llm_path  
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        llm = AutoModelForCausalLM.from_pretrained(llm_model_id, 
                                                   quantization_config=bnb_config,
                                                   low_cpu_mem_usage=True
                                                   )
        
        
        target_modules = cfg.target_modules.split('.')
        
        config = LoraConfig(
            r=cfg.lora_rank, 
            lora_alpha=cfg.lora_alpha, 
            target_modules=target_modules, 
            lora_dropout=0.05, 
            bias="none", 
            task_type="CAUSAL_LM" 
        )

        llm = get_peft_model(llm, config)
        llm.print_trainable_parameters()

        
        tokenizer = AutoTokenizer.from_pretrained(llm_model_id)

        return cls(av_romanizer, llm.base_model.model, tokenizer, cfg)
    

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        return state_dict

    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        super().set_num_updates(num_updates)
        self.num_updates = num_updates
        
    def state_dict(self):
        old_state = super().state_dict()
        state = {k:v for k,v in old_state.items() if k not in self.freeze_params}
        return state
    
    def load_state_dict(self,state,**kwargs):
        super().load_state_dict(state, strict=False)   
        
    def get_roman_tokens(self, roman_emissions):
        roman_toks_list = []
        roman_toks = roman_emissions.argmax(dim=-1) # B x T x C
        roman_toks_list = []
        for i in range(roman_toks.size(0)):
            roman_tok = roman_toks[i].unique_consecutive()
            roman_tok = roman_tok[roman_tok != self.blank]
            roman_tok = self.av_romanizer.roman_dict.string(roman_tok).replace(" ", "").replace("|", " ").strip()
            roman_toks_list.append(roman_tok)
        return roman_toks_list
        
        
    def prepare_inputs_labels_for_roman_tokens(self, instructions, roman_tokens, labels=None, zero_shot_samples=None):
        llm_input_list = []
        llm_labels_list = []
        for i in range(len(instructions)):
            if zero_shot_samples is not None and zero_shot_samples[i]:
                continue
            
            instruction = instructions[i]
            roman_token = self.tokenizer(roman_tokens[i], return_tensors="pt").input_ids[0][1:].to(instruction.device)
            
            if labels is not None:
                label = labels[i]
                combined = torch.cat([instruction, roman_token, label])
            else:
                combined = torch.cat([instruction, roman_token])
                
            llm_input_list.append(combined)
        
            if labels is not None:
                label_mask = torch.full(combined.size(), -100, dtype=combined.dtype, device=combined.device)
                offset = instruction.size(0) + roman_token.size(0)
                label_mask[offset:] = label
                llm_labels_list.append(label_mask)
            
        pad_value = self.tokenizer("<|finetune_right_pad_id|>").input_ids[1]
        llm_inputs = pad_sequence(llm_input_list, batch_first=True, padding_value=pad_value)
        attention_mask = (llm_inputs != pad_value).long()
        
        llm_inputs = self.llama.model.embed_tokens(llm_inputs)
        
        if labels is not None:
            llm_labels = pad_sequence(llm_labels_list, batch_first=True, padding_value=-100)
        else:
            llm_labels = None
            
        return llm_inputs, attention_mask, llm_labels
     
     
     
    def prepare_inputs_labels_for_speech_embs(self, instructions, speech_embs, speech_embs_lens, labels=None, zero_shot_samples=None):
        llm_input_list = []
        llm_labels_list = []
        for i in range(len(instructions)):
            if zero_shot_samples is not None and zero_shot_samples[i]:
                continue
            
            instruction = instructions[i]
            
            speech_emb_len = speech_embs_lens[i].item()
            speech_emb = speech_embs[i][:speech_emb_len,:]
 
            inst_emb = self.llama.model.embed_tokens(instruction.unsqueeze(0)).squeeze(0)

              
            if labels is not None:
                label = labels[i]
                label_emb = self.llama.model.embed_tokens(label.unsqueeze(0)).squeeze(0)    
                combined = torch.cat([inst_emb, speech_emb, label_emb], dim=0)
            else:
                combined = torch.cat([inst_emb, speech_emb], dim=0)
                
            llm_input_list.append(combined)
        
            if labels is not None:
                label_mask = torch.full((combined.size(0),), -100, dtype=instruction.dtype, device=instruction.device)
                offset = inst_emb.size(0) + speech_emb.size(0)
                label_mask[offset:] = label
                llm_labels_list.append(label_mask)
            
        pad_value = 0.0  
        llm_inputs = pad_sequence(llm_input_list, batch_first=True, padding_value=pad_value)
        attention_mask = (llm_inputs.abs().sum(dim=-1) != 0).long() 
        
        pad_token_id = self.tokenizer("<|finetune_right_pad_id|>").input_ids[1]
        pad_token_tensor = torch.tensor([pad_token_id], device=llm_inputs.device)
        pad_embedding = self.llama.model.embed_tokens(pad_token_tensor).squeeze(0)  
        llm_inputs[attention_mask == 0] = pad_embedding

        if labels is not None:
            llm_labels = pad_sequence(llm_labels_list, batch_first=True, padding_value=-100)
        else:
            llm_labels = None
            
        return llm_inputs, attention_mask, llm_labels
      
    def forward(self, **kwargs):
        with torch.no_grad():
            output, roman_emissions = self.av_romanizer.extract_features(**kwargs) # T x B x C
    
        instructions = kwargs['source']['instruction'] # (List[torch.Tensor]), B
        labels = kwargs['target_list'] # (List[torch.Tensor]), B
        zero_shot_samples = kwargs['source']['zero_shot_samples']
        
        batch_size = len(labels)
        
        task_prob = np.random.random()
        gt_roman_toks = kwargs['source']['roman_sources'] # GT roman tokens
        
        padding_mask = output['padding_mask'][:, 1::2]
        padding_mask = (~padding_mask).long()
        speech_embs_lens = padding_mask.sum(dim=-1)

        non_en_gt_roman_toks = []
        non_en_labels = []
        non_en_instructions = []
        
        langs = kwargs['source']['langs']
        
        for idx, lang in enumerate(langs):
            if lang != 'English':
                non_en_labels.append(labels[idx])
                non_en_gt_roman_toks.append(gt_roman_toks[idx])
                non_en_instructions.append(instructions[idx])
                
        if non_en_labels == []:
            task_prob = 1
        
        if task_prob <= self.back_trans_prob :
            llm_inputs, attention_mask, llm_labels = self.prepare_inputs_labels_for_roman_tokens(non_en_instructions, non_en_gt_roman_toks, non_en_labels)
        
        else:
            if zero_shot_samples.sum().item() != batch_size:
                speech_embs = output['encoder_out'] # B, T, C , Embedding
                speech_embs = self.av_feat_1d_conv(speech_embs.transpose(1, 2)).transpose(1, 2) # B x T x C - > B x T/2 x C (12.5Hz)
                speech_embs = self.avfeat_to_llm(speech_embs) # B x T/2 x C -> B x T/2 x C' (llm_embed_dim)
                llm_inputs, attention_mask, llm_labels = self.prepare_inputs_labels_for_speech_embs(instructions, speech_embs, speech_embs_lens, labels, zero_shot_samples)
            else:
                llm_inputs, attention_mask, llm_labels = self.prepare_inputs_labels_for_roman_tokens(instructions, gt_roman_toks, labels)


        llm_out = self.llama(inputs_embeds=llm_inputs, attention_mask=attention_mask, labels=llm_labels, return_dict=True, use_cache=False)

        loss = llm_out.loss
        logits = llm_out.logits
        
        return loss, logits, llm_labels


    def prepare_inputs_for_generation(self, instructions, speech_embs, speech_embs_lens):
        llm_input_list = []
        for i in range(len(instructions)):
            instruction = instructions[i]
            speech_emb_len = speech_embs_lens[i].item()
            speech_emb = speech_embs[i][:speech_emb_len, :]

            inst_emb = self.llama.model.embed_tokens(instruction.unsqueeze(0)).squeeze(0)
            combined = torch.cat([inst_emb, speech_emb], dim=0)
            llm_input_list.append(combined)
        

        max_length = max(seq.size(0) for seq in llm_input_list)
        

        pad_token_id = self.tokenizer("<|finetune_left_pad_id|>").input_ids[1]
        pad_token_tensor = torch.tensor([pad_token_id], device=llm_input_list[0].device)
        pad_embedding = self.llama.model.embed_tokens(pad_token_tensor).squeeze(0)

        padded_inputs = []
        attention_masks = []
        for seq in llm_input_list:
            seq_len = seq.size(0)
            pad_length = max_length - seq_len

            if pad_length > 0:
                pad_tensor = pad_embedding.unsqueeze(0).repeat(pad_length, 1)
                padded_seq = torch.cat([pad_tensor, seq], dim=0)
                mask = torch.cat([
                    torch.zeros(pad_length, dtype=torch.long, device=seq.device),
                    torch.ones(seq_len, dtype=torch.long, device=seq.device)
                ])
            else:
                padded_seq = seq
                mask = torch.ones(seq_len, dtype=torch.long, device=seq.device)
            
            padded_inputs.append(padded_seq)
            attention_masks.append(mask)
        
        llm_inputs = torch.stack(padded_inputs, dim=0)
        attention_mask = torch.stack(attention_masks, dim=0)
        
        return llm_inputs, attention_mask


    @torch.no_grad()
    def generate(self,
                num_beams=2,
                temperature=0.6,
                max_length=100,
                min_length=1,
                repetition_penalty=1.0,
                length_penalty=0.0,
                use_speech_embs=False,
                use_roman_toks=False,
                  **kwargs,
                ):
        output, roman_emissions = self.av_romanizer.extract_features(**kwargs) # T x B x C
        instructions = kwargs['source']['instruction'] # (List[torch.Tensor]), B

        B, T, D = output['encoder_out'].size()  # B=1, T=136, D=1024

        padding_mask = output['padding_mask'][:, 1::2]
        padding_mask = (~padding_mask).long()
        speech_embs_lens = padding_mask.sum(dim=-1)


        speech_embs = output['encoder_out'] # B, T, C , Embedding
        speech_embs = self.av_feat_1d_conv(speech_embs.transpose(1, 2)).transpose(1, 2) # B x T x C - > B x T/2 x C (12.5Hz)
        speech_embs = self.avfeat_to_llm(speech_embs) # B x T/2 x C -> B x T/2 x C' (llm_embed_dim)
        llm_inputs, attention_mask = self.prepare_inputs_for_generation(instructions, speech_embs, speech_embs_lens)


        self.llama.generation_config.pad_token_id = self.llama.generation_config.eos_token_id

        outputs = self.llama.generate(inputs_embeds=llm_inputs,
                                    attention_mask=attention_mask,
                                    num_beams=num_beams,
                                    temperature=temperature,  
                                    max_new_tokens=max_length, 
                                    min_length=min_length,
                                    )

        return outputs
        
        
def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m
