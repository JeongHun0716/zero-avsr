# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import sys,logging
import contextlib
import tempfile
from argparse import Namespace
from typing import Any, Optional, Tuple
import os

import torch.nn.functional as F

import torch
import torch.nn as nn
from dataclasses import dataclass, field
from fairseq import checkpoint_utils, tasks, utils
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.models import BaseFairseqModel, FairseqEncoder, FairseqEncoderDecoderModel, register_model
from fairseq.models.wav2vec.wav2vec2 import TransformerEncoder

from fairseq.models.hubert.hubert import MASKING_DISTRIBUTION_CHOICES
from fairseq.checkpoint_utils import load_model_ensemble_and_task
from fairseq.tasks import FairseqTask
from omegaconf import II, MISSING, open_dict


        
from avhubert.hubert_asr import AVHubertAsrConfig

#from avhubert.hubert import SubModel
from copy import deepcopy


DBG=True if len(sys.argv) == 1 else False
EXTRACTOR_MODE_CHOICES = ChoiceEnum(["default", "layer_norm"])
import pdb


logger = logging.getLogger(__name__)

EXTRACTOR_MODE_CHOICES = ChoiceEnum(["default", "layer_norm"])
MASKING_DISTRIBUTION_CHOICES = ChoiceEnum(
    ["static", "uniform", "normal", "poisson"]
)

@dataclass
class AV_RomanizerConfig(AVHubertAsrConfig):
    w2v_path: str = field(
        default=MISSING, metadata={"help": "path to hubert model"}
    )
    blank_weight: float = field(
        default=0.0, metadata={"help": "blank_weight"}
    )
    blank_mode: str = field(
        default='add', metadata={"help": "blank_mode"}
    )
    encoder_embed_dim: int = field(
        default=1024,
        metadata={"help": "encoder_embed_dim"},
    )
    label_rate: int = field(
        default=-1,
        metadata={"help": "label frame rate. -1 for sequence label"},
    )
    skip_masked: bool = field(
        default=False,
        metadata={"help": "skip computing losses over masked frames"},
    )
    skip_nomask: bool = field(
        default=False,
        metadata={"help": "skip computing losses over unmasked frames"},
    )
    modality_dropout: float = field(default=0, metadata={'help': 'drop one modality'})
    audio_dropout: float = field(default=0, metadata={'help': 'drop audio feature'})
    modality_fuse: str = field(default='concat', metadata={'help': 'fusing two modalities: add,concat'})
    selection_type : str = field(default='same_other_seq', metadata={'help': 'type of selectig images, same_other_seq: replace masked span with span from another sequence, same_seq: repace masked span with span of the same sequence'})
    masking_type : str = field(default='input', metadata={'help': 'input or feature masking'})
    
    
     # masking
    mask_length_audio: int = field(default=10, metadata={"help": "mask length"})
    mask_prob_audio: float = field(
        default=0.65,
        metadata={"help": "probability of replacing a token with mask"},
    )
    mask_length_image: int = field(default=10, metadata={"help": "mask length"})
    mask_prob_image: float = field(
        default=0.65,
        metadata={"help": "probability of replacing a token with mask"},
    )
    mask_selection: MASKING_DISTRIBUTION_CHOICES = field(
        default="static", metadata={"help": "how to choose mask length"}
    )
    mask_other: float = field(
        default=0,
        metadata={
            "help": "secondary mask argument "
            "(used for more complex distributions), "
            "see help in compute_mask_indicesh"
        },
    )
    no_mask_overlap: bool = field(
        default=False, metadata={"help": "whether to allow masks to overlap"}
    )
    mask_min_space: int = field(
        default=1,
        metadata={
            "help": "min space between spans (if no overlap is enabled)"
        },
    )
    encoder_layerdrop: float = field(
        default=0.0,
        metadata={"help": "probability of dropping a tarnsformer layer"},
    )
    dropout_features: float = field(
        default=0.0,
        metadata={
            "help": "dropout to apply to the features (after feat extr)"
        },
    )
    layer_norm_first: bool = field(
        default=False,
        metadata={"help": "apply layernorm first in the transformer"},
    )



class AVRomanizerWrapper(FairseqEncoder):
    def __init__(self, w2v_model):
        super().__init__(None)
        self.w2v_model = w2v_model

    def forward(self, source, padding_mask, **kwargs):
        w2v_args = {
            "source": source,
            "padding_mask": padding_mask,
        }

        x, padding_mask = self.w2v_model.extract_finetune(**w2v_args, mask=False)
    
        return {
            "encoder_out": x.transpose(0,1),  # B x T x C -> T x B x C
            "encoder_padding_mask": padding_mask,  # B x T
            "padding_mask": padding_mask
        }

    def extract_features(self, source, padding_mask, **kwargs):
        w2v_args = {
            "source": source,
            "padding_mask": padding_mask,
        }
        with torch.no_grad():
            x, padding_mask = self.w2v_model.extract_finetune(**w2v_args)

        return {
            "encoder_out": x,  # T x B x C
            "encoder_padding_mask": padding_mask,  # B x T
            "padding_mask": padding_mask
        }


    def reorder_encoder_out(self, encoder_out, new_order):
        if encoder_out["encoder_out"] is not None:
            encoder_out["encoder_out"] = encoder_out[
                "encoder_out"
            ].index_select(1, new_order)
        if encoder_out["encoder_padding_mask"] is not None:
            encoder_out["encoder_padding_mask"] = encoder_out[
                "encoder_padding_mask"
            ].index_select(0, new_order)
        if encoder_out["padding_mask"] is not None:
            encoder_out["padding_mask"] = encoder_out[
                "padding_mask"
            ].index_select(0, new_order)
        return encoder_out



@register_model("av_romanizer", dataclass=AV_RomanizerConfig)
class AV_RomanizerModel(BaseFairseqModel):
    def __init__(self, cfg: AV_RomanizerConfig, av_romanizer, target_dict):
        super().__init__()
        self.cfg = cfg        
        self.blank_weight = cfg.blank_weight
        self.blank_mode = cfg.blank_mode
        self.ctc_proj = nn.Linear(cfg.encoder_embed_dim,32)
        self.av_romanizer = av_romanizer
        self.roman_dict = target_dict
        
        
    @classmethod
    def build_model(cls, cfg: AV_RomanizerConfig, task: FairseqTask):
        """Build a new model instance."""
        
        arg_overrides = {
            "dropout": cfg.dropout,
            "activation_dropout": cfg.activation_dropout,
            "dropout_input": cfg.dropout_input,
            "attention_dropout": cfg.attention_dropout,
            "mask_length": cfg.mask_length,
            "mask_prob": cfg.mask_prob,
            "mask_selection": cfg.mask_selection,
            "mask_other": cfg.mask_other,
            "no_mask_overlap": cfg.no_mask_overlap,
            "mask_channel_length": cfg.mask_channel_length,
            "mask_channel_prob": cfg.mask_channel_prob,
            "mask_channel_selection": cfg.mask_channel_selection,
            "mask_channel_other": cfg.mask_channel_other,
            "no_mask_channel_overlap": cfg.no_mask_channel_overlap,
            "encoder_layerdrop": cfg.layerdrop,
            "feature_grad_mult": cfg.feature_grad_mult,
        }
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        w2v_path = f'{root_dir}/pretrained_models/avhubert/large_vox_iter5.pt'
        
        if cfg.w2v_args is None:
            state = checkpoint_utils.load_checkpoint_to_cpu(
                w2v_path, arg_overrides
            )
            w2v_args = state.get("cfg", None)
            if w2v_args is None:
                w2v_args = convert_namespace_to_omegaconf(state["args"])
            cfg.w2v_args = w2v_args
        else:
            state = None
            w2v_args = cfg.w2v_args
            if isinstance(w2v_args, Namespace):
                cfg.w2v_args = w2v_args = convert_namespace_to_omegaconf(
                    w2v_args
                )

        assert cfg.normalize == w2v_args.task.normalize, (
            "Fine-tuning works best when data normalization is the same. "
            "Please check that --normalize is set or unset for "
            "both pre-training and here"
        )

        w2v_args.task.data = cfg.data

        task_pretrain = tasks.setup_task(w2v_args.task)
        if state is not None:
            task_pretrain.load_state_dict(state['task_state'])

        encoder_ = task_pretrain.build_model(w2v_args.model)

        encoder = AVRomanizerWrapper(encoder_)
        if state is not None and not cfg.no_pretrained_weights:
            # set strict=False because we omit some modules
            del state['model']['mask_emb']
            encoder.w2v_model.load_state_dict(state["model"], strict=False)

        encoder.w2v_model.remove_pretraining_modules()

        return cls(cfg, encoder, task.target_dictionary)
    
    
    def forward(self, **kwargs):
        output = self.av_romanizer(**kwargs)
        x = self.ctc_proj(output['encoder_out']) # T x B x C - > T x B x 32
        padding_mask = output['padding_mask']    
    
        return {
            "ctc_out": x,  # T x B x 32
            "encoder_padding_mask": padding_mask,  # B x T
            "padding_mask": padding_mask
        }
    
    def extract_features(self, **kwargs):
        output = self.av_romanizer.extract_features(**kwargs)
        roman_char = self.ctc_proj(output['encoder_out'])
        return output, roman_char
    
    def get_logits(self, net_output, normalize=False):
        logits = net_output["ctc_out"]
        if self.blank_weight != 0:
            if self.blank_mode == "add":
                logits[..., 0] += self.blank_weight
            elif self.blank_mode == "set":
                logits[..., 0] = self.blank_weight
            else:
                raise Exception(f"invalid blank mode {self.blank_mode}")

        if net_output["padding_mask"] is not None and net_output["padding_mask"].any():
            number_of_classes = logits.size(-1)
            masking_tensor = torch.ones(
                number_of_classes, device=logits.device
            ) * float("-inf")
            masking_tensor[0] = 0

            if logits.size(0) > net_output["padding_mask"].size(1):
                net_output["padding_mask"] = F.pad(
                    net_output["padding_mask"], (1, 0), value=False
                )

            logits[net_output["padding_mask"].T] = masking_tensor.type_as(logits)

        if normalize:
            logits = utils.log_softmax(logits.float(), dim=-1)

        return logits

    def get_normalized_probs(self, net_output, log_probs):
        """Get normalized probabilities (or log probs) from a net's output."""

        logits = self.get_logits(net_output)

        if log_probs:
            return utils.log_softmax(logits.float(), dim=-1)
        else:
            return utils.softmax(logits.float(), dim=-1)
