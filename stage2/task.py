# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os, glob
import sys
from typing import List, Optional, Any, OrderedDict

from fairseq import search
from dataclasses import dataclass, field
from fairseq.data import Dictionary
from fairseq.tasks import register_task
from omegaconf import MISSING, II

from avhubert.hubert_pretraining import AVHubertPretrainingConfig, AVHubertPretrainingTask, LabelEncoder, LabelEncoderS2SToken
logger = logging.getLogger(__name__)


from .dataset import zero_avsr_dataset


@dataclass
class Zero_AVSRConfig(AVHubertPretrainingConfig):
    time_mask: bool = field(default=False)
    random_erase: bool = field(default=False)
    target_dictionary: Optional[str] = field(
        default=None,
        metadata={
            "help": "override default dictionary location"
        }
    )
    label_rate: int = field(default=-1)
    llm_path: str = field(
        default=MISSING, metadata={"help": "path to llama checkpoint"}
    )
    ## 
    zero_shot: bool = field(
        default=True, metadata={"help": "select zero shot setting"}
    )
    zero_shot_lang: str = field(
        default='deu', metadata={"help": "ara, deu, ell, spa, fra, ita, por, rus"}
    )


@register_task("zero-avsr-task", dataclass=Zero_AVSRConfig)
class Zero_AVSR_TrainingTask(AVHubertPretrainingTask):
    cfg: Zero_AVSRConfig

    def __init__(
        self,
        cfg: Zero_AVSRConfig,
    ) -> None:
        super().__init__(cfg)

        logger.info(f"current directory is {os.getcwd()}")
        logger.info(f"Zero_AVSRConfig Config {cfg}")

        self.blank_symbol = "<s>"

    @property
    def source_dictionary(self) -> Optional[Dictionary]:
        return None

    @property
    def target_dictionary(self) -> Optional[Dictionary]:
        return None
    
    @property
    def dictionaries(self) -> List[Dictionary]:
        return None
    
    
    @classmethod
    def setup_task(
        cls, cfg: Zero_AVSRConfig, **kwargs
    ) -> "Zero_AVSR_TrainingTask":
        if cfg.pdb:
            import pdb
            pdb.set_trace()
        return cls(cfg)

    def get_label_dir(self) -> str:
        if self.cfg.label_dir is None:
            return self.cfg.data
        return self.cfg.label_dir

    
    def load_dataset(self, split: str, **kwargs) -> None:
        manifest = f"{self.cfg.data}/{split}.tsv"
        logger.info(f"Using tokenizer")
        paths = [
            f"{self.get_label_dir()}/{split}.{l}" for l in self.cfg.labels
        ]
        image_aug = self.cfg.image_aug if split == 'train' else False
        noise_fn, noise_snr = f"{self.cfg.noise_wav}/{split}.tsv" if self.cfg.noise_wav is not None else None, eval(self.cfg.noise_snr)
        noise_num = self.cfg.noise_num # 

        self.datasets[split] = zero_avsr_dataset(
            manifest,
            sample_rate=self.cfg.sample_rate,
            llm_path=self.cfg.llm_path,
            label_paths=paths,
            label_rates=self.cfg.label_rate,
            max_keep_sample_size=self.cfg.max_sample_size,
            min_keep_sample_size=self.cfg.min_sample_size,
            max_sample_size=self.cfg.max_trim_sample_size,
            pad_audio=self.cfg.pad_audio,
            normalize=self.cfg.normalize,
            store_labels=True,
            random_crop=self.cfg.random_crop,
            single_target=self.cfg.single_target,
            stack_order_audio=self.cfg.stack_order_audio,
            skip_verify=self.cfg.skip_verify,
            image_mean=self.cfg.image_mean,
            image_std=self.cfg.image_std,
            image_crop_size=self.cfg.image_crop_size,
            image_aug=image_aug,
            modalities=self.cfg.modalities,
            is_s2s=self.cfg.is_s2s,
            noise_fn=noise_fn,
            noise_prob=self.cfg.noise_prob,
            noise_snr=noise_snr,
            noise_num=noise_num,
            time_mask=self.cfg.time_mask,
            random_erase=self.cfg.random_erase,
            zero_shot=self.cfg.zero_shot,
            zero_shot_lang=self.cfg.zero_shot_lang,
        )
