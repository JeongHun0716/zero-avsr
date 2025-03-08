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



from avhubert.hubert_pretraining import AVHubertPretrainingConfig, AVHubertPretrainingTask, LabelEncoder, LabelEncoderS2SToken
logger = logging.getLogger(__name__)


from .dataset import AVRomanDataset


@dataclass
class AVRomanConfig(AVHubertPretrainingConfig):
    time_mask: bool = field(default=False)
    random_erase: bool = field(default=False)
    target_dictionary: Optional[str] = field(
        default=None,
        metadata={
            "help": "override default dictionary location"
        }
    )
    label_rate: int = field(default=-1)



@register_task("av-roman-task", dataclass=AVRomanConfig)
class AV_RomanTask(AVHubertPretrainingTask):
    cfg: AVRomanConfig

    def __init__(
        self,
        cfg: AVRomanConfig,
    ) -> None:
        super().__init__(cfg)

        logger.info(f"current directory is {os.getcwd()}")
        logger.info(f"AV_RomanTask Config {cfg}")

        self.blank_symbol = "<s>"
        self.state.add_factory("target_dictionary", self.load_target_dictionary)

    def load_target_dictionary(self):
        if self.cfg.labels:
            target_dictionary = self.cfg.data
            if self.cfg.target_dictionary:  # override dict
                target_dictionary = self.cfg.target_dictionary
            root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            dict_path = f'{root_dir}/roman_dict/dict.ltr.txt'
            logger.info('Using dict_path : {}'.format(dict_path))
            return Dictionary.load(dict_path)
        return None
    
    def load_dataset(self, split: str, **kwargs) -> None:
        manifest = f"{self.cfg.data}/{split}.tsv"
        dictionaries = [self.target_dictionary] 
        pad_list = [dictionary.pad() for dictionary in dictionaries]
        eos_list = [dictionary.eos() for dictionary in dictionaries]

        procs = [LabelEncoder(dictionary) for dictionary in dictionaries]
        paths = [
            f"{self.get_label_dir()}/{split}.{l}" for l in self.cfg.labels
        ]
        image_aug = self.cfg.image_aug if split == 'train' else False
        noise_fn, noise_snr = f"{self.cfg.noise_wav}/{split}.tsv" if self.cfg.noise_wav is not None else None, eval(self.cfg.noise_snr)
        noise_num = self.cfg.noise_num # 

        self.datasets[split] = AVRomanDataset(
            manifest,
            sample_rate=self.cfg.sample_rate,
            label_paths=paths,
            label_rates=self.cfg.label_rate,
            pad_list=pad_list,
            eos_list=eos_list,
            label_processors=procs,
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
        )
