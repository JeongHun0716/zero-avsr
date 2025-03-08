# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import ast
from itertools import chain
import logging
import math
import os
import sys
import json
import hashlib
import editdistance
from argparse import Namespace
import pdb

import numpy as np
import torch
from fairseq import checkpoint_utils, options, tasks, utils, distributed_utils
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.logging import progress_bar
from fairseq.logging.meters import StopwatchMeter, TimeMeter
from omegaconf import DictConfig

from pathlib import Path
import hydra
from hydra.core.config_store import ConfigStore
from fairseq.dataclass.configs import (
    CheckpointConfig,
    CommonConfig,
    CommonEvalConfig,
    GenerationConfig,
    DatasetConfig,
    DistributedTrainingConfig,
    FairseqDataclass,
)
from dataclasses import dataclass, field, is_dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
from omegaconf import OmegaConf, MISSING
import sacrebleu

logging.root.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

config_path = Path(__file__).resolve().parent / "conf"

@dataclass
class OverrideConfig(FairseqDataclass):
    noise_wav: Optional[str] = field(default=None, metadata={'help': 'noise wav file'})
    noise_prob: float = field(default=0, metadata={'help': 'noise probability'})
    noise_snr: float = field(default=0, metadata={'help': 'noise SNR in audio'})
    modalities: List[str] = field(default_factory=lambda: ["video"], metadata={'help': 'which modality to use'})
    data: Optional[str] = field(default=None, metadata={'help': 'path to test data directory'})
    label_dir: Optional[str] = field(default=None, metadata={'help': 'path to test label directory'})
    eval_bleu: bool = field(default=False, metadata={'help': 'evaluate bleu score'})
    av_romanizer_path: str = field(default=MISSING, metadata={'help': 'path to av-romanizer checkpoint'})
    llm_path: str = field(default=MISSING, metadata={'help': 'path to llama checkpoint'})
    use_speech_embs: bool = field(default=False)
    use_roman_toks: bool = field(default=False)

@dataclass
class InferConfig(FairseqDataclass):
    task: Any = None
    generation: GenerationConfig = GenerationConfig()
    common: CommonConfig = CommonConfig()
    common_eval: CommonEvalConfig = CommonEvalConfig()
    checkpoint: CheckpointConfig = CheckpointConfig()
    distributed_training: DistributedTrainingConfig = DistributedTrainingConfig()
    dataset: DatasetConfig = DatasetConfig()
    override: OverrideConfig = OverrideConfig()
    is_ax: bool = field(
        default=False,
        metadata={
            "help": "if true, assumes we are using ax for tuning and returns a tuple for ax to consume"
        },
    )


def main(cfg: DictConfig):

    if isinstance(cfg, Namespace):
        cfg = convert_namespace_to_omegaconf(cfg)

    assert cfg.common_eval.path is not None, "--path required for recognition!"
    
    if cfg.common_eval.results_path is not None:
        os.makedirs(cfg.common_eval.results_path, exist_ok=True)
        output_path = os.path.join(cfg.common_eval.results_path, "decode.log")
        with open(output_path, "w", buffering=1, encoding="utf-8") as h:
            return _main(cfg, h)

    return _main(cfg, sys.stdout)

from fairseq import tasks
from transformers import AutoTokenizer

def _main(cfg, output_file):
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        stream=output_file,
    )
    # logger = logging.getLogger("hybrid.speech_recognize")
    # if output_file is not sys.stdout:  # also print to stdout
    #     logger.addHandler(logging.StreamHandler(sys.stdout))

    utils.import_user_module(cfg.common)

    tokenizer = AutoTokenizer.from_pretrained(cfg.override.llm_path)
    model_override_cfg = {
    'model': {
        'llm_path': cfg.override.llm_path,
        'av_romanizer_path': cfg.override.av_romanizer_path
    }
    }
    models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task([cfg.common_eval.path],model_override_cfg,strict=False)

    saved_cfg.task.modalities = cfg.override.modalities
    task = tasks.setup_task(saved_cfg.task)
    task.build_tokenizer(saved_cfg.tokenizer)
    task.build_bpe(saved_cfg.bpe)

    logger.info(cfg)

    # Fix seed for stochastic decoding
    if cfg.common.seed is not None :
        np.random.seed(cfg.common.seed)
        utils.set_torch_seed(cfg.common.seed)

    use_cuda = torch.cuda.is_available()

    # Set dictionary
    dictionary = task.target_dictionary

    # loading the dataset should happen after the checkpoint has been loaded so we can give it the saved task config

    task.cfg.llm_path = cfg.override.llm_path
    task.cfg.noise_prob = cfg.override.noise_prob
    task.cfg.noise_snr = cfg.override.noise_snr
    task.cfg.noise_wav = cfg.override.noise_wav

    if cfg.override.data is not None:
        task.cfg.data = cfg.override.data
    if cfg.override.label_dir is not None:
        task.cfg.label_dir = cfg.override.label_dir

    task.load_dataset(cfg.dataset.gen_subset, task_cfg=cfg.task)

    lms = [None]

    # Optimize ensemble for generation

    for model in chain(models, lms):
        if model is None:
            continue
        if use_cuda and not cfg.distributed_training.pipeline_model_parallel:
            model.cuda()
            model.half()

    itr = task.get_batch_iterator(
        dataset=task.dataset(cfg.dataset.gen_subset),
        max_tokens=cfg.dataset.max_tokens,
        max_sentences=cfg.dataset.batch_size,
        max_positions=utils.resolve_max_positions(
            task.max_positions(), *[m.max_positions() for m in models]
        ),
        ignore_invalid_inputs=cfg.dataset.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=cfg.dataset.required_batch_size_multiple,
        seed=cfg.common.seed,
        num_shards=cfg.distributed_training.distributed_world_size,
        shard_id=cfg.distributed_training.distributed_rank,
        num_workers=cfg.dataset.num_workers,
        data_buffer_size=cfg.dataset.data_buffer_size,
    ).next_epoch_itr(shuffle=False)
    progress = progress_bar.progress_bar(
        itr,
        log_format=cfg.common.log_format,
        log_interval=cfg.common.log_interval,
        default_log_format=("tqdm" if not cfg.common.no_progress_bar else "simple"),
    )

    gen_timer = StopwatchMeter()
    def decode_fn(x):
        symbols_ignore = {"<unk>", "<mask>","<pad>", "</s>"}
        if hasattr(task.datasets[cfg.dataset.gen_subset].label_processors[0], 'decode'):
            return tokenizer.decode(x, skip_special_tokens=True)
        chars = dictionary.string(x, extra_symbols_to_ignore=symbols_ignore)
        words = " ".join("".join(chars.split()).replace('|', ' ').split())
        return words

    num_sentences = 0
    has_target = True
    wps_meter = TimeMeter()
    result_dict = {'langs':[], 'ref': [], 'hypo': [], 'instruction': []}
    model = models[0]
    model.eval()
    for sample in progress:
        sample = utils.move_to_cuda(sample) if use_cuda else sample
        if "net_input" not in sample:
            continue
        
        if sample['net_input']['source']['video'] is not None:
            sample['net_input']['source']['video'] = sample['net_input']['source']['video'].to(torch.half)
            
        sample['net_input']['source']['audio'] = sample['net_input']['source']['audio'].to(torch.half)

        with torch.no_grad():
            best_hypo = model.generate(num_beams=cfg.generation.beam,
                                    temperature=cfg.generation.temperature,
                                    use_speech_embs=cfg.override.use_speech_embs,
                                    use_roman_toks=cfg.override.use_roman_toks,
                                    **sample["net_input"])
        best_hypo = tokenizer.batch_decode(
                best_hypo, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
        
        for i in range(len(sample["id"])):
            result_dict['langs'].append(sample['net_input']['source']['langs'][i].split(',')[0])
            target = sample['target'][i].masked_fill(
                sample['target'][i] == -100, 0
            )
            ref_sent = tokenizer.decode(target.int().cpu(), skip_special_tokens=True, clean_up_tokenization_spaces=False)
            result_dict['ref'].append(ref_sent)
            hypo_str = best_hypo[i]
            instruction = tokenizer.decode(sample['net_input']['source']['instruction'][i].int().cpu(), skip_special_tokens=True, clean_up_tokenization_spaces=False)
            result_dict['instruction'].append(instruction)
            result_dict['hypo'].append(hypo_str)
            logger.info(f"\nINST:{instruction}\nREF:{ref_sent}\nHYP:{hypo_str}\n")

    yaml_str = OmegaConf.to_yaml(cfg.generation)
    fid = int(hashlib.md5(yaml_str.encode("utf-8")).hexdigest(), 16)
    fid = fid % 1000000
    result_fn = f"{cfg.common_eval.results_path}/hypo-{fid}.json"
    json.dump(result_dict, open(result_fn, 'w'), indent=4)

    nine_langs = ['Arabic','German', 'Greek',  'Spanish', 'French', 'Italian', 'Portuguese','Russian', 'English']
    nine_langs_results = {}
    for lang in nine_langs:
        nine_langs_results[lang] = {'n_err': 0, 'n_total': 0, 'c_err': 0, 'c_total': 0}

    if not cfg.override.eval_bleu:
        n_err, n_total = 0, 0
        assert len(result_dict['hypo']) == len(result_dict['ref']) == len(result_dict['langs'])
        for lang, hypo, ref in zip(result_dict['langs'], result_dict['hypo'], result_dict['ref']):
            hypo_w, ref_w = hypo.strip().split(), ref.strip().split()
            nine_langs_results[lang]['n_err'] += editdistance.eval(hypo_w, ref_w)
            nine_langs_results[lang]['n_total'] += len(ref_w)
            
            hypo_chars = hypo.strip().replace(" ", "")
            ref_chars  = ref.strip().replace(" ", "")
            nine_langs_results[lang]['c_err'] += editdistance.eval(hypo_chars, ref_chars)
            nine_langs_results[lang]['c_total'] += len(ref_chars)
                
        total_n_err, total_n_total = 0, 0
        total_c_err, total_c_total = 0, 0

        non_eng_n_err, non_eng_n_total = 0, 0
        non_eng_c_err, non_eng_c_total = 0, 0

        for lang in nine_langs:
            n_err = nine_langs_results[lang]['n_err']
            n_total = nine_langs_results[lang]['n_total']
            c_err = nine_langs_results[lang]['c_err']
            c_total = nine_langs_results[lang]['c_total']
            
            total_n_err += n_err
            total_n_total += n_total
            total_c_err += c_err
            total_c_total += c_total

            if lang != 'English':
                non_eng_n_err += n_err
                non_eng_n_total += n_total
                non_eng_c_err += c_err
                non_eng_c_total += c_total

            if n_total > 0:
                wer = 100 * n_err / n_total
            else:
                wer = 0  

            wer_fn = f"{cfg.common_eval.results_path}/wer.{fid}"
            with open(wer_fn, "w") as fo:
                fo.write(f'Language: {lang}\n')
                fo.write(f"WER: {wer}\n")
                fo.write(f"err / num_ref_words = {n_err} / {n_total}\n\n")
                fo.write(f"{yaml_str}")
            
            if c_total > 0:
                cer = 100 * c_err / c_total
            else:
                cer = 0
                
            cer_fn = f"{cfg.common_eval.results_path}/cer.{fid}"
            with open(cer_fn, "w") as fo:
                fo.write(f'Language: {lang}\n')
                fo.write(f"CER: {cer}\n")
                fo.write(f"err / num_ref_chars = {c_err} / {c_total}\n\n")
                fo.write(f"{yaml_str}\n")
                
            logger.info(f"\n{lang} WER: {wer}%\n{lang} CER: {cer}%")

        # -----------------------------
        if total_n_total > 0:
            overall_wer = 100 * total_n_err / total_n_total
        else:
            overall_wer = 0

        if total_c_total > 0:
            overall_cer = 100 * total_c_err / total_c_total
        else:
            overall_cer = 0

        if non_eng_n_total > 0:
            non_eng_wer = 100 * non_eng_n_err / non_eng_n_total
        else:
            non_eng_wer = 0

        if non_eng_c_total > 0:
            non_eng_cer = 100 * non_eng_c_err / non_eng_c_total
        else:
            non_eng_cer = 0

        overall_fn = f"{cfg.common_eval.results_path}/overall.{fid}"
        with open(overall_fn, "w") as fo:
            fo.write("Overall Metrics (All Languages):\n")
            fo.write(f"WER: {overall_wer}\n")
            fo.write(f"CER: {overall_cer}\n\n")
            fo.write("Metrics Excluding English:\n")
            fo.write(f"WER: {non_eng_wer}\n")
            fo.write(f"CER: {non_eng_cer}\n")
            
        logger.info(f"\nOverall WER (All Languages): {overall_wer}%")
        logger.info(f"Overall CER (All Languages): {overall_cer}%")
        logger.info(f"\nNon-English WER: {non_eng_wer}%")
        logger.info(f"Non-English CER: {non_eng_cer}%")

            
    else:
        bleu = sacrebleu.corpus_bleu(result_dict['hypo'], [result_dict['ref']])
        bleu_score = bleu.score
        bleu_fn = f"{cfg.common_eval.results_path}/bleu.{fid}"
        with open(bleu_fn, "w") as fo:
            fo.write(f"BLEU: {bleu_score}\n")
            fo.write(f"{yaml_str}")
        logger.info(f"BLEU: {bleu_score}\n")
    return


@hydra.main(config_path=config_path, config_name="infer")
def hydra_main(cfg: InferConfig) -> Union[float, Tuple[float, Optional[float]]]:
    container = OmegaConf.to_container(cfg, resolve=True, enum_to_str=True)
    cfg = OmegaConf.create(container)
    OmegaConf.set_struct(cfg, True)

    try:
        if cfg.common.profile:
            with torch.cuda.profiler.profile():
                with torch.autograd.profiler.emit_nvtx():
                    distributed_utils.call_main(cfg, main)
        else:
            distributed_utils.call_main(cfg, main)

    except BaseException as e:  # pylint: disable=broad-except
        if not cfg.common.suppress_crashes:
            raise
        else:
            logger.error("Crashed! %s", str(e))
    return


def cli_main() -> None:
    try:
        from hydra._internal.utils import (
            get_args,
        )  # pylint: disable=import-outside-toplevel

        cfg_name = get_args().config_name or "infer"
    except ImportError:
        logger.warning("Failed to get config name from hydra args")
        cfg_name = "infer"

    cs = ConfigStore.instance()
    cs.store(name=cfg_name, node=InferConfig)

    for k in InferConfig.__dataclass_fields__:
        if is_dataclass(InferConfig.__dataclass_fields__[k].type):
            v = InferConfig.__dataclass_fields__[k].default
            cs.store(name=k, node=v)

    hydra_main()  # pylint: disable=no-value-for-parameter


if __name__ == "__main__":
    cli_main()