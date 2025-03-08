#! /bin/bash
# Copyright (c) Meta Platforms, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

ROOT=$(pwd)
STAGE_PTH=$ROOT/stage2
AV_ROMA_PATH=$ROOT/pretrained_models/av-romanizer/all/checkpoint_best.pt
LLM_PATH=meta-llama/Llama-3.2-3B
noise_wav=$ROOT/noise_wav/all

NGPUS=7
update_freq=9
OUT_PATH=$ROOT/pretrained_models/zero-avsr/rus_back_trans_prob_0


export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
PYTHONPATH=$ROOT/fairseq \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 fairseq-hydra-train \
    --config-dir ${STAGE_PTH}/conf/ \
    --config-name zero_avsr.yaml \
    task.data=$ROOT/marc/manifest/stage2 \
    task.label_dir=$ROOT/marc/manifest/stage2 \
    task.llm_path=${LLM_PATH} \
    task.noise_wav=$noise_wav \
    task.zero_shot_lang=rus \
    checkpoint.save_interval_updates=2050 \
    common.empty_cache_freq=500 \
    hydra.run.dir=${OUT_PATH} \
    common.user_dir=${STAGE_PTH} \
    common.seed=1 \
    model.av_romanizer_path=${AV_ROMA_PATH} \
    model.av_romanizer_embed_dim=1024 \
    model._name=zero-avsr \
    model.use_roman_tok=false \
    model.use_speech_emb=true \
    model.back_trans_prob=0 \
    model.llm_path=${LLM_PATH} \
    model.llama_embed_dim=3072 \
    model.target_modules=q_proj.k_proj.v_proj.o_proj \
    model.lora_rank=16 \
    model.lora_alpha=32 \
    optimization.update_freq=[$update_freq] \
    optimization.lr=[1e-4] \
    optimization.max_update=50000 \
    optimization.max_epoch=5 \
    distributed_training.distributed_world_size=${NGPUS} \
    distributed_training.nprocs_per_node=${NGPUS} \
    distributed_training.find_unused_parameters=true
