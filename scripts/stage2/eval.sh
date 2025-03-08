#! /bin/bash
# Copyright (c) Meta Platforms, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the

# LICENSE file in the root directory of this source tree.

ROOT=$(pwd)
SRC_PTH=$ROOT/stage2


MODEL_PATH=$ROOT/pretrained_models/zero-avsr/all/checkpoint_best.pt
LLM_PATH=meta-llama/Llama-3.2-3B
AV_ROMA_PATH=$ROOT/pretrained_models/av-romanizer/all/checkpoint_best.pt
MANIFEST=$ROOT/marc/manifest/stage2/

OUT_PATH=$ROOT/results


export OMP_NUM_THREADS=1
PYTHONPATH=$ROOT/fairseq \
CUDA_VISIBLE_DEVICES=4 python -B $SRC_PTH/eval.py --config-dir ${SRC_PTH}/conf --config-name s2s_decode \
    dataset.gen_subset=test \
    common.user_dir=${SRC_PTH} \
    generation.beam=2 \
    generation.temperature=0.3 \
    override.llm_path=${LLM_PATH} \
    override.av_romanizer_path=${AV_ROMA_PATH} \
    override.modalities=['video','audio'] \
    common_eval.path=${MODEL_PATH} \
    common_eval.results_path=${OUT_PATH} \
    override.use_speech_embs=true \
    override.use_roman_toks=false \
    +override.data=$MANIFEST \
    +override.label_dir=$MANIFEST 
