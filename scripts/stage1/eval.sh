root=$(pwd)

model_path=$root/pretrained_models/av-romanizer/all/checkpoint_best.pt # place the downloaded uroman model here

for lang in ara deu ell spa fra ita por rus
do

manifest_pth=$root/marc/eval/muavic/$lang 
subset=test
fairseq_pth=$root/fairseq
out_pth=$root/evaluation/clean/stage1/${lang}

export OMP_NUM_THREADS=1
PYTHONPATH=$fairseq_pth \
CUDA_VISIBLE_DEVICES=0 python $root/stage1/infer.py \
    --config-dir $root/stage1/conf/ \
    --config-name infer_common \
    decoding.type=viterbi \
    decoding.results_path=${out_pth} \
    dataset.max_tokens=1000 \
    distributed_training.distributed_world_size=1 \
    common_eval.path=$model_path \
    hydra.run.dir=${root} \
    task.data=$manifest_pth \
    dataset.gen_subset=$subset \
    common_eval.post_process=letter \
    common.user_dir=$root/stage1 \
    +override.data=$manifest_pth \
    +override.label_dir=$manifest_pth 


if [ "$lang" == "ara" ]; then
    target_lang=arabic
elif [ "$lang" == "deu" ]; then
    target_lang=german
elif [ "$lang" == "ell" ]; then
    target_lang=greek
elif [ "$lang" == "fra" ]; then
    target_lang=french
elif [ "$lang" == "ita" ]; then
    target_lang=italian
elif [ "$lang" == "por" ]; then
    target_lang=portuguese
elif [ "$lang" == "rus" ]; then
    target_lang=russian
elif [ "$lang" == "spa" ]; then
    target_lang=spanish
elif [ "$lang" == "eng" ]; then
    target_lang=english
fi

python $root/stage1/sort_hypo_files.py \
    --input_pth $out_pth/hypo.word \
    --output_pth $out_pth/sorted_hypo.word


PYTHONPATH=$root \
python $root/stage1/de_romanize_w_gpt_api.py \
    --gt_label $manifest_pth/test.wrd \
    --target_lang $target_lang \
    --lang $lang \
    --predicted_roman_file $out_pth/sorted_hypo.word \
    --output_pth $out_pth/avsr.json


done
