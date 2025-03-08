current_pth=$(pwd)
fairseq_pth=$current_pth/fairseq
src_pth=$current_pth/stage1


for zero_shot_lang in ara deu ell spa fra ita por rus; do


data_pth=$current_pth/marc/manifest/zero_shot/no_${zero_shot_lang}
#checkpoint_save_pth=$current_pth/pretrained_models/av-romanizer/no_${zero_shot_lang}
checkpoint_save_pth=$current_pth/debug
num_gpus=8
update_freq=8
metric=uer # wer or uer
noise_wav=$current_pth/noise_wav/all


export OMP_NUM_THREADS=1
PYTHONPATH=$fairseq_pth \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 fairseq-hydra-train \
    --config-dir $src_pth/conf/ \
    --config-name av_romanizer.yaml \
    task.data=$data_pth \
    task.label_dir=$data_pth \
    hydra.run.dir=$checkpoint_save_pth \
    common.user_dir=$src_pth \
    model._name=av_romanizer \
    model.w2v_path=$current_pth/pretrained_models/avhubert/large_vox_iter5.pt \
    distributed_training.distributed_world_size=$num_gpus \
    distributed_training.nprocs_per_node=$num_gpus \
    optimization.max_update=100000 \
    optimization.lr=[1e-4] \
    optimization.update_freq=[$update_freq] \
    checkpoint.best_checkpoint_metric=uer \
    task.noise_wav=$noise_wav 
done