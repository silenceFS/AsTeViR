gpu=$1
ssnV=$2
ssnT=$3
CUDA_VISIBLE_DEVICES=$gpu python trainer.py \
    --rootpath /data/fs/hybrid_space_dataset \
    --overwrite 1 \
    --max_violation \
    --text_norm \
    --visual_norm \
    --collection msrvtt10k \
    --visual_feature resnext101-resnet152 \
    --space latent \
    --batch_size 128 \
    --style distill \
    --associate_modal "VT" \
    --alpha_V 0.05 \
    --beta_V 0.25 \
    --gamma_V 0.1 \
    --support_set_number_V $ssnV \
    --support_set_number_T $ssnT \
    --similarity_type diag \
    --postfix "ssnV_${ssnV}_ssnT_${ssnT}"

