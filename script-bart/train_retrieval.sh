#!/bin/bash

# In this example, we show how to train SimCSE using multiple GPU cards and PyTorch's distributed data parallel on supervised NLI dataset.
# Set how many GPUs to use

NUM_GPU=3

# Randomly set a port number
# If you encounter "address already used" error, just run again or manually set an available port id.
PORT_ID=$(expr $RANDOM + 1000)

# Allow multiple threads
export OMP_NUM_THREADS=8
export DATA_TYPE=redial
export BACKBONE_MODEL=bart

# Use distributed data parallel
# If you only want to use one card, uncomment the following line and comment the line with "torch.distributed.launch"

# python src/contrastive_train.py \

CUDA_VISIBLE_DEVICES="0,1,2" \
python -m torch.distributed.launch --nproc_per_node $NUM_GPU --master_port $PORT_ID src/contrastive_train.py \
    --model_name_or_path facebook/bart-base \
    --train_file data/${DATA_TYPE}/${BACKBONE_MODEL}/pretrain/contrastive_learning.csv \
    --output_dir result/my-sup-simcse-bart-test-crs \
    --num_train_epochs 3 \
    --per_device_train_batch_size 64 \
    --learning_rate 5e-5 \
    --max_seq_length 128 \
    --evaluation_strategy steps \
    --eval_steps 1000 \
    --pooler_type avg_first_last \
    --overwrite_output_dir \
    --temp 0.05 \
    --do_train \
    --do_eval \
    --fp16 \
    --crs_model_path save/${DATA_TYPE}/${BACKBONE_MODEL}/backbone/pretrain_${BACKBONE_MODEL}.pth \
    --data_root data/${DATA_TYPE}/ \
    --data_type ${DATA_TYPE} \
    --backbone_model ${BACKBONE_MODEL} \
    --use_crs true \
    #  --test_mode true
    # --load_best_model_at_end \
    # --metric_for_best_model stsb_spearman \

    "$@"