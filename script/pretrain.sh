# CUDA_VISIBLE_DEVICES="4,5,6,7" 
accelerate launch \
    src/pretrain.py \
    --data_file_path ./data/ \
    --logging_dir ./logs \
    --per_device_eval_batch_size 16 \
    --per_device_train_batch_size 16 \
    --learning_rate 5e-5 \
    --mlm_probability 0.9 \
    --num_train_epochs 2 \
    --gradient_accumulation_steps 4 \
    --output_dir ./save \
    --backbone_model bert \
    --reloadDataset true \
    --with_tracking \
    --data_type redial \
    --checkpointing_steps epoch \
    # --train_file_path /mnt/zhoukun/zhipeng/MLM/dialogue_shuffle/processed_dialogue.pkl \
    # --validation_file_path /mnt/zhoukun/zhipeng/MLM/dialogue_shuffle/processed_dialogue_valid.pkl \