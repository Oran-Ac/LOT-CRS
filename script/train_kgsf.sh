CUDA_VISIBLE_DEVICES="0" \
python src/train_kgsf.py \
    --config_path config/modeling_kgsf.yaml \
    --data_type redial \
    --logging_dir ./logs \
    --output_dir save/redial/kgsf \
    --with_tracking \
    --per_device_test_batch_size 128  \
    --per_device_train_batch_size 128  \
    --gradient_accumulation_steps 1 \
    --pre_learning_rate 1e-3 \
    --rec_learning_rate 1e-3  \
    --pre_num_train_epochs 3 \
    --rec_num_train_epochs 8 \
    --save_data true \
    --reload_data true \

