CUDA_VISIBLE_DEVICES="0" \
python src/dataset/build_store.py \
    --data_type redial \
    --backbone_model bart \
    --crs_model_path ./result/my-sup-simcse-bart-test-crs/checkpoint-4000/pytorch_model.bin \
    --representation_position avg_first_last \
    # --test_mode \