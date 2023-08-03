CUDA_VISIBLE_DEVICES="0" \
python src/dataset/build_store.py \
    --data_type redial \
    --backbone_model bert \
    --crs_model_path ./result/my-sup-simcse-bert-test-crs/checkpoint-4000/pytorch_model.bin \
    --representation_position cls \
    # --test_mode \