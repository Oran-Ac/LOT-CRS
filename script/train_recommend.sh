MODEL="bert"
DATA_TYPE="redial"
MODEL_PATH="result/my-sup-simcse-bert-test-crs/checkpoint-4000/pytorch_model.bin"
QUERY_POSITION="cls"
DSTORE_PATH=data/${DATA_TYPE}/${MODEL}/knn/${QUERY_POSITION}

CUDA_VISIBLE_DEVICES="1" python \
src/train_rec.py \
    --config_path config/modeling_kgsf.yaml \
    --data_type ${DATA_TYPE} \
    --kgsf_model_path save/${DATA_TYPE}/kgsf/kgsf.pth \
    --movie_embedding_file_path  data/${DATA_TYPE}/ \
    --query_position $QUERY_POSITION \
    --learning_rate 2e-5 \
    --num_train_epochs 15 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 2 \
    --output_dir  save/${DATA_TYPE}/${MODEL}/recommendation \
    --backbone_model $MODEL \
    --dstore_path $DSTORE_PATH \
    --retrieval_k 10 \
    --beta 0.4 \
    --alpha 0.6 \
    --load_trained_model_path $MODEL_PATH \
    --load_trained_model  \
    --with_tracking \
    --logging_dir ./logs \
    --save_data true \
    --reload_data true \
    # --debug
    # --add_knowledge_prompt
