MODEL="bart"
DATA_TYPE="redial"
MODEL_PATH="save/${DATA_TYPE}/${MODEL}/recommendation/${MODEL}_recommend.pth"
QUERY_POSITION="avg_first_last"
DSTORE_PATH=data/${DATA_TYPE}/${MODEL}/knn/${QUERY_POSITION}
echo "MODEL_PATH: ${MODEL_PATH}"

CUDA_VISIBLE_DEVICES="1" python \
src/train_conv.py \
    --config_path config/modeling_kgsf.yaml \
    --data_type ${DATA_TYPE} \
    --movie_embedding_file_path  data/${DATA_TYPE}/ \
    --query_position $QUERY_POSITION \
    --learning_rate 1e-4 \
    --num_train_epochs 10 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --output_dir  save/${DATA_TYPE}/${MODEL}/conversation \
    --backbone_model $MODEL \
    --dstore_path $DSTORE_PATH \
    --retrieval_k 10 \
    --load_trained_model_path $MODEL_PATH \
    --load_trained_model  \
    --with_tracking \
    --logging_dir ./logs \
    --num_warmup_steps 6000 \
    --tokenizer microsoft/DialoGPT-small \
    --model microsoft/DialoGPT-small \
    --ignore_pad_token_for_loss \
    --save_data true \
    # --reload_data true \
    # --debug
    # --add_knowledge_prompt
