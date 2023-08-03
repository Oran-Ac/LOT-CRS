import argparse
import math
import os
import sys
import time
import logging
import datasets
from utils.config import redial_config
import json
import numpy as np
import torch
import transformers
from accelerate import Accelerator
from accelerate.utils import set_seed
from accelerate.logging import get_logger
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    MODEL_MAPPING,
    AdamW,
    # DataCollatorForLanguageModeling,
    # SchedulerType,
    get_linear_schedule_with_warmup,
    AutoModel,
    AutoTokenizer
)
from transformers.utils.versions import require_version
from utils.args import gpt2_special_tokens_dict, prompt_special_tokens_dict
from utils.args import BACKBONE_MODEL_MAPPINGS,Movie_Name_Path,AT_TOKEN_NUMBER,MODEL_TYPES_MAPPING

from dataset.dataset_conv import CRSDatasetConversation,CRSDataConversationCollator
from utils.evaluate_conv import ConvEvaluator
from model.modeling_conv import PromptGPT2forCRS
from model.modeling_crs import CRSModel
from utils.utils import faiss_search
import numpy as np
from utils.utils import resize_token_embeddings,add_tokens_for_tokenizer
import faiss
import copy

logger = get_logger(__name__)
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        type=str,
        default='./modeling_kgsf.yaml',
        help = 'path of config for kgsf model'
    )
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument("--debug", action='store_true', help="Debug mode.")
    # data
    parser.add_argument(
        "--data_type",
        type=str,
        choices=['redial','inspire'],
        help = 'data used to train and test'
    )
    parser.add_argument(
        '--dstore_path',
        type=str,
        default='./dstore',
        help = 'path to dstore'
    )
    parser.add_argument(
        '--movie_embedding_file_path',
        type=str,
        default='./pretrainDataset',
        help = 'path to load the movie embedding'
    )
    parser.add_argument(
        '--data_file_path',
        type=str,
        default='./data',
        help = 'path to load data'
    )
    parser.add_argument(
        '--save_data',
        type=bool,
        help = 'whether to save data for re-use',
        default= False
    )
    parser.add_argument(
        '--reload_data',
        type=bool,
        help = 'whether to re-use',
        default= False
    )
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument("--tokenizer", type=str)
    parser.add_argument("--ignore_pad_token_for_loss", action='store_true')
    # model
    parser.add_argument(
        '--backbone_model',
        type=str,
        choices=['bert','bart'],
        help = 'backbone'
    )
    parser.add_argument(
        '--query_position',
        type=str,
        default=None,
    )
    parser.add_argument(
        '--retrieval_k',
        type=int,
    )
    parser.add_argument(
        '--load_trained_model_path',
        type=str,
        default=None
    )
    parser.add_argument(
        '--load_trained_model',
        action = 'store_true'
    )
    parser.add_argument(
        '--add_knowledge_prompt',
        action='store_true',
    )
    parser.add_argument("--max_gen_len", type=int, default=50)
    parser.add_argument("--model", type=str, default="gpt2")
    # optim
    parser.add_argument("--num_train_epochs", type=int, default=10, help="Total number of training epochs to perform.")
    parser.add_argument("--max_train_steps", type=int, default=None,
                        help="Total number of training steps to perform. If provided, overrides num_train_epochs.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4,
                        help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4,
                        help="Batch size (per device) for the evaluation dataloader.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                        help="Initial learning rate (after the potential warmup period) to use.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay to use.")
    parser.add_argument('--max_grad_norm', type=float)
    parser.add_argument('--num_warmup_steps', type=int, default=10000)
    parser.add_argument('--fp16', action='store_true', help='use automatic mixed precision to speed up.')
    # logging
    parser.add_argument(
        '--with_tracking',
        action='store_true',
        help = 'output log or not'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        help = 'output dir'
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default=None,
        help="Where to store the logs.",
    )
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    config = redial_config(args.config_path,args.data_type)
    config['data_type'] = args.data_type

    # Initialize the accelerator. We will let the accelerator handle device placement for us.
    accelerator = Accelerator(log_with="wandb", project_dir=args.output_dir) if args.with_tracking else Accelerator()
    if accelerator.is_main_process:
        # suffix = 'init'
        # if args.load_trained_model:
        #     suffix = args.load_trained_model_path.replace('.pth','')
        #     suffix = suffix.split('/')[-1]
        logging.basicConfig( 
                    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                    datefmt="%m/%d/%Y %H:%M:%S",
                    level=logging.INFO,           
                    filename=os.path.join(args.logging_dir,f'conv_{args.backbone_model}_{args.query_position}.log'),
                    filemode='a',##模式，有w和a，w就是写模式，每次都会重新写日志，覆盖之前的日志                    #a是追加模式，默认如果不写的话，就是追加模式                    format=                    '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'                    #日志格式
                    )
        logger.info(json.dumps(vars(args),indent=2))
        logger.info(accelerator.state, main_process_only=False)
    
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    # else:
    #     datasets.utils.logging.set_verbosity_error()
    #     transformers.utils.logging.set_verbosity_error()


    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"*[BACK BONE MODEL]*: {args.backbone_model}")


    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer.add_special_tokens(gpt2_special_tokens_dict)
    model = PromptGPT2forCRS.from_pretrained(args.model)
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    config['conv'] = {}
    config['conv']['hidden_size'] = model.config.n_embd
    config['conv']['num_layers'] = model.config.n_layer
    config['conv']['num_heads'] = model.config.n_head
    config['conv']['num_blocks'] = 2

    text_tokenizer = AutoTokenizer.from_pretrained(BACKBONE_MODEL_MAPPINGS[args.backbone_model]) 
    
    text_encoder = MODEL_TYPES_MAPPING[args.backbone_model].from_pretrained(BACKBONE_MODEL_MAPPINGS[args.backbone_model])
    # resize model and tokenizer
    new_embeddings = torch.load(os.path.join(args.movie_embedding_file_path,args.backbone_model,'movie_embedding.pt'))
    token_begin = len(text_tokenizer)
    new_num_tokens = len(text_tokenizer) + new_embeddings.shape[0]
    text_encoder = resize_token_embeddings(text_encoder,new_embeddings,new_num_tokens)
    text_tokenizer = add_tokens_for_tokenizer(text_tokenizer,args.data_type)
    config['prompt'] = {}
    config['prompt']['msk_context_idx'] = text_tokenizer.mask_token_id
    config['prompt']['pad_context_idx'] = text_tokenizer.pad_token_id
    config['prompt']['eos_context_idx'] = text_tokenizer.eos_token_id


    #load the model
    text_encoder = CRSModel(text_encoder,
                    model_type =args.backbone_model,
                    query_position = args.query_position,
                    opt=config,
                    add_knowledge_prompt = args.add_knowledge_prompt,
                    conv=True)
    if args.load_trained_model:
        check_point= torch.load(args.load_trained_model_path)
        logger.info('[Train Conv after Rec]')
        text_encoder.load_state_dict(check_point,strict=False)
        logger.info(f'[Load the trained model]: {args.load_trained_model_path}')
    
    # freeze the model
    fix_modules = [model, text_encoder]
    for module in fix_modules:
        module.requires_grad_(False)
    # unfreeze the conv layers
    for name, param in text_encoder.named_parameters():
        if 'conv' in name:
            param.requires_grad_(True)
            logger.info(f'[Unfreeze]: {name}')
    
    # faiss
    """
    dstore_keys: 用于query算相似度排序
    dstore_vals: 用于算multihead attention
    """

    dstore_keys = np.load(os.path.join(args.dstore_path,'history_representations.npy'))
    dstore_vals = np.load(os.path.join(args.dstore_path,'response_representations.npy'))
    logger.info(f"[dstore_keys]:{dstore_keys.shape}")
    logger.info("  [Begin FAISS TO GPU]")
    provider = faiss.StandardGpuResources()  # use a single GPU
    faiss_index = faiss.IndexFlatIP(768)
    faiss_index = faiss.index_cpu_to_gpu(provider, 0, faiss_index)
    faiss_index.add(dstore_keys)
    logger.info("  [Finish FAISS TO GPU]")

    # optim & amp
    modules = [text_encoder]
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for model in modules for n, p in model.named_parameters()
                       if not any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for model in modules for n, p in model.named_parameters()
                       if any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    # data
    train_dataset = CRSDatasetConversation(
        args.data_file_path,args.backbone_model,args.data_type,'train', 
        word_pad_index = config['vocab']['pad_word_idx'],entity_pad_index = config['vocab']['pad_entity_idx'],
        context_tokenizer=text_tokenizer, gen_tokenizer=tokenizer,
        debug=args.debug,
        context_max_length=config['token_max_length'], resp_max_length=config['resp_max_length'],
        entity_max_length=config['entity_max_length'],word_max_length=config['word_max_length'],
        prompt_max_length=config['token_max_length'],
        reload_data=args.reload_data,save_data=args.save_data
    )
    valid_dataset = CRSDatasetConversation(
        args.data_file_path,args.backbone_model,args.data_type,'valid', 
        word_pad_index = config['vocab']['pad_word_idx'],entity_pad_index = config['vocab']['pad_entity_idx'],
        context_tokenizer=text_tokenizer, gen_tokenizer=tokenizer,
        debug=args.debug,
        context_max_length=config['token_max_length'], resp_max_length=config['resp_max_length'],
        entity_max_length=config['entity_max_length'],word_max_length=config['word_max_length'],
        prompt_max_length=config['token_max_length'],
        reload_data=args.reload_data,save_data=args.save_data
    )
    test_dataset = CRSDatasetConversation(
        args.data_file_path,args.backbone_model,args.data_type,'test', 
        word_pad_index = config['vocab']['pad_word_idx'],entity_pad_index = config['vocab']['pad_entity_idx'],
        context_tokenizer=text_tokenizer, gen_tokenizer=tokenizer,
        debug=args.debug,
        context_max_length=config['token_max_length'], resp_max_length=config['resp_max_length'],
        entity_max_length=config['entity_max_length'],word_max_length=config['word_max_length'],
        prompt_max_length=config['token_max_length'],
        reload_data=args.reload_data,save_data=args.save_data
    )
    # dataloader
    data_collator_teacher = CRSDataConversationCollator(
        gen_tokenizer=tokenizer,  use_amp=accelerator.use_fp16, debug=args.debug, gen=False,
        ignore_pad_token_for_loss=args.ignore_pad_token_for_loss,
        context_max_length=config['token_max_length'] + config['resp_max_length'],   
        vocab=config['prompt'],
        device=accelerator.device,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=data_collator_teacher,
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=args.per_device_eval_batch_size,
        num_workers=args.num_workers,
        collate_fn=data_collator_teacher,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.per_device_eval_batch_size,
        num_workers=args.num_workers,
        collate_fn=data_collator_teacher,
    )
    data_collator_generator = CRSDataConversationCollator(
        gen_tokenizer=tokenizer,  use_amp=accelerator.use_fp16, debug=args.debug, gen=True,
        ignore_pad_token_for_loss=args.ignore_pad_token_for_loss,
        context_max_length=config['token_max_length'],   
        vocab=config['prompt'],
        device=accelerator.device,
    )
    valid_gen_dataloader = DataLoader(
        valid_dataset,
        batch_size=args.per_device_eval_batch_size,
        num_workers=args.num_workers,
        collate_fn=data_collator_generator,
    )
    test_gen_dataloader = DataLoader(
        test_dataset,
        batch_size=args.per_device_eval_batch_size,
        num_workers=args.num_workers,
        collate_fn=data_collator_generator,
    )
    local_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    gen_file_path = os.path.join('gen', f'gen_{local_time}.jsonl')
    evaluator = ConvEvaluator(tokenizer=tokenizer, log_file_path=gen_file_path)
    model,text_encoder, optimizer = accelerator.prepare(
        model,text_encoder, optimizer
    )
    # step, epoch, batch size
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    completed_steps = 0
    # lr_scheduler
    lr_scheduler = get_linear_schedule_with_warmup(optimizer, args.num_warmup_steps, args.max_train_steps)
    lr_scheduler = accelerator.prepare(lr_scheduler)
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = 'Linear'
        accelerator.init_trackers("conversation", experiment_config)
    # training info
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  Query position: {args.query_position}")
    if args.debug:
        logger.info(f"  Debugging mode")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)

    # save model with best metric
    metric, mode = 'loss', -1
    assert mode in (-1, 1)
    if mode == 1:
        best_metric = 0
    else:
        best_metric = float('inf')
    # best_metric_dir = os.path.join(args.output_dir, 'best')
    # os.makedirs(best_metric_dir, exist_ok=True)
    best_model_epoch = 0
    best_model_state_dict = None
    # train loop
    for epoch in range(args.num_train_epochs):
        train_loss = []
        text_encoder.train()
        model.eval()
        for step, batch in enumerate(train_dataloader):
            # get the representation for retrieval
            query_representation,last_hidden_states = text_encoder(batch['prompt_batch'],mode ='query_representation')
            # retrieve the representation
            faiss_aug = faiss_search(representation = query_representation,
                                    faiss_index = faiss_index,
                                    dstore_keys = dstore_keys,
                                    dstore_vals = dstore_vals,
                                    dstore_labels = None,
                                    k = args.retrieval_k,
                                    device = accelerator.device)
            # fuse the prompt and the context
            assert faiss_aug['batch_faiss_vecs'].shape[0] == last_hidden_states.shape[0]
            batch['prompt_batch']['faiss_aug_representation'] = faiss_aug['batch_faiss_vecs']
            batch['prompt_batch']['last_hidden_states'] = last_hidden_states
            prompt_embeds = text_encoder(
                batch['prompt_batch'], mode='faiss_aug_conversation'
            )
            # generate the response
            batch['context_batch']['prompt_embeds'] = prompt_embeds

            loss = model(**batch['context_batch'], conv=True,
                         conv_labels=batch['resp_batch']).conv_loss / args.gradient_accumulation_steps
            accelerator.backward(loss)
            train_loss.append(float(loss))
            # optim step
            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                if args.max_grad_norm is not None:
                    accelerator.clip_grad_norm_(text_encoder.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                progress_bar.update(1)
                completed_steps += 1
                if args.with_tracking:
                    accelerator.log({'loss': np.mean(train_loss) * args.gradient_accumulation_steps})

            if completed_steps >= args.max_train_steps:
                break

        # metric
        train_loss = np.mean(train_loss) * args.gradient_accumulation_steps
        logger.info(f'epoch {epoch} train loss {train_loss}')

        del train_loss, batch

        # dev
        valid_loss = []
        text_encoder.eval()
        for batch in tqdm(valid_dataloader, disable=not accelerator.is_local_main_process):
            with torch.no_grad():
                query_representation,last_hidden_states = text_encoder(batch['prompt_batch'],mode ='query_representation')
                # retrieve the representation
                faiss_aug = faiss_search(representation = query_representation,
                                        faiss_index = faiss_index,
                                        dstore_keys = dstore_keys,
                                        dstore_vals = dstore_vals,
                                        dstore_labels = None,
                                        k = args.retrieval_k,
                                        device = accelerator.device)
                # fuse the prompt and the context
                assert faiss_aug['batch_faiss_vecs'].shape[0] == last_hidden_states.shape[0]
                batch['prompt_batch']['faiss_aug_representation'] = faiss_aug['batch_faiss_vecs']
                batch['prompt_batch']['last_hidden_states'] = last_hidden_states
                prompt_embeds = text_encoder(
                    batch['prompt_batch'], mode='faiss_aug_conversation'
                )
                # generate the response
                batch['context_batch']['prompt_embeds'] = prompt_embeds
                loss = model(**batch['context_batch'], conv=True, conv_labels=batch['resp_batch']).conv_loss
                valid_loss.append(float(loss))

        evaluator.log_file.write(f'\n\n*** valid-{evaluator.log_cnt} ***\n\n')
        for batch in tqdm(valid_gen_dataloader, disable=not accelerator.is_local_main_process):
            with torch.no_grad():
                query_representation,last_hidden_states = text_encoder(batch['prompt_batch'],mode ='query_representation')
                # retrieve the representation
                faiss_aug = faiss_search(representation = query_representation,
                                        faiss_index = faiss_index,
                                        dstore_keys = dstore_keys,
                                        dstore_vals = dstore_vals,
                                        dstore_labels = None,
                                        k = args.retrieval_k,
                                        device = accelerator.device)
                # fuse the prompt and the context
                assert faiss_aug['batch_faiss_vecs'].shape[0] == last_hidden_states.shape[0]
                batch['prompt_batch']['faiss_aug_representation'] = faiss_aug['batch_faiss_vecs']
                batch['prompt_batch']['last_hidden_states'] = last_hidden_states
                prompt_embeds = text_encoder(
                    batch['prompt_batch'], mode='faiss_aug_conversation'
                )
                # generate the response
                batch['context_batch']['prompt_embeds'] = prompt_embeds

                gen_seqs = accelerator.unwrap_model(model).generate(
                    **batch['context_batch'],
                    max_new_tokens=args.max_gen_len,
                    no_repeat_ngram_size=3
                )
                gen_resp_ids = []
                for gen_seq, length in zip(gen_seqs, batch['context_len_batch']):
                    gen_seq = [token_id for token_id in gen_seq if token_id != tokenizer.pad_token_id]
                    gen_resp_ids.append(gen_seq[length:])
                evaluator.evaluate(gen_resp_ids, batch['resp_batch'], log=accelerator.is_local_main_process)

        # metric
        accelerator.wait_for_everyone()
        report = evaluator.report()
        valid_report = {}
        for k, v in report.items():
            valid_report[f'valid/{k}'] = v
        valid_loss = np.mean(valid_loss)
        valid_report['valid/loss'] = valid_loss
        valid_report['epoch'] = epoch
        logger.info(valid_report)
        if args.with_tracking:
            accelerator.log(
                valid_report
            )
        evaluator.reset_metric()

        if valid_report[f'valid/{metric}'] * mode > best_metric * mode:
            best_metric = valid_report[f'valid/{metric}']
            best_model_epoch = epoch
            best_model_state_dict = copy.deepcopy(text_encoder.state_dict())
            logger.info(f'[new best model saved] at epoch {best_model_epoch}')


        # test
        test_loss = []
        text_encoder.eval()
        for batch in tqdm(test_dataloader, disable=not accelerator.is_local_main_process):
            with torch.no_grad():
                query_representation,last_hidden_states = text_encoder(batch['prompt_batch'],mode ='query_representation')
                # retrieve the representation
                faiss_aug = faiss_search(representation = query_representation,
                                        faiss_index = faiss_index,
                                        dstore_keys = dstore_keys,
                                        dstore_vals = dstore_vals,
                                        dstore_labels = None,
                                        k = args.retrieval_k,
                                        device = accelerator.device)
                # fuse the prompt and the context
                assert faiss_aug['batch_faiss_vecs'].shape[0] == last_hidden_states.shape[0]
                batch['prompt_batch']['faiss_aug_representation'] = faiss_aug['batch_faiss_vecs']
                batch['prompt_batch']['last_hidden_states'] = last_hidden_states
                prompt_embeds = text_encoder(
                    batch['prompt_batch'], mode='faiss_aug_conversation'
                )
                # generate the response
                batch['context_batch']['prompt_embeds'] = prompt_embeds
                loss = model(**batch['context_batch'], conv=True, conv_labels=batch['resp_batch']).conv_loss
                test_loss.append(float(loss))

        evaluator.log_file.write(f'\n*** test-{evaluator.log_cnt} ***\n\n')
        for batch in tqdm(test_gen_dataloader, disable=not accelerator.is_local_main_process):
            with torch.no_grad():
                query_representation,last_hidden_states = text_encoder(batch['prompt_batch'],mode ='query_representation')
                # retrieve the representation
                faiss_aug = faiss_search(representation = query_representation,
                                        faiss_index = faiss_index,
                                        dstore_keys = dstore_keys,
                                        dstore_vals = dstore_vals,
                                        dstore_labels = None,
                                        k = args.retrieval_k,
                                        device = accelerator.device)
                # fuse the prompt and the context
                assert faiss_aug['batch_faiss_vecs'].shape[0] == last_hidden_states.shape[0]
                batch['prompt_batch']['faiss_aug_representation'] = faiss_aug['batch_faiss_vecs']
                batch['prompt_batch']['last_hidden_states'] = last_hidden_states
                prompt_embeds = text_encoder(
                    batch['prompt_batch'], mode='faiss_aug_conversation'
                )
                # generate the response
                batch['context_batch']['prompt_embeds'] = prompt_embeds

                gen_seqs = accelerator.unwrap_model(model).generate(
                    **batch['context_batch'],
                    max_new_tokens=args.max_gen_len,
                    no_repeat_ngram_size=3,
                )
                gen_resp_ids = []
                for gen_seq, length in zip(gen_seqs, batch['context_len_batch']):
                    gen_seq = [token_id for token_id in gen_seq if token_id != tokenizer.pad_token_id]
                    gen_resp_ids.append(gen_seq[length:])
                evaluator.evaluate(gen_resp_ids, batch['resp_batch'], log=accelerator.is_local_main_process)

        # metric
        accelerator.wait_for_everyone()
        report = evaluator.report()
        test_report = {}
        for k, v in report.items():
            test_report[f'test/{k}'] = v
        test_loss = np.mean(test_loss)
        test_report['test/loss'] = test_loss
        test_report['epoch'] = epoch
        logger.info(test_report)
        if args.with_tracking:
            accelerator.log(test_report)
        evaluator.reset_metric()

        evaluator.log_cnt += 1

    accelerator.end_training()
    # 模型保存
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        os.makedirs(args.output_dir,exist_ok=True)
        accelerator.save(best_model_state_dict,os.path.join(args.output_dir,f'{args.backbone_model}_conv.pth'))
        logger.info(f'[Model saved]')
