import argparse
from utils.config import redial_config
import json
import logging
import os
from torch.nn import functional as F
from torch import nn
from dataset.dataset import KGSFDataset,KGSFDatasetCollator
import pandas as pd
import datasets
import torch
from model.modeling_kgsf import KGSF
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import set_seed,DistributedDataParallelKwargs
import transformers
from transformers import (
    AdamW,
    # DataCollatorForLanguageModeling,
    # SchedulerType,
    get_linear_schedule_with_warmup,
)
from transformers.utils.versions import require_version
from utils.args import BACKBONE_MODEL_MAPPINGS,Movie_Name_Path,AT_TOKEN_NUMBER
from utils.utils import resize_token_embeddings,add_tokens_for_tokenizer
import numpy as np
from utils.evaluate_rec import RecEvaluator
import math

logger = get_logger(__name__)
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")
# MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
# MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a Masked Language Modeling task")
    parser.add_argument(
        "--config_path",
        type=str,
        default='./modeling_kgsf.yaml',
        help = 'path of config for kgsf model'
    )
    parser.add_argument(
        "--data_type",
        type=str,
        choices=['redial','inspire'],
        help = 'data used to train and test'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        help = 'output dir'
    )
    parser.add_argument(
        '--with_tracking',
        action='store_true',
        help = 'output log or not'
    )
    parser.add_argument(
        '--seed',
        type=str,
        default=None,
        help = 'seed'
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
    parser.add_argument(
        '--per_device_train_batch_size',
        type=int,
        default=16,
        help = 'bz to train'
    )
    parser.add_argument(
        '--per_device_eval_batch_size',
        type=int,
        default=16,
        help = 'bz to eval'
    )
    parser.add_argument(
        '--per_device_test_batch_size',
        type=int,
        default=16,
        help = 'bz to test'
    )
    parser.add_argument(
        '--weight_decay',
        default=0.0,
        type=float,
    )
    parser.add_argument(
        '--pre_learning_rate',
        type=float,
    )
    parser.add_argument(
        '--rec_learning_rate',
        type=float,
    )
    parser.add_argument(
        '--gradient_accumulation_steps',
        type=int,
    )
    parser.add_argument(
        '--pre_num_train_epochs',
        type=int,
    )
    parser.add_argument(
        '--max_pre_train_steps',
        type=int,
        default=None
    )
    parser.add_argument(
        '--rec_num_train_epochs',
        type=int,
    )
    parser.add_argument(
        '--max_rec_train_steps',
        type=int,
        default=None
    )
    parser.add_argument(
        '--num_warmup_steps',
        type=int,
        default=0
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help = 'debug mode'
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default=None,
        help="Where to store the logs.",
    )
    args = parser.parse_args()
    return args
    
    


def evaluate_kgsf(KgsfModel,test_dataloader,evaluator,accelerator):
    evaluator.reset_metric()
    KgsfModel.eval()
    for step, batch in enumerate(test_dataloader):
        with torch.no_grad():
            kgsf_outputs = KgsfModel.recommend(batch,'test')
        rec_scores = kgsf_outputs['rec_movie_scores'] #(bz,#n_movie)
        labels = batch['movie_batch'].tolist() #(bz,1) #n_movie
        ranks = torch.topk(rec_scores, k=50, dim=-1).indices.tolist()
        evaluator.evaluate(ranks,labels)
    
    # metric
    report = accelerator.gather(evaluator.report())
    for k, v in report.items():
        report[k] = v.sum().item()

    test_report = {}
    for k, v in report.items():
        if  'recall' in k:
            test_report[f'kgsf/test/{k}'] = v / report['count']
        elif 'coverage' in k:
            test_report[f'kgsf/test/{k}'] = v / len(evaluator.movie2num)
        elif 'Correcttail' in k:
            test_report[f'kgsf/test/{k}'] = v / evaluator.longTail_test
        elif 'tail_coverage' in k:
            test_report[f'kgsf/test/{k}'] = v / evaluator.longTail_total




    evaluator.reset_metric()
    return test_report


def main():
    
    args = parse_args()
    config = redial_config(args.config_path,args.data_type)


    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will pick up all supported trackers in the environment
    accelerator = Accelerator(log_with="wandb", project_dir=args.output_dir) if args.with_tracking else Accelerator()
    # Make one log on every process with the configuration for debugging.
    if accelerator.is_main_process:
        logging.basicConfig( 
                    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                    datefmt="%m/%d/%Y %H:%M:%S",
                    level=logging.INFO,           
                    filename=os.path.join(args.logging_dir,f'kgsf.log'),
                    filemode='a',##模式，有w和a，w就是写模式，每次都会重新写日志，覆盖之前的日志                    #a是追加模式，默认如果不写的话，就是追加模式                    format=                    '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'                    #日志格式
                    )
        logger.info(json.dumps(vars(args),indent=2))
        logger.info(accelerator.state, main_process_only=False)
    
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        logger.info("Training kgsf model")

    KGSFModel =  KGSF(config,accelerator.device)

    evaluator = RecEvaluator()
    
    # 加载训练集 测试集

    train_dataset = KGSFDataset(args.data_file_path,args.data_type,'train',word_pad_index = config['vocab']['pad_word_idx'],
                                            entity_pad_index = config['vocab']['pad_entity_idx'],
                                            entity_max_length=config['entity_max_length'],word_max_length=config['word_max_length'],
                                            save_data= args.save_data,reload_data = args.reload_data,n_entity = config['graph']['n_entity'],
                                            movie_ids = config['movie_ids'],
                                            debug = args.debug
                                            )
    test_dataset = KGSFDataset(args.data_file_path,args.data_type,'test',word_pad_index = config['vocab']['pad_word_idx'],
                                            entity_pad_index = config['vocab']['pad_entity_idx'],
                                            entity_max_length=config['entity_max_length'],word_max_length=config['word_max_length'],
                                            save_data= args.save_data,reload_data = args.reload_data,n_entity = config['graph']['n_entity'],
                                            movie_ids = config['movie_ids'],
                                            debug = args.debug
                                            )
    valid_dataset = KGSFDataset(args.data_file_path,args.data_type,'valid',word_pad_index = config['vocab']['pad_word_idx'],
                                            entity_pad_index = config['vocab']['pad_entity_idx'],
                                            entity_max_length=config['entity_max_length'],word_max_length=config['word_max_length'],
                                            save_data= args.save_data,reload_data = args.reload_data,n_entity = config['graph']['n_entity'],
                                            movie_ids = config['movie_ids'],
                                            debug = args.debug
                                            )
    data_collator = KGSFDatasetCollator()
    # DataLoaders creation:
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(valid_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)
    test_dataloader = DataLoader(test_dataset, collate_fn=data_collator, batch_size=args.per_device_test_batch_size)
    
    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in KGSFModel.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in KGSFModel.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    # logger.info([n for n, p in model.named_parameters()])
    pre_optimizer = AdamW(optimizer_grouped_parameters, lr=args.pre_learning_rate)
    rec_optimizer = AdamW(optimizer_grouped_parameters, lr=args.rec_learning_rate)

    # On TPU, the tie weights in our model have been disconnected, so we need to restore the ties.
    if accelerator.distributed_type == DistributedType.TPU:
        model.tie_weights()
     # Note -> the training dataloader needs to be prepared before we grab his length below (cause its length will be
    # shorter in multiprocess)

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_pre_train_steps is None:
        args.max_pre_train_steps = args.pre_num_train_epochs * num_update_steps_per_epoch
    else:
        args.pre_num_train_epochs = math.ceil(args.max_pre_train_steps / num_update_steps_per_epoch)
    if args.max_rec_train_steps is None:
        args.max_rec_train_steps = args.rec_num_train_epochs * num_update_steps_per_epoch
    else:
        args.rec_num_train_epochs = math.ceil(args.max_rec_train_steps / num_update_steps_per_epoch)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    args.max_pre_train_steps = args.pre_num_train_epochs * num_update_steps_per_epoch
    args.max_rec_train_steps = args.rec_num_train_epochs * num_update_steps_per_epoch

    # Prepare everything with our `accelerator`.
    KGSFModel, pre_optimizer,rec_optimizer, train_dataloader, eval_dataloader,test_dataloader = accelerator.prepare(
        KGSFModel, pre_optimizer,rec_optimizer, train_dataloader, eval_dataloader,test_dataloader
    )
    # Initialize our Trainer
    if args.with_tracking:
        experiment_config = vars(args)
        accelerator.init_trackers("recommend", experiment_config)
    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    if accelerator.is_main_process:
        logger.info("***** Basic Information *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info("***** Pretrain *****")
        logger.info(f"  Total optimization steps = {args.max_pre_train_steps}")
        logger.info(f"  Num pretrain Epochs = {args.pre_num_train_epochs}")
        logger.info("***** Rec *****")
        logger.info(f"  Total optimization steps = {args.max_rec_train_steps}")
        logger.info(f"  Num pretrain Epochs = {args.rec_num_train_epochs}")
        if args.debug:
            logger.info(f"  Debug Mode")
        logger.info("\n\n")

    # test KSGF first
    logger.info(f"  [REC TEST KGSF MODEL]")
    test_report = evaluate_kgsf(KGSFModel,test_dataloader,evaluator,accelerator)
    logger.info(json.dumps(test_report,indent=2))

    # Only show the progress bar once on each machine.
    # pretrain
    if accelerator.is_main_process:
        logger.info(f"***** Running Pretrain *****")
    progress_bar = tqdm(range(args.max_pre_train_steps), disable=not accelerator.is_main_process)
    completed_steps = 0
    for epoch in range(args.pre_num_train_epochs):
        KGSFModel.train()
        pretrain_loss = []
        for step, batch in enumerate(train_dataloader):
            kgsf_outputs = KGSFModel.pretrain(batch)
            loss = kgsf_outputs['info_loss']
            # We keep track of the loss at each epoch
            pretrain_loss.append(float(loss.detach()))
            loss = loss / args.gradient_accumulation_steps
            accelerator.backward(loss)
            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                pre_optimizer.step()
                pre_optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1
        logger.info(f'epoch {epoch}\n train loss: {np.mean(pretrain_loss)}')
        if args.with_tracking:
            accelerator.log(
                {
                    "epoch": epoch,
                    "pre_total_loss": np.mean(pretrain_loss),
                }
            )

    # test KSGF first
    logger.info(f"  [REC TEST KGSF MODEL]")
    test_report = evaluate_kgsf(KGSFModel,test_dataloader,evaluator,accelerator)
    logger.info(json.dumps(test_report,indent=2))

    # Only show the progress bar once on each machine.
    # rec
    if accelerator.is_main_process:
        logger.info(f"  ***** Running Rec *****")
    progress_bar = tqdm(range(args.max_rec_train_steps), disable=not accelerator.is_main_process)
    completed_steps = 0
    for epoch in range(args.rec_num_train_epochs):
        KGSFModel.train()
        total_rec_loss = []
        total_info_loss = []
        total_loss = []
        for step, batch in enumerate(train_dataloader):
            kgsf_outputs = KGSFModel(batch,'train')
            rec_loss, info_loss = kgsf_outputs['rec_loss'], kgsf_outputs['info_loss']
            
            if info_loss:
                loss = rec_loss + 0.025 * info_loss
            else:
                loss = rec_loss
            # We keep track of the loss at each epoch
            total_rec_loss.append(float(rec_loss.detach()))
            total_info_loss.append(float(info_loss.detach()))
            total_loss.append(float(loss.detach()))

            loss = loss / args.gradient_accumulation_steps
            accelerator.backward(loss)
            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                rec_optimizer.step()
                rec_optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1
        logger.info(f'epoch {epoch}\n train loss: {np.mean(total_loss)}\n rec loss: {np.mean(total_rec_loss)}\n info loss: {np.mean(total_info_loss)}')
        if args.with_tracking:
            accelerator.log(
                {
                    "epoch": epoch,
                    "rec_total_loss": np.mean(total_loss),
                    "rec_loss": np.mean(total_rec_loss),
                    "info_loss": np.mean(total_info_loss),

                }
            )
        # test KSGF
        logger.info(f"  [REC TEST KGSF MODEL]")
        test_report = evaluate_kgsf(KGSFModel,test_dataloader,evaluator,accelerator)
        if args.with_tracking:
            accelerator.log(test_report)
        logger.info(json.dumps(test_report,indent=2))
    
    

    accelerator.end_training()
    # save model
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(KGSFModel)
    if accelerator.is_main_process:
        os.makedirs(args.output_dir,exist_ok=True)
        accelerator.save(unwrapped_model.state_dict(),os.path.join(args.output_dir,f'kgsf.pth'))
        logger.info(f'[Model saved]')

            
if __name__ == '__main__':
    main()

