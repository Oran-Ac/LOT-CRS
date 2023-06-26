import argparse
from config import redial_config
import json
import logging
import math
import os
import faiss
from torch.nn import functional as F
from torch import nn
from dataset import CRSDatasetRecommendation,CRSDataRecommendationCollator
import pandas as pd
import datasets
import torch
from modeling_kgsf import KGSF
from modeling_crs import CRSModel
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from torchviz import make_dot
import transformers
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import set_seed,DistributedDataParallelKwargs
from transformers import (
    MODEL_MAPPING,
    AdamW,
    # DataCollatorForLanguageModeling,
    # SchedulerType,
    get_linear_schedule_with_warmup,
    AutoModelForMaskedLM,
    AutoTokenizer
)
from transformers.utils.versions import require_version
from args import BACKBONE_MODEL_MAPPINGS,Movie_Name_Path,AT_TOKEN_NUMBER
from utils import resize_token_embeddings,add_tokens_for_tokenizer
import numpy as np
from utils import faiss_search,faiss_search_train_retrieval
from evaluate_rec import RecEvaluator

logger = get_logger(__name__)
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

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
        '--backbone_model',
        type=str,
        choices=['bert','bart'],
        help = 'backbone'
    )
    parser.add_argument(
        '--kgsf_model_path',
        type=str,
        default=None,
        help = 'path to load the trained kgsf model'
    )
    parser.add_argument(
        '--query_position',
        type=str,
        default=None,
    )
    parser.add_argument(
        '--movie_embedding_file_path',
        type=str,
        default='./pretrainDataset',
        help = 'path to load the movie embedding'
    )
    parser.add_argument(
        '--dstore_path',
        type=str,
        default='./dstore',
        help = 'path to dstore'
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
        '--learning_rate',
        type=float,
    )
    parser.add_argument(
        '--gradient_accumulation_steps',
        type=int,
    )
    parser.add_argument(
        '--num_train_epochs',
        type=int,
    )
    parser.add_argument(
        '--max_train_steps',
        type=int,
        default=None
    )
    parser.add_argument(
        '--num_warmup_steps',
        type=int,
        default=0
    )
    parser.add_argument(
        '--retrieval_k',
        type=int,
    )
    parser.add_argument('--correct_k_num', type=int,default=4, help='number of k')
    parser.add_argument('--fake_k_num', type=int,default=512,  help='number of k')
    parser.add_argument('--temperature', type=int,default=1, help='temperature for retrieval')
    parser.add_argument(
        '--beta',
        type=float,
    )
    parser.add_argument(
        '--alpha',
        type=float,
    )
    parser.add_argument(
        '--load_trained_model',
        action='store_true',
    )
    parser.add_argument(
        '--load_trained_model_path',
        type=str,
        default=None
    )
    parser.add_argument(
        '--faiss_weight',
        type=float,
        default=1.0
    )
    parser.add_argument(
        '--add_knowledge_prompt',
        action='store_true',
    )
    args = parser.parse_args()
    return args
    
    


def evaluate_kgsf(KgsfModel,test_dataloader,evaluator,accelerator):
    evaluator.reset_metric()
    KgsfModel.eval()
    for step, batch in enumerate(test_dataloader):
        with torch.no_grad():
            kgsf_outputs = KgsfModel.recommend(batch)
        rec_scores = kgsf_outputs['rec_scores'] #[0-6924]
        labels = batch['rec_dbpedia_movie_label_batch'].tolist() #[0-6924]
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
    return json.dumps(test_report,indent=2)


def main():
    
    args = parse_args()
    config = redial_config(args.config_path,args.data_type)


    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will pick up all supported trackers in the environment
    accelerator = Accelerator(log_with="all", logging_dir=args.output_dir) if args.with_tracking else Accelerator()
    # Make one log on every process with the configuration for debugging.
    if accelerator.is_main_process:
        suffix = 'init'
        if args.load_trained_model:
            suffix = args.load_trained_model_path.replace('.pth','')
            suffix = suffix.split('/')[-1]
        logging.basicConfig( 
                    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                    datefmt="%m/%d/%Y %H:%M:%S",
                    level=logging.INFO,           
                    filename=f'recommend_{suffix}_{args.query_position}.log',
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

        if accelerator.is_main_process:
            logger.info("Training new model from scratch")
    logger.info(f"*[BACK BONE MODEL]*: {args.backbone_model}")

    backbone_model = AutoModelForMaskedLM.from_pretrained(BACKBONE_MODEL_MAPPINGS[args.backbone_model])
    tokenizer = AutoTokenizer.from_pretrained(BACKBONE_MODEL_MAPPINGS[args.backbone_model]) 
    KGSFModel =  KGSF(config,accelerator.device,args.kgsf_model_path)

    # resize model and tokenizer
    new_embeddings = torch.load(os.path.join(args.movie_embedding_file_path,args.backbone_model,'movie_embedding.pt'))
    token_begin = len(tokenizer)
    new_num_tokens = len(tokenizer) + new_embeddings.shape[0]
    backbone_model = resize_token_embeddings(backbone_model,new_embeddings,new_num_tokens)
    # assert backbone_model.config.vocab_size == 37446


    #load the model
    if args.load_trained_model:
        check_point= torch.load(args.load_trained_model_path)
        logger.info('[Train Retrieval after Pretrain]')
        backbone_model.load_state_dict(check_point,strict=True)
        logger.info(f'[Load the trained model]: {args.load_trained_model_path}')
    
    tokenizer = add_tokens_for_tokenizer(tokenizer)
    config['vocab']['msk_context_idx'] = tokenizer.mask_token_id
    config['vocab']['pad_context_idx'] = tokenizer.pad_token_id
    config['vocab']['eos_context_idx'] = tokenizer.eos_token_id
    
    model = CRSModel(backbone_model,model_type =args.backbone_model,query_position = args.query_position,opt=config)
    kd_function = nn.KLDivLoss(reduction='batchmean') # 真实kl
    retrieval_function = nn.CrossEntropyLoss()
    evaluator = RecEvaluator()
    
    # 加载训练集 测试集
    # undo: 数据集处理，应该和unicrs对齐（1，添加word分词 2用@替换电影）
    # 分词器1.添加带@的新词 2. 分词 3. 获得@的表示并保存，便于后续进行初始化
    train_dataset = CRSDatasetRecommendation(args.data_file_path,args.data_type,'train',word_pad_index = config['vocab']['pad_word_idx'],
                                            entity_pad_index = config['vocab']['pad_entity_idx'],context_tokenizer = tokenizer,
                                            dbpedia_tokenzier = config['graph']['entity2id'],word_tokenizer = config['graph']['token2id'],
                                            token_max_length = config['token_max_length'], entity_max_length=config['entity_max_length'],word_max_length=config['word_max_length'],
                                            prompt_text = config['prompt_text'],
                                            save_data= args.save_data,reload_data = args.reload_data,
                                            )
    test_dataset = CRSDatasetRecommendation(args.data_file_path,args.data_type,'test',word_pad_index = config['vocab']['pad_word_idx'],
                                            entity_pad_index = config['vocab']['pad_entity_idx'],context_tokenizer = tokenizer,
                                            dbpedia_tokenzier = config['graph']['entity2id'],word_tokenizer = config['graph']['token2id'],
                                            token_max_length = config['token_max_length'], entity_max_length=config['entity_max_length'],word_max_length=config['word_max_length'],
                                            prompt_text = config['prompt_text'],
                                            save_data= args.save_data,reload_data = args.reload_data,
                                            )
    valid_dataset = CRSDatasetRecommendation(args.data_file_path,args.data_type,'valid',word_pad_index = config['vocab']['pad_word_idx'],
                                            entity_pad_index = config['vocab']['pad_entity_idx'],context_tokenizer = tokenizer,
                                            dbpedia_tokenzier = config['graph']['entity2id'],word_tokenizer = config['graph']['token2id'],
                                            token_max_length = config['token_max_length'], entity_max_length=config['entity_max_length'],word_max_length=config['word_max_length'],
                                            prompt_text = config['prompt_text'],
                                            save_data= args.save_data,reload_data = args.reload_data,
                                            )
    data_collator = CRSDataRecommendationCollator(config['vocab'])
    # DataLoaders creation:
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(valid_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)
    test_dataloader = DataLoader(test_dataset, collate_fn=data_collator, batch_size=args.per_device_test_batch_size)
    
    logger.info(f'[Loading PretrainDataset knn encode by Pretrain]: {args.dstore_path}')
    dstore_keys = np.load(os.path.join(args.dstore_path,args.backbone_model,'keys.npy'))
    dstore_vals = np.load(os.path.join(args.dstore_path,args.backbone_model,'vals.npy'))
    logger.info(f"[dstore_keys]:{dstore_keys.shape}")
    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    # logger.info([n for n, p in model.named_parameters()])
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # On TPU, the tie weights in our model have been disconnected, so we need to restore the ties.
    if accelerator.distributed_type == DistributedType.TPU:
        model.tie_weights()
     # Note -> the training dataloader needs to be prepared before we grab his length below (cause its length will be
    # shorter in multiprocess)

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )
    # Prepare everything with our `accelerator`.
    model, KGSFModel, optimizer, train_dataloader, eval_dataloader,test_dataloader,lr_scheduler = accelerator.prepare(
        model,KGSFModel, optimizer, train_dataloader, eval_dataloader,test_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = 'Linear'
        accelerator.init_trackers("recommend", experiment_config)
    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    if accelerator.is_main_process:
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num Epochs = {args.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {args.max_train_steps}")
        logger.info(f"  Query position: {args.query_position}")
    # test KSGF first
    logger.info(f"  [TEST KGSF MODEL]")
    test_kgsf = evaluate_kgsf(KGSFModel,test_dataloader,evaluator,accelerator)
    if args.with_tracking:
        accelerator.log(test_kgsf)
    else:
        logger.info(test_kgsf)

    # faiss_to_gpu 
    # todo:检查能否正常工作,特别是指定gpu
    provider = faiss.StandardGpuResources()  # use a single GPU
    faiss_index = faiss.IndexFlatIP(768)
    faiss_index = faiss.index_cpu_to_gpu(provider, 0, faiss_index)
    faiss_index.add(dstore_keys)
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_main_process)
    completed_steps = 0
    for epoch in range(args.num_train_epochs):
        model.train()
        KGSFModel.eval()
        total_loss = []
        kd_loss = []
        rec_loss = []
        faiss_loss = []
        for step, batch in enumerate(train_dataloader):
            '''具体训练过程
            1.先得到表示
            2.召回
            3.推荐
            4.蒸馏loss
            '''
            with torch.no_grad():
                kgsf_outputs = KGSFModel.recommend(batch)
            batch['word_representations'] = kgsf_outputs['word_representations']
            batch['entity_representations'] = kgsf_outputs['entity_representations']
            batch['entity_graph_representations'] = kgsf_outputs['entity_graph_representations']
            query_representation,last_hidden_states = model(batch,mode ='query_representation',faiss_weight=args.faiss_weight,add_knowledge_prompt=args.add_knowledge_prompt)
            batch['last_hidden_states'] = last_hidden_states
            #done:检查这里的device加载对没有
            faiss_aug = faiss_search(query_representation,faiss_index,dstore_keys,dstore_vals,args.retrieval_k,accelerator.device)
            # print(query_representation.shape)
            # print(last_hidden_states.shape)
            # print(faiss_aug['batch_faiss_vecs'].shape)
            assert faiss_aug['batch_faiss_vecs'].shape[0] == last_hidden_states.shape[0]
            batch['faiss_aug_representation'] = faiss_aug['batch_faiss_vecs']
            # recommend
            outputs =  model(batch,mode ='faiss_aug_recommendation',faiss_weight=args.faiss_weight,add_knowledge_prompt=args.add_knowledge_prompt)
            # 蒸馏
            # todo:check 顺序是matched
            student_predict = outputs['movie_scores']
            teacher_predict = kgsf_outputs['rec_scores']

            student_predict = F.log_softmax(student_predict,dim=1)
            teacher_predict = F.softmax(teacher_predict,dim=1)
            loss_kd = kd_function(student_predict,teacher_predict)

            # 召回的loss
            ref,logist = faiss_search_train_retrieval(faiss_index,dstore_keys,dstore_vals,query_representation,args.correct_k_num,args.fake_k_num,accelerator.device,batch['label_batch'].tolist()) #[bz,n,hidden]
            labels = torch.zeros(logist.shape[0],device=accelerator.device).long()
            logist = logist.to(accelerator.device)
            retrieval_loss = None
            
            if logist is not None: #长尾数据可能出现为none的情况
                retrieval_loss = retrieval_function(torch.div(logist,args.temperature),labels)
            if retrieval_loss is None:
                loss =  outputs['rec_loss'] + args.beta * loss_kd
            else:
                loss = outputs['rec_loss'] + args.beta * loss_kd + args.alpha* retrieval_loss
            
            # We keep track of the loss at each epoch
            
            total_loss.append(float(loss.detach()))
            kd_loss.append(float(loss_kd.detach()))
            faiss_loss.append(float(retrieval_loss.detach()))
            rec_loss.append(float(outputs['rec_loss'].detach()))

            loss = loss / args.gradient_accumulation_steps
            accelerator.backward(loss)
            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1
        logger.info(f'epoch {epoch}\n train loss: {np.mean(total_loss)}, kd_loss: {np.mean(kd_loss)}, faiss_loss:{np.mean(faiss_loss)}, rec_loss: {np.mean(rec_loss)}')
        # test
        model.eval()
        test_loss = []
        logger.info('[test]')
        for step, batch in tqdm(enumerate(test_dataloader),total=len(test_dataloader)):

            with torch.no_grad():
                
                kgsf_outputs = KGSFModel.recommend(batch)
                batch['word_representations'] = kgsf_outputs['word_representations']
                batch['entity_representations'] = kgsf_outputs['entity_representations']
                batch['entity_graph_representations'] = kgsf_outputs['entity_graph_representations']
                query_representation,last_hidden_states = model(batch,mode ='query_representation',faiss_weight=args.faiss_weight,add_knowledge_prompt=args.add_knowledge_prompt)
                batch['last_hidden_states'] = last_hidden_states
                #done:检查这里的device加载对没有
                faiss_aug = faiss_search(query_representation,faiss_index,dstore_keys,dstore_vals,args.retrieval_k,accelerator.device)
                # print(query_representation.shape)
                # print(last_hidden_states.shape)
                # print(faiss_aug['batch_faiss_vecs'].shape)
                assert faiss_aug['batch_faiss_vecs'].shape[0] == last_hidden_states.shape[0]
                batch['faiss_aug_representation'] = faiss_aug['batch_faiss_vecs']
                # recommend
                outputs =  model(batch,mode ='faiss_aug_recommendation',faiss_weight=args.faiss_weight,add_knowledge_prompt=args.add_knowledge_prompt)
            test_loss.append(float(outputs['rec_loss']))
            rec_scores = outputs['movie_scores'] #[0-6924] /[0-len(self.movie-id)]
            ranks = torch.topk(rec_scores, k=50, dim=-1).indices.tolist()
            # 下面的要求是让区间在【0-6924】之间，方便后面的长尾的指标计算
            if args.add_knowledge_prompt:
                labels = batch['rec_dbpedia_movie_label_batch']
            else:
                labels = batch['label_batch']- token_begin
            labels = labels.tolist() # undo: check
            evaluator.evaluate(ranks,labels)
        
        # metric
        report = accelerator.gather(evaluator.report())
        for k, v in report.items():
            report[k] = v.sum().item()

        test_report = {}
        for k, v in report.items():
            if  'recall' in k:
                test_report[f'{args.backbone_model}/test/{k}'] = v / report['count']
            elif 'coverage' in k:
                test_report[f'{args.backbone_model}/test/{k}'] = v / len(evaluator.movie2num)
            elif 'Correcttail' in k:
                test_report[f'{args.backbone_model}f/test/{k}'] = v / evaluator.longTail_test
            elif 'tail_coverage' in k:
                test_report[f'{args.backbone_model}/test/{k}'] = v / evaluator.longTail_total
        
        

        test_report['test/loss'] = np.mean(test_loss)
        test_report['epoch'] = epoch
        test_report = json.dumps(test_report,indent=2)
        logger.info(f'{test_report}')
        if args.with_tracking:
                accelerator.log(
            test_report
        )
        evaluator.reset_metric()
    
    # undo:模型保存

            
if __name__ == '__main__':
    main()

