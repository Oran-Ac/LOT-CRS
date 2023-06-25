import argparse
import pickle
import json
import logging
import math
import os
import time

from dataset.dataset import PretrainDataset,DataCollatorForLanguageModeling
import pandas as pd
import datasets
import torch

from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import set_seed,DistributedDataParallelKwargs
from transformers import (
    MODEL_MAPPING,
    AdamW,
    get_linear_schedule_with_warmup,
    AutoTokenizer,
)
from transformers.utils.versions import require_version
from utils.args import BACKBONE_MODEL_MAPPINGS,MODEL_TYPES_MAPPING
from utils.utils import get_embeddings_for_movie_name,resize_token_embeddings


logger = get_logger(__name__)
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)



def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a Masked Language Modeling task")
    parser.add_argument(
        "--data_file_path", type=str, default=None, help="data path"
    )
    parser.add_argument(
        "--validation_split_percentage",
        default=5,
        help="The percentage of the train set used as validation set in case there's no validation split",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    # parser.add_argument(
    #     "--lr_scheduler_type",
    #     default="linear",
    #     help="The scheduler type to use.",
    #     choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    # )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--mlm_probability", type=float, default=0.15, help="Ratio of tokens to mask for masked language modeling loss"
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to load in all available experiment trackers from the environment and use them for logging.",
    )
    parser.add_argument(
        "--backbone_model",
        type=str,
        default=None,
        help="use different backbone",
    )
    parser.add_argument(
        "--reloadDataset",
        type=bool,
        default=False,
        help="get the dataset",
    )
    parser.add_argument(
        "--test_mode",
        type=bool,
        default=False,
    )
    parser.add_argument(
        "--data_type",
        type=str,
        choices=['redial','inspired'],
        help = 'data used to train and test'
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default=None,
        help="Where to store the logs.",
    )
    args = parser.parse_args()
    if args.backbone_model in ['roberta','bart']: #undo： 不清楚bart prefix重要吗
        args.prefix_matters = True
    else:
        args.prefix_matters = False

    args.output_dir = os.path.join(args.output_dir,args.data_type)
    os.makedirs(args.output_dir,exist_ok=True)
    args.output_dir = os.path.join(args.output_dir,args.backbone_model)
    os.makedirs(args.output_dir,exist_ok=True)
    args.output_dir = os.path.join(args.output_dir,"backbone")
    os.makedirs(args.output_dir,exist_ok=True)

    args.data_file_path = os.path.join(args.data_file_path,args.data_type)
    os.makedirs(args.data_file_path,exist_ok=True)

    os.makedirs(args.logging_dir,exist_ok=True)

    return args



def get_process_data(original_data_path,tokenizer,prefix_matters,test_mode=False):
    logger.info(['Get Process Data'])
    with open(original_data_path,'rb') as f:
        data = pickle.load(f)
    if test_mode:
        data = data[:10]
    new_data = []
    special_token = set()
    for e in tqdm(data,total=len(data)):
        total_keys = 0
        context = ''
        for i_th,each in enumerate(e): 
            if context != '':
                context += tokenizer.sep_token
            context += each['utter']
            special_token |= set(each['keyword'])
            if ( len(each['keyword']) > 0 and total_keys > 2 ) or (i_th == len(e) -1 and len(each['keyword']) > 1):
                new_data.append(context)
            total_keys += len(each['keyword'])
        
    new_dialogue = []
    # hypo：假设出现的keyword都不常见，因此用这种方式进行匹配(undo:是否有更好的解决办法&该假设是否成立)
    special_token_ids = [ tokenizer.encode(each,add_special_tokens=False) for each in special_token]
    if prefix_matters:
        special_token_ids.extend([ tokenizer.encode('Ġ'+each,add_special_tokens=False) for each in special_token])
    # print(special_token_ids  == sum([],special_token_ids))
    temp_special_token_ids = special_token_ids[:]
    special_token_ids = []

    for i in temp_special_token_ids:
        for j in i:
            if prefix_matters:
                if j =='Ġ':
                    continue
            special_token_ids.append(j)
    print('#special token spilt',len(special_token_ids),len(list(set(special_token_ids))))
    special_token_ids = list(set(special_token_ids))
    for n_d in tqdm(new_data,total = len(new_data)):
        context_tokens  = tokenizer.encode(
                            text = n_d,
                            add_special_tokens = True,
                            max_length = 512,
                            padding='max_length',
                            truncation=True
                            )
        context_special_tokens_mask = [1 if e in special_token_ids else 0 for e in context_tokens]
        dia = {
            'context_tokens':context_tokens,
            'tokenize_tokens': tokenizer.tokenize(n_d),
            'context_special_tokens_mask':context_special_tokens_mask
        }
        if test_mode:
            dia['tokenize_tokens'] = tokenizer.tokenize(n_d)
        new_dialogue.append(dia)
    return new_dialogue


def main():

    args = parse_args()
   


    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will pick up all supported trackers in the environment
    accelerator = Accelerator(log_with="wandb", project_dir=args.output_dir) if args.with_tracking else Accelerator()
    # Make one log on every process with the configuration for debugging.
    if accelerator.is_main_process:
        logging.basicConfig(
            filename=os.path.join(args.logging_dir, f"{args.data_type}-{args.backbone_model}-pre-training-{time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())}.log"),
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        logger.info(accelerator.state, main_process_only=False)
    
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)
    if accelerator.is_main_process:
        logger.info("Training new model from scratch")
    logger.info(f"*[BACK BONE MODEL]*: {args.backbone_model}")
    print(MODEL_TYPES_MAPPING[args.backbone_model],BACKBONE_MODEL_MAPPINGS[args.backbone_model])
    model = MODEL_TYPES_MAPPING[args.backbone_model].from_pretrained(BACKBONE_MODEL_MAPPINGS[args.backbone_model])
    tokenizer = AutoTokenizer.from_pretrained(BACKBONE_MODEL_MAPPINGS[args.backbone_model]) 



    # 加载训练集 测试集
    # undo: 这个地方有问题，dataset不能加载此前的，此前的进行过填充，应该1. 如果没有用过该分词处理，则用分词处理，且@的表示应该保留 2. 如果处理后，可以直接加载
    # 分词器1.添加带@的新词 2. 分词 3. 获得@的表示并保存，便于后续进行初始化
    if args.reloadDataset and os.path.exists(os.path.join(args.data_file_path,args.backbone_model,'processed_dialogue_valid.pkl')):

        with open(os.path.join(args.data_file_path,args.backbone_model,'pretrain','processed_dialogue.pkl'),'rb') as f:
            train_dataset = pickle.load(f)
        with open(os.path.join(args.data_file_path,args.backbone_model,'pretrain','processed_dialogue_valid.pkl'),'rb') as f:
            eval_dataset = pickle.load(f)
        
        if args.test_mode:
            train_dataset = train_dataset[:10]
            eval_dataset = eval_dataset[:10]


        train_dataset = PretrainDataset(train_dataset)
        eval_dataset = PretrainDataset(eval_dataset)
        new_embeddings = torch.load(os.path.join(args.data_file_path,args.backbone_model,'movie_embedding.pt'))

        new_num_tokens = len(tokenizer) + new_embeddings.shape[0]

        if accelerator.is_main_process:
            logger.info('[Load dataset]')
    else:
        if not os.path.exists(os.path.join(args.data_file_path,args.backbone_model)):
            os.mkdir(os.path.join(args.data_file_path,args.backbone_model))
        train_dataset_path = os.path.join(args.data_file_path,args.backbone_model,'pretrain','processed_dialogue.pkl')
        eval_dataset_path = os.path.join(args.data_file_path,args.backbone_model,'pretrain','processed_dialogue_valid.pkl')
        logger.info(['Get Embeddings For Movie Name'])
        tokenizer,new_embeddings = get_embeddings_for_movie_name(tokenizer,model,os.path.join(args.data_file_path,args.backbone_model),args.backbone_model,args.data_type)
        new_num_tokens = len(tokenizer)

        eval_dataset =  get_process_data(os.path.join(args.data_file_path,'original','pretrain','dialogue_shuffle_valid.pkl'),tokenizer,args.prefix_matters)
        train_dataset = get_process_data(os.path.join(args.data_file_path,'original','pretrain','dialogue_shuffle.pkl'),tokenizer,args.prefix_matters)

        with open(train_dataset_path,'wb') as f:
            pickle.dump(train_dataset,f)
        with open(eval_dataset_path,'wb') as f:
            pickle.dump(eval_dataset,f)
        with open(os.path.join(args.data_file_path,args.backbone_model,'movie_embedding.pt'),'wb') as f:
            torch.save(new_embeddings,f)
        

        if accelerator.is_main_process:
            logger.info('[Process dataset]')

    model = resize_token_embeddings(model,new_embeddings,new_num_tokens)
    # assert model.config.vocab_size == 37446

    # Data collator
    # This one will take care of randomly masking the tokens.
    # done 这里要更改，只mask key word部分(由PretrainDataset与dialogue.py共同实现)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=args.mlm_probability)

    # DataLoaders creation:
    train_dataloader = DataLoader(
        train_dataset, shuffle=False, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    # undo:这里可能有问题 =》 done：没问题，一切正常
    
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
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    # Figure out how many steps we should save the Accelerator states
    if hasattr(args.checkpointing_steps, "isdigit"):
        checkpointing_steps = args.checkpointing_steps
        if args.checkpointing_steps.isdigit():
            checkpointing_steps = int(args.checkpointing_steps)
    else:
        checkpointing_steps = None

    # We need to initialize the trackers we use, and also store our configuration
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = 'Linear'
        accelerator.init_trackers("mlm_no_trainer", experiment_config)

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
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_main_process)
    completed_steps = 0
    starting_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            accelerator.print(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
            accelerator.load_state(args.resume_from_checkpoint)
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
        else:
            resume_step = int(training_difference.replace("step_", ""))
            starting_epoch = resume_step // len(train_dataloader)
            resume_step -= starting_epoch * len(train_dataloader)

    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        if args.with_tracking:
            total_loss = 0
        for step, batch in enumerate(train_dataloader):
            # We need to skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == starting_epoch:
                if resume_step is not None and step < resume_step:
                    completed_steps += 1
                    continue
            attention_mask = (batch['input_ids'] != tokenizer.pad_token_id).long()

            
            outputs_bert = model(input_ids =batch['input_ids'],labels=batch['labels'],attention_mask = attention_mask)
            
            loss = outputs_bert.loss
            # We keep track of the loss at each epoch
            if args.with_tracking:
                total_loss += loss.detach().float()
            loss = loss / args.gradient_accumulation_steps
            accelerator.backward(loss)
            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1

            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0:
                    output_dir = f"step_{completed_steps }"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    accelerator.save_state(output_dir)

            if completed_steps >= args.max_train_steps:
                break

        model.eval()
        losses = []
        for step, batch in enumerate(eval_dataloader):
            attention_mask = (batch['input_ids'] != tokenizer.pad_token_id).long()
            with torch.no_grad():
                outputs_bert = model(input_ids =batch['input_ids'],labels=batch['labels'],attention_mask=attention_mask)

            loss = outputs_bert.loss
            losses.append(accelerator.gather(loss.repeat(args.per_device_eval_batch_size)))

        losses = torch.cat(losses)
        losses = losses[: len(eval_dataset)]
        try:
            perplexity = math.exp(torch.mean(losses))
        except OverflowError:
            perplexity = float("inf")
        if accelerator.is_main_process:
            logger.info(f"epoch {epoch}: perplexity: {perplexity}")

        if args.with_tracking:
            accelerator.log(
                {"perplexity": perplexity, "train_loss": total_loss, "epoch": epoch, "step": completed_steps},
            )



        if args.checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            accelerator.save_state(output_dir)
        accelerator.end_training()


    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        # done:用accelerator.save保存模型，先判断accelerator.is_main_process
        if accelerator.is_main_process:
            accelerator.save(unwrapped_model.state_dict(),os.path.join(args.output_dir,f'pretrian_{args.backbone_model}.pth'))
            logger.info(f'[Model saved]: pretrian_{args.backbone_model}.pth')
        '''
        unwrapped_model.save_pretrained(
            args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
        )
        '''
        # if accelerator.is_main_process:
        #     tokenizer.save_pretrained(args.output_dir)

        with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
            json.dump({"perplexity": perplexity}, f)
    return


if __name__ == "__main__":
    device_count = torch.cuda.device_count()
    print(f"Number of available GPUs: {device_count}")
    main()