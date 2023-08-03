
from torch.utils.data.dataset import Dataset
import torch
from transformers.tokenization_utils_base import PreTrainedTokenizerBase,BatchEncoding
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
import pickle
from torch.utils.data import DataLoader
from dataclasses import dataclass
import random
import os
from tqdm import tqdm
import json
from collections import defaultdict
class CRSDatasetConversation(Dataset):
    def __init__(
        self, 
        dataset, 
        backbone_model,
        data_type,
        split, 
        word_pad_index,
        entity_pad_index,
        context_tokenizer, # for backbone
        gen_tokenizer,  # for gpt
        debug=False,
        context_max_length=None,  # for gpt
        resp_max_length=None,  # for gpt
        entity_max_length=None, # for kg
        word_max_length=None, # for kg
        prompt_max_length=None, # for backbone,
        reload_data=False,
        save_data=False
    ):
        super(CRSDatasetConversation, self).__init__()
        self.context_tokenizer = context_tokenizer
        self.gen_tokenizer = gen_tokenizer
        self.debug = debug

        self.word_pad_index = word_pad_index
        self.entity_pad_index = entity_pad_index


        self.context_max_length = context_max_length
        if self.context_max_length is None:
            self.context_max_length = self.gen_tokenizer.model_max_length
        self.context_max_length -= 1

        self.resp_max_length = resp_max_length
        if self.resp_max_length is None:
            self.resp_max_length = self.gen_tokenizer.model_max_length

        self.entity_max_length = entity_max_length
        self.word_max_length = word_max_length
        

        self.prompt_max_length = prompt_max_length
        if self.prompt_max_length is None:
            self.prompt_max_length = self.context_tokenizer.model_max_length

        dataset_dir = os.path.join(dataset,data_type)
        data_file = os.path.join(dataset_dir, 'original',f'{split}_data_processed.jsonl')
        self.data = []
        if reload_data:
            reload_data_file = os.path.join(dataset_dir,backbone_model,'conv',f'{split}_data_processed.pkl')
            if os.path.exists(os.path.dirname(reload_data_file)):
                with open(reload_data_file,'rb') as f:
                    self.data = pickle.load(f)
            else:
                self.prepare_data(data_file)
        else:
            self.prepare_data(data_file)

        if save_data:
            save_data_file = os.path.join(dataset_dir,backbone_model,'conv',f'{split}_data_processed.pkl')
            if not os.path.exists(os.path.dirname(save_data_file)):
                os.makedirs(os.path.dirname(save_data_file))
            with open(save_data_file,'wb') as f:
                pickle.dump(self.data,f)

    def prepare_data(self, data_file):
        with open(data_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        if self.debug:
            lines = lines[:1024]

        for line in tqdm(lines):
            dialog = json.loads(line)

            context = ''
            prompt_context = ''
            

            for i, utt in enumerate(dialog['context']):
                if utt == '':
                    continue
                if i % 2 == 0:
                        context += 'User: '
                        prompt_context += 'User: '
                else:
                    context += 'System: '
                    prompt_context += 'System: '
                context += utt
                prompt_context += utt
                context += self.gen_tokenizer.sep_token if self.gen_tokenizer.sep_token is not None and self.gen_tokenizer.eos_token  != self.gen_tokenizer.sep_token else " "
                prompt_context += self.context_tokenizer.sep_token if self.context_tokenizer.sep_token is not None and self.context_tokenizer.eos_token  != self.context_tokenizer.sep_token else " "
            if context == '':
                continue

            prompt_context += self.context_tokenizer.eos_token if self.context_tokenizer.eos_token is not None else ""
            prompt_context_ids = self.context_tokenizer.convert_tokens_to_ids(self.context_tokenizer.tokenize(prompt_context))
            prompt_context_ids = prompt_context_ids[-self.prompt_max_length+1:]
            if self.context_tokenizer.cls_token_id is not None:
                prompt_context_ids.insert(0,self.context_tokenizer.cls_token_id)
            prompt_context_ids = prompt_context_ids + [self.context_tokenizer.pad_token_id] *(self.prompt_max_length - len(prompt_context_ids)) #pad

            context += self.gen_tokenizer.eos_token if self.gen_tokenizer.eos_token is not None else ""
            context_ids = self.gen_tokenizer.convert_tokens_to_ids(self.gen_tokenizer.tokenize(context))
            context_ids = context_ids[-self.context_max_length+1:]
            if self.gen_tokenizer.cls_token_id is not None:
                context_ids.insert(0,self.gen_tokenizer.cls_token_id)
            # context_ids = context_ids + [self.gen_tokenizer.pad_token_id] *(self.context_max_length - len(context_ids)) #pad
            

            word = dialog['word'][-self.word_max_length:]
            word_ids = word
            # word_ids = [self.word_tokenizer.index(w) for w in word]
            if len(word_ids) < self.word_max_length:
                word_ids = word_ids + [self.word_pad_index] *(self.word_max_length - len(word_ids))

            entity = dialog['entity'][-self.entity_max_length:]
            entity_ids = entity
            # entity_ids = [self.entity_tokenizer.index(e) for e in entity]

            if len(entity_ids) < self.entity_max_length:
                entity_ids = entity_ids + [self.entity_pad_index] *(self.entity_max_length - len(entity_ids))

            resp = dialog['resp']
            resp = 'System: ' + resp
            response_ids = self.gen_tokenizer.convert_tokens_to_ids(self.gen_tokenizer.tokenize(resp))
            response_ids = response_ids[:self.resp_max_length]
            response_ids.append(self.gen_tokenizer.eos_token_id)


            
            data = {
                'context_ids': context_ids,
                'prompt_context_ids': prompt_context_ids,
                'response_ids': response_ids,
                'entity_ids':entity_ids,
                'word_ids':word_ids
            }
            self.data.append(data)
            
    def __getitem__(self, ind):
        return self.data[ind]

    def __len__(self):
        return len(self.data)

class CRSDataConversationCollator:
    def __init__(
        self, 
        gen_tokenizer, 
        gen=False, 
        use_amp=False, 
        debug=False, 
        ignore_pad_token_for_loss=True,
        context_max_length=None, 
        vocab=None,
        device=None
    ):
        self.gen_tokenizer = gen_tokenizer
        self.use_amp = use_amp
        self.ignore_pad_token_for_loss = ignore_pad_token_for_loss
        self.gen = gen
        self.debug = debug
        self.device = device

        self.padding = 'max_length' if self.debug else True
        self.pad_to_multiple_of = 8 if use_amp else None

        self.context_max_length = context_max_length
        if self.context_max_length is None:
            self.context_max_length = self.gen_tokenizer.model_max_length
  
        self.generate_prompt_ids = self.gen_tokenizer.convert_tokens_to_ids(self.gen_tokenizer.tokenize('System:'))
        self.prompt_context_msk_index = vocab['msk_context_idx'] #text_tokenizer
        self.prompt_context_eos_index = vocab['eos_context_idx'] #text_tokenizer
        self.prompt_context_pad_index = vocab['pad_context_idx'] #text_tokenizer

    def __call__(self, data_batch):
        context_batch = defaultdict(list)
        prompt_batch = defaultdict(list)
        entity_batch = []
        word_batch = []
        resp_batch = []
        context_len_batch = []
        if self.gen:
            self.gen_tokenizer.padding_side = 'left'
            for data in data_batch:
                context_ids = data['context_ids']
                context_ids = context_ids[-(self.context_max_length - len(self.generate_prompt_ids)):]
                context_len_batch.append(len(context_ids))
                context_ids += self.generate_prompt_ids
                context_batch['input_ids'].append(context_ids)

                prompt_batch['context_batch'].append(data['prompt_context_ids'])
                resp_batch.append(data['response_ids'])
                entity_batch.append(data['entity_ids'])
                word_batch.append(data['word_ids'])
        else:
            self.gen_tokenizer.padding_side = 'right'
            for data in data_batch:
                input_ids = data['context_ids'] + data['response_ids']
                input_ids = input_ids[-self.context_max_length:]
                context_batch['input_ids'].append(input_ids)

                prompt_batch['context_batch'].append(data['prompt_context_ids'])
                entity_batch.append(data['entity_ids'])
                word_batch.append(data['word_ids'])
        
        # print([len(i) for i in prompt_batch['context_batch']])
        # print( prompt_batch['context_batch'][0])
        # print( prompt_batch['context_batch'][1])
        prompt_batch['context_batch'] = torch.LongTensor(prompt_batch['context_batch']).to(self.device)
        # print(context_batch)
        context_batch = self.gen_tokenizer.pad(
            context_batch, padding=self.padding, pad_to_multiple_of=self.pad_to_multiple_of,
            max_length=self.context_max_length
        )
        # print(prompt_batch)
        prompt_batch['context_batch_attn'] = (prompt_batch['context_batch'] != self.prompt_context_pad_index).long().to(self.device)
        prompt_batch['context_batch_mlm_position'] = (prompt_batch['context_batch'] == self.prompt_context_msk_index).long().to(self.device)
        prompt_batch['context_batch_last_position'] = (prompt_batch['context_batch'] == self.prompt_context_eos_index).long().to(self.device) if self.prompt_context_eos_index is not None else None
        if not self.gen:
            resp_batch = context_batch['input_ids']
            resp_batch = [[token_id if token_id != self.gen_tokenizer.pad_token_id else -100 for token_id in resp] for resp
                          in resp_batch]
            resp_batch = torch.as_tensor(resp_batch).to(self.device)
        else:
            resp_batch = resp_batch
        
        
        for k, v in context_batch.items():
            if not isinstance(v, torch.Tensor):
                context_batch[k] = torch.as_tensor(v).to(self.device)
        return{
            "context_batch": context_batch,
            "prompt_batch": prompt_batch,
            "entity_batch": torch.as_tensor(entity_batch).to(self.device),
            "word_batch": torch.as_tensor(word_batch).to(self.device),
            "resp_batch": resp_batch,
            "context_len_batch": torch.as_tensor(context_len_batch).to(self.device)
        }