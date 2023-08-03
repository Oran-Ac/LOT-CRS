
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

class PretrainDataset(Dataset):
    def __init__(self,dataset) -> None:
        self.dataset = dataset
        
    def __getitem__(self, index):
        # 要返回 word, token,special_tokens_mask  =》已编码版本
        context_token = self.dataset[index]['context_tokens']
        # context_words = self.dataset[index]['context_words']
        # if len(context_words) != 128: # 由于此时数据处理的一些bug，先这样写，最终代码应该删除
        #     context_words = context_words[-128:]
        #     context_words = context_words +[0]*(128-len(context_words))
        context_special_tokens_mask = self.dataset[index]['context_special_tokens_mask']
        
        # return context_token,context_special_tokens_mask
        context_token_vector = torch.LongTensor(context_token)
        # context_words_vector = torch.LongTensor(context_words)
        context_special_tokens_mask_vector = torch.LongTensor(context_special_tokens_mask)
        # return context_token_vector,context_words_vector,context_special_tokens_mask_vector
        return context_token_vector,context_special_tokens_mask_vector
    def __len__(self):
        return len(self.dataset)


@dataclass
class DataCollatorForLanguageModeling:

    tokenizer: PreTrainedTokenizerBase
    mlm: bool = True
    mlm_probability: float = 0.15
    special_tokens_mask:Optional[torch.Tensor] = None

    def __post_init__(self):
        if self.mlm and self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. "
                "You should pass `mlm=False` to train on causal language modeling instead."
            )

    def __call__(
        self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        
        batch = self._collate_batch(examples, self.tokenizer)

        # If special token mask has been preprocessed, pop it from the dict.
        if self.mlm:
            batch["input_ids"], batch["labels"] = self.mask_tokens(
                batch["input_ids"], special_tokens_mask=batch["special_mask_tokens"]
            )
        else:
            labels = batch["input_ids"].clone()
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            batch["labels"] = labels
        return batch

    def mask_tokens(
        self, inputs: torch.Tensor, special_tokens_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, 0,dtype=torch.float64)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=self.mlm_probability)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        # 更难的任务应该是属性之间交换
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels

    def _collate_batch(self,examples, tokenizer):
        """Collate `examples` into a batch, using the information in `tokenizer` for padding if necessary."""
        # Tensorize if necessary.
        
        context_token_vector,context_special_tokens_mask_vector =[],[]
        # context_words_vector,context_entities_vector=[],[]
        for e in examples:
            # c_t_v,c_w_v,c_m_v = e
            c_t_v,c_m_v = e
            # context_words_vector.append(c_w_v)
            context_token_vector.append(c_t_v)
            # context_entities_vector.append(torch.LongTensor([0]))
            context_special_tokens_mask_vector.append(c_m_v)

        # context_words_vector_stack = torch.stack(context_words_vector, dim=0)
        context_token_vector_stack = torch.stack(context_token_vector, dim=0)
        context_special_tokens_mask_vector_stack = torch.stack(context_special_tokens_mask_vector, dim=0)
        # context_entities_vector_stack = torch.stack(context_entities_vector,dim=0)
        return_value ={
                'input_ids':context_token_vector_stack,
                # 'word_batch':context_words_vector_stack,
                'special_mask_tokens':context_special_tokens_mask_vector_stack,
                # 'entity_batch':context_entities_vector_stack
        }
        return return_value



# undo: 这里要同一句话才对？
class RetrievalDataset(Dataset):
    def __init__(self,dataset,shuffle=False) -> None:

        self.dataset = dataset
        if shuffle:
            random.shuffle(self.dataset)
        
    def __getitem__(self, index):
        context_token = self.dataset[index]['context_tokens']
        context_label = self.dataset[index]['context_label']
        
        # return context_words,context_token,context_special_tokens_mask
        context_token_vector = torch.LongTensor(context_token)
        context_label_vector = torch.LongTensor([context_label])
        return context_token_vector,context_label_vector
    def __len__(self):
        return len(self.dataset)

@dataclass
class DataCollatorForTrainRetrieval:
    def __call__(
        self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        batch = self._collate_batch(examples, self.tokenizer)

    def _collate_batch(self,examples, tokenizer):
        """Collate `examples` into a batch, using the information in `tokenizer` for padding if necessary."""
        # Tensorize if necessary.
        
        context_token_vector,context_label_vector =[],[]
        for e in examples:
            c_t_v,c_m_v = e
            context_token_vector.append(c_t_v)
            context_label_vector.append(c_m_v)

        context_token_vector_stack = torch.stack(context_token_vector, dim=0)
        context_label_vector_stack = torch.stack(context_label_vector, dim=0)
        context_mlm_position_vector_stack = (context_token_vector_stack == tokenizer.mask_token_id).long()
        context_mlm_attn_vector_stack = (context_token_vector_stack != tokenizer.pad_token_id ).long()
        if tokenizer.eos_token_id is not None:
            context_last_token_position_stack  = (context_token_vector_stack == tokenizer.eos_token_id ).long()
        return_value ={
                'input_ids':context_token_vector_stack,
                'attn':context_mlm_attn_vector_stack,
                'mlm_position':context_mlm_position_vector_stack,
                'labels':context_label_vector_stack,
                'last_token_position':context_last_token_position_stack if tokenizer.eos_token_id is not None else None
        }
        return return_value


class CRSDatasetRecommendation(Dataset):
    def __init__(self,
                dataset,
                backbone_model,
                data_type,
                split,
                word_pad_index,
                entity_pad_index,
                debug = False,
                context_tokenizer = None,
                dbpedia_tokenzier = None,
                word_tokenizer = None,
                token_max_length = 256,
                entity_max_length = 100,
                word_max_length = 100,
                prompt_text = None,
                padding = 'max_length',
                save_data = True,
                reload_data =True,
                ):
        super(CRSDatasetRecommendation, self).__init__()
        self.debug = debug
        self.prompt_text= prompt_text.replace(" [MASK]",context_tokenizer.mask_token) #如果不把前面的空格一起去掉的话会把" <mask>"翻成unk
        self.padding = padding
        self.word_pad_index = word_pad_index
        self.entity_pad_index = entity_pad_index

        self.context_tokenizer = context_tokenizer
        self.dbpedia_tokenzier = dbpedia_tokenzier
        self.word_tokenizer = word_tokenizer

        self.max_length = token_max_length
        self.entity_max_length = entity_max_length
        self.word_max_length = word_max_length

        dataset_dir = os.path.join(dataset,data_type)
        data_file = os.path.join(dataset_dir, 'original',f'{split}_data_processed.jsonl')
        self.data = []
        if reload_data:
            reload_data_file = os.path.join(dataset_dir,backbone_model,'recommend',f'{split}_data_processed.pkl')
            if os.path.exists(os.path.dirname(reload_data_file)):
                with open(reload_data_file,'rb') as f:
                    self.data = pickle.load(f)
            else:
                self.prepare_data(data_file)
        else:
            self.prepare_data(data_file)

        if save_data:
            save_data_file = os.path.join(dataset_dir,backbone_model,'recommend',f'{split}_data_processed.pkl')
            if not os.path.exists(os.path.dirname(save_data_file)):
                os.makedirs(os.path.dirname(save_data_file))
            with open(save_data_file,'wb') as f:
                pickle.dump(self.data,f)

    
    def prepare_data(self, data_file):
        with open(data_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        if self.debug:
            lines = lines[:1024]
            self.padding = False

        for line in tqdm(lines):
            dialog = json.loads(line)
            if len(dialog['rec_movie_token']) == 0:
                continue
            if len(dialog['context']) == 1 and dialog['context'][0] == '':
                continue

            context = ''
            word = []

            for i, utt in enumerate(dialog['context']):
                if utt == '':
                    continue
                context += utt
                context += self.context_tokenizer.sep_token if self.context_tokenizer.eos_token is not None and self.context_tokenizer.eos_token  != self.context_tokenizer.sep_token else ""
            context += self.prompt_text
            context += self.context_tokenizer.eos_token if self.context_tokenizer.eos_token is not None else ""
            context_ids = self.context_tokenizer.convert_tokens_to_ids(self.context_tokenizer.tokenize(context))
            context_ids = context_ids[-self.max_length+1:]
            context_ids.insert(0,self.context_tokenizer.cls_token_id)
            context_ids = context_ids + [self.context_tokenizer.pad_token_id] *(self.max_length - len(context_ids)) #pad


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



            for j,item in enumerate(dialog['rec_movie_token']):
                item_token_ids = self.context_tokenizer.convert_tokens_to_ids(item)

                data = {
                    'context_ids': context_ids,
                    'rec':item_token_ids,
                    'rec_idx':dialog['rec_movie_index'][j],
                    'entity_ids':entity_ids,
                    'word_ids':word_ids
                }
                self.data.append(data)
            
            


    def __getitem__(self, ind):
        return self.data[ind]

    def __len__(self):
        return len(self.data)


@dataclass
class CRSDataRecommendationCollator:
    def __init__(self,
                vocab
        ) -> None:
        self.entity_pad_index = vocab['pad_entity_idx']
        self.word_pad_index = vocab['pad_word_idx']
        self.context_pad_index = vocab['pad_context_idx'] #由tokenzier初始化
        self.context_msk_index = vocab['msk_context_idx'] #由tokenzier初始化
        self.context_eos_index = vocab['eos_context_idx'] #由tokenzier初始化


    def __call__(self, data_batch) -> Any:
        context_batch = []
        entity_batch = []
        word_batch = []
        label_batch = []
        rec_dbpedia_label_batch = []
        '''
        1. get the attn mask
        2. get the mlm position
        3. place them on device or use accelerate we don't need it 
        data = {
                    'context_ids': context_ids,
                    'rec':item_token_ids,
                    'entity_ids':entity_ids,
                    'word_ids':word_ids
                }
        '''
        for e in data_batch:
            context_batch.append(e['context_ids'])
            entity_batch.append(e['entity_ids'])
            word_batch.append(e['word_ids'])
            label_batch.append(e['rec'])
            rec_dbpedia_label_batch.append(e['rec_idx'])
      
        context_batch = torch.LongTensor(context_batch)
        entity_batch = torch.LongTensor(entity_batch)
        word_batch = torch.LongTensor(word_batch)
        label_batch = torch.LongTensor(label_batch)
        rec_dbpedia_label_batch = torch.LongTensor(rec_dbpedia_label_batch)
       
        return {
            'context_batch':context_batch,
            'entity_batch':entity_batch,
            'word_batch':word_batch,
            'label_batch':label_batch,
            'rec_dbpedia_movie_label_batch':rec_dbpedia_label_batch,
            'context_batch_attn':(context_batch != self.context_pad_index).long(),
            # 'word_batch_attn':(word_batch != self.word_pad_index).long(),
            # 'entity_batch_attn': (entity_batch != self.entity_pad_index).long(),
            'context_batch_mlm_position':(context_batch == self.context_msk_index).long(),
            'context_batch_last_position':(context_batch == self.context_eos_index).long() if self.context_eos_index is not None else None
        }


class KGSFDataset(Dataset):
    def __init__(self,
                dataset,
                data_type,
                split,
                word_pad_index,
                entity_pad_index,
                debug = False,
                entity_max_length = 100,
                word_max_length = 100,
                padding = 'max_length',
                save_data = True,
                reload_data =True,
                n_entity = None,
                movie_ids = None
                ):
        super(KGSFDataset, self).__init__()
        self.debug = debug
        self.word_pad_index = word_pad_index
        self.entity_pad_index = entity_pad_index

        self.entity_max_length = entity_max_length
        self.word_max_length = word_max_length

        self.n_entity = n_entity
        self.movie_ids = movie_ids

        dataset_dir = os.path.join(dataset,data_type)
        data_file = os.path.join(dataset_dir, 'original',f'{split}_data_processed.jsonl')
        self.data = []
        if reload_data:
            reload_data_file = os.path.join(dataset_dir,'kgsf',f'{split}_data_processed.pkl')
            with open(reload_data_file,'rb') as f:
                self.data = pickle.load(f)
        else:
            self.prepare_data(data_file)

        if save_data:
            os.makedirs(os.path.join(dataset_dir,'kgsf'),exist_ok=True)
            save_data_file = os.path.join(dataset_dir,'kgsf',f'{split}_data_processed.pkl')
            with open(save_data_file,'wb') as f:
                pickle.dump(self.data,f)

    
    def prepare_data(self, data_file):
        with open(data_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        if self.debug:
            lines = lines[:512]

        for line in tqdm(lines):
            dialog = json.loads(line)
            if len(dialog['rec_movie_token']) == 0:
                continue
            if len(dialog['context']) == 1 and dialog['context'][0] == '':
                continue

            word = dialog['word'][-self.word_max_length:]
            word_ids = word
            # word_ids = [self.word_tokenizer.index(w) for w in word]
            if len(word_ids) < self.word_max_length:
                word_ids = word_ids + [self.word_pad_index] *(self.word_max_length - len(word_ids))

            entity = dialog['entity'][-self.entity_max_length:]
            entity_ids = entity
            # entity_ids = [self.entity_tokenizer.index(e) for e in entity]

            # get onehot entity labels before padding
            entity_labels = self.get_onehot(entity_ids,self.n_entity)

            if len(entity_ids) < self.entity_max_length:
                entity_ids = entity_ids + [self.entity_pad_index] *(self.entity_max_length - len(entity_ids))



            for j,item in enumerate(dialog['rec_movie_index']):
                item_ids = self.movie_ids[item] # convert #n_movie to #n_entity
                movie_ids = item

                data = {
                    'word_ids':word_ids,
                    'entity_ids':entity_ids,
                    'item_ids':item_ids,
                    'entity_labels':entity_labels,
                    'movie_ids':movie_ids
                }
                self.data.append(data)
            
    def get_onehot(self,data_list, categories):

        onehot_label =  [0] * categories
        for label in data_list:
            onehot_label[label] = 1.0 / len(data_list)
        return onehot_label

    def __getitem__(self, ind):
        return self.data[ind]

    def __len__(self):
        return len(self.data)


@dataclass
class KGSFDatasetCollator:
    def __init__(self,
        ) -> None:
        pass
    def __call__(self, data_batch) -> Any:
        word_batch = []
        entity_batch = []
        item_batch = []
        entity_labels_batch = []
        movie_batch = []
        for e in data_batch:
            word_batch.append(e['word_ids'])
            entity_batch.append(e['entity_ids'])
            item_batch.append(e['item_ids'])
            entity_labels_batch.append(e['entity_labels'])
            movie_batch.append(e['movie_ids'])

        word_batch = torch.LongTensor(word_batch)
        entity_batch = torch.LongTensor(entity_batch)
        item_batch = torch.LongTensor(item_batch)
        entity_labels_batch = torch.FloatTensor(entity_labels_batch)
        movie_batch = torch.LongTensor(movie_batch)
        return {
            'word_batch':word_batch,
            'entity_batch':entity_batch,
            'item_batch':item_batch,
            'entity_labels_batch':entity_labels_batch,
            'movie_batch':movie_batch
        }
