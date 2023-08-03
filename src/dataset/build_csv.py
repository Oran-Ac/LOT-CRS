# build csv for contrastive learning
# column[0]: sentence_1 column[1]: sentence_2 (they are with the same label)
import pandas as pd
import pickle
from tqdm import tqdm
import argparse
from transformers import AutoTokenizer
import os
import sys
sys.path.append("src")
from utils.args import BACKBONE_MODEL_MAPPINGS,MODEL_TYPES_MAPPING
from utils.utils import *


def get_process_data(original_data_path,tokenizer,prefix_matters,test_mode=False):
    print(['Get Process Data'])
    with open(original_data_path,'rb') as f:
        data = pickle.load(f)
    if test_mode:
        data = data[:10]
        print("Test Mode")
    new_data = {}
    for e in tqdm(data,total=len(data)):
        total_keys = 0
        context = ''
        for i_th,each in enumerate(e): 
            if context != '':
                context += tokenizer.sep_token
            context += each['utter']
            if ( len(each['keyword']) > 0 and total_keys > 2 ) or (i_th == len(e) -1 and len(each['keyword']) > 1):
                for k in each['keyword']:
                    if k[0] == '@':
                        if k not in new_data:
                            new_data[k] = []
                        new_data[k].append(context)
            total_keys += len(each['keyword'])
    
    for k in new_data:
        new_data[k] = list(set(new_data[k]))
        min_num = min(1e10,len(new_data[k])) 
    print(min_num,len(new_data)) # min:10
    return new_data

def split_into_csv(key_word_utters,tokenizer,save_path,k_num):
    print(['Split into csv'])
    rows = []
    for k in tqdm(key_word_utters,total=len(key_word_utters)):
        for i,utter in enumerate(key_word_utters[k][:k_num]):
            utter = utter.replace(k,tokenizer.mask_token) # replace the key word with mask token to advoid information leakage
            for utter_2 in key_word_utters[k][i+1:k_num+1]:
                utter_2 = utter_2.replace(k,tokenizer.mask_token)
                if utter != utter_2:
                    rows.append([utter,utter_2])
    df = pd.DataFrame(rows,columns=['sentence_1','sentence_2'])
    # shuffle
    df = df.sample(frac=1).reset_index(drop=True)
    df.to_csv(save_path,index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_type',type=str,default=None)
    parser.add_argument('--save_file',type=str,default='contrastive_learning.csv')
    parser.add_argument('--k_num',type=int,default=10)
    parser.add_argument('--test_mode',action='store_true')
    parser.add_argument('--backbone_model',type=str,default=None)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(BACKBONE_MODEL_MAPPINGS[args.backbone_model]) 
    tokenizer = add_tokens_for_tokenizer(tokenizer,args.data_type) # 这里不用也行，没有分词
    if args.backbone_model in ['roberta','bart']:
        prefix_matters = True
    else:
        prefix_matters = False
    original_data_path = os.path.join('data',args.data_type,'original','pretrain','dialogue_shuffle.pkl')
    save_path = os.path.join('data',args.data_type,args.backbone_model,'pretrain',args.save_file)
    new_data =  get_process_data(original_data_path,tokenizer,prefix_matters,test_mode=args.test_mode)
    split_into_csv(new_data,tokenizer,save_path,args.k_num)
    