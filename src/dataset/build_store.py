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
import torch


def get_process_data(original_data_path,tokenizer,prefix_matters,test_mode=False):
    print(['Get Process Data'])
    with open(original_data_path,'rb') as f:
        data = pickle.load(f)
    if test_mode:
        data = data[:100]
        print("Test Mode")
    # get key word utters
    '''
    key: @xxx
    value: [utter,response]
    attention:这里的utter是history+response
    '''
    new_data = {}
    for e in tqdm(data,total=len(data)):
        total_keys = 0
        context = ''
        for i_th,each in enumerate(e): 
            if context != '' and context[-1] != tokenizer.sep_token:
                context += tokenizer.sep_token
            if each['utter'] == '':
                continue
            
            if ( len(each['keyword']) > 0 and total_keys > 2 ) or (i_th == len(e) -1 and len(each['keyword']) > 1):
                for k in each['keyword']:
                    if k[0] == '@':
                        if k not in new_data:
                            new_data[k] = []
                        # only mask the last key word
                        index = each['utter'].rfind(k)
                        masked_context = context + each['utter'][:index] + tokenizer.mask_token + each['utter'][index+len(k):]
                        new_data[k].append((masked_context,each['utter'].replace(k,tokenizer.mask_token)))
            context += each['utter']
            total_keys += len(each['keyword'])
    
    for k in new_data:
        new_data[k] = list(set(new_data[k]))
        min_num = min(1e10,len(new_data[k])) 
    print(min_num,len(new_data)) # min:10
    return new_data

def split(key_word_utters,tokenizer,save_path,k_num):
    print(['Split into csv'])
    rows = []
    for k in tqdm(key_word_utters,total=len(key_word_utters)):
        for i,(utter,response) in enumerate(key_word_utters[k][:k_num]):
            rows.append([utter,response,k])
    df = pd.DataFrame(rows,columns=['history','response','label'],dtype=str)
    # df = df.sample(frac=1).reset_index(drop=True) # shuffle,对于store来说，不需要shuffle
    save_file = os.path.join(save_path,'store.csv')
    df.to_csv(save_file,index=False)
    return df

def encode(df,tokenizer,knn_num):
    #batch_encode_plus
    batch_size = knn_num * 3
    print(['Encode KNN'])
    history_utter = df['history'].tolist()
    response_utter = df['response'].tolist()
    label = df['label'].tolist()
    response_max_length = 32
    history_max_length = 128
    # deal with one by one
    number_ids = 0
    batch_encode = {'history':{'input_ids':[],'attention_mask':[],'last_eos_representations_mask':[]},
                    'response':{'input_ids':[],'attention_mask':[],'last_eos_representations_mask':[]},
                    'label':[]}
    for i in tqdm(range(len(history_utter)),total=len(history_utter)):
        # tokenize
        history = tokenizer.encode(str(history_utter[i]),add_special_tokens=False)
        response = tokenizer.encode(str(response_utter[i]),add_special_tokens=False)
        # print(i)
        # print(history_utter[i])
        # print(response_utter[i])

        # add special tokens
        history = [tokenizer.cls_token_id] + history[-history_max_length+2:] + [tokenizer.eos_token_id if tokenizer.eos_token_id else tokenizer.sep_token_id]
        response = [tokenizer.cls_token_id] + response[-response_max_length+2:] + [tokenizer.eos_token_id if tokenizer.eos_token_id else tokenizer.sep_token_id]
        # print(history)
        # print(response)

        len_history = len(history)
        len_response = len(response)

        # padding
        history = history + [tokenizer.pad_token_id] * (history_max_length - len_history)
        response = response + [tokenizer.pad_token_id] * (response_max_length - len_response)
        history_attention_mask = [1] * len_history + [0] * (history_max_length - len_history)
        response_attention_mask = [1] * len_response + [0] * (response_max_length - len_response)
        history_last_eos_representation_mask = [0]*(len_history-1) + [1] + [0] * (history_max_length - len_history)
        response_last_eos_representations_mask = [0]*(len_response-1) + [1] + [0] * (response_max_length - len_response)
        
        # print(history)
        # print(response)
        # print(history_attention_mask)
        # print(response_attention_mask)
        # print(history_last_eos_representation_mask)
        # print(response_last_eos_representations_mask)
        # print('\n\n\n')
        # if i == 1:
        #     raise

        # add to batch_encode
        batch_encode['history']['input_ids'].append(history)
        batch_encode['history']['attention_mask'].append(history_attention_mask)
        batch_encode['history']['last_eos_representations_mask'].append(history_last_eos_representation_mask)
        batch_encode['response']['input_ids'].append(response)
        batch_encode['response']['attention_mask'].append(response_attention_mask)
        batch_encode['response']['last_eos_representations_mask'].append(response_last_eos_representations_mask)
        label_id = tokenizer.convert_tokens_to_ids(label[i])
        assert label_id != tokenizer.unk_token_id
        batch_encode['label'].append(label_id)
        number_ids += 1
        if number_ids == batch_size or i == len(history) - 1:
            # convert to tensor            
            for key in batch_encode:
                if key == 'label':
                    batch_encode[key] = torch.tensor(batch_encode[key],dtype=torch.long)
                else:
                    for sub_key in batch_encode[key]:
                        batch_encode[key][sub_key] = torch.tensor(batch_encode[key][sub_key],dtype=torch.long)
            yield batch_encode
            batch_encode = {'history':{'input_ids':[],'attention_mask':[],'last_eos_representations_mask':[]},
                            'response':{'input_ids':[],'attention_mask':[],'last_eos_representations_mask':[]},
                            'label':[]}
            number_ids = 0
        
    

def store(df,tokenizer,save_path,k_num,model,representation_position):
    print(['Store KNN'])
    all_history_representations,all_response_representations,all_recommendation_representations,all_label = None,None,None,None
    for i,batch_encode in enumerate(encode(df,tokenizer,k_num)):
        with torch.no_grad():
            batch_encode['history']['input_ids'] = batch_encode['history']['input_ids'].cuda()
            batch_encode['history']['attention_mask'] = batch_encode['history']['attention_mask'].cuda()
            batch_encode['response']['input_ids'] = batch_encode['response']['input_ids'].cuda()
            batch_encode['response']['attention_mask'] = batch_encode['response']['attention_mask'].cuda()
            #convert list to numpy
            batch_encode['label'] = np.array(batch_encode['label'])
            
            history_outputs = model(batch_encode['history']['input_ids'],batch_encode['history']['attention_mask'],return_dict=True,output_hidden_states=True )
            if 'hidden_states'  in history_outputs:
                history_last_layer_representations = history_outputs.hidden_states[-1]
                history_first_layer_representations = history_outputs.hidden_states[1]
            else: # Seq2SeqLMOutput, e.g., BartModel
                history_last_layer_representations = history_outputs.decoder_hidden_states[-1]
                history_first_layer_representations = history_outputs.encoder_hidden_states[1]

            response_outputs = model(batch_encode['response']['input_ids'],batch_encode['response']['attention_mask'],return_dict=True,output_hidden_states=True )
            if 'hidden_states'  in response_outputs:
                response_last_layer_representations = response_history_outputs.hidden_states[-1]
                response_first_layer_representations = response_history_outputs.hidden_states[1]
            else: # Seq2SeqLMOutput, e.g., BartModel
                response_last_layer_representations = response_outputs.decoder_hidden_states[-1]
                response_first_layer_representations = response_outputs.encoder_hidden_states[1]
                # print(response_last_layer_representations.shape)
                # print(response_first_layer_representations.shape)

            if representation_position == 'last':
                history_representations = history_last_layer_representations[torch.where(batch_encode['history']['last_eos_representations_mask'] == 1)] # [batch_size,hidden_size]
                response_representations = response_last_layer_representations[torch.where(batch_encode['response']['last_eos_representations_mask'] == 1)] # [batch_size,hidden_size]
                recommendation_represenations = history_representations
            elif representation_position == 'cls':
                history_representations = history_last_layer_representations[:,0,:]
                response_representations = response_last_layer_representations[:,0,:]
                recommendation_represenations = history_last_layer_representations[torch.where(batch_encode['history']['input_ids'] == tokenizer.mask_token_id)]
            elif representation_position == 'avg_first_last':
                # get the average of first and last layer with attention mask
                history_representations = ((history_first_layer_representations + history_last_layer_representations) / 2.0 * batch_encode['history']['attention_mask'].unsqueeze(-1)).sum(1) / batch_encode['history']['attention_mask'].sum(-1).unsqueeze(-1)
                response_representations = ((response_first_layer_representations + response_last_layer_representations) / 2.0 * batch_encode['response']['attention_mask'].unsqueeze(-1)).sum(1) / batch_encode['response']['attention_mask'].sum(-1).unsqueeze(-1)
                recommendation_represenations = history_representations
            else:
                raise
            
            # store
            if i == 0:
                all_history_representations = history_representations.cpu().detach().numpy()
                all_response_representations = response_representations.cpu().detach().numpy()
                all_recommendation_representations = recommendation_represenations.cpu().detach().numpy()
                all_label = batch_encode['label']
            else:
                all_history_representations = np.concatenate((all_history_representations,history_representations.cpu().detach().numpy()),axis=0)
                all_response_representations = np.concatenate((all_response_representations,response_representations.cpu().detach().numpy()),axis=0)
                all_recommendation_representations = np.concatenate((all_recommendation_representations,recommendation_represenations.cpu().detach().numpy()),axis=0)
                all_label = np.concatenate((all_label,batch_encode['label']),axis=0)
    # save as npy
    save_path = os.path.join(save_path,representation_position)
    os.makedirs(save_path,exist_ok=True)
    print(all_history_representations.shape)
    np.save(os.path.join(save_path,'history_representations.npy'),all_history_representations)
    np.save(os.path.join(save_path,'response_representations.npy'),all_response_representations)
    np.save(os.path.join(save_path,'recommendation_representations.npy'),all_recommendation_representations)
    np.save(os.path.join(save_path,'label.npy'),all_label)
            



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_type',type=str,default=None)
    parser.add_argument('--save_dict',type=str,default='knn')
    parser.add_argument('--k_num',type=int,default=10)
    parser.add_argument('--test_mode',action='store_true')
    parser.add_argument('--backbone_model',type=str,default=None)
    parser.add_argument('--crs_model_path',type=str,default=None)
    parser.add_argument('--representation_position',type=str,default='cls')
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(BACKBONE_MODEL_MAPPINGS[args.backbone_model]) 
    tokenizer = add_tokens_for_tokenizer(tokenizer,args.data_type) # 一定要resize

    new_embeddings = torch.load(os.path.join('data',args.data_type,args.backbone_model,'movie_embedding.pt'))
    new_num_tokens = len(tokenizer)
    model = MODEL_TYPES_MAPPING[args.backbone_model].from_pretrained(BACKBONE_MODEL_MAPPINGS[args.backbone_model])
    model = resize_token_embeddings(model,new_embeddings,new_num_tokens) # 一定要resize
    model.load_state_dict(torch.load(args.crs_model_path),strict=False)



    model.cuda()
    model.eval()

    if args.backbone_model in ['roberta','bart']:
        prefix_matters = True
    else:
        prefix_matters = False
    original_data_path = os.path.join('data',args.data_type,'original','pretrain','dialogue_shuffle.pkl')
    save_path = os.path.join('data',args.data_type,args.backbone_model,args.save_dict)
    os.makedirs(save_path,exist_ok=True)
    new_data = get_process_data(original_data_path,tokenizer,prefix_matters,test_mode=args.test_mode)
    new_data = split(new_data,tokenizer,save_path,args.k_num)
    store(new_data,tokenizer,save_path,args.k_num,model,representation_position=args.representation_position)
    