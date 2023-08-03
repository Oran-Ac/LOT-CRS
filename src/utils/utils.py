import logging
from tqdm import tqdm
import torch
import pandas as pd
from . import args
import os
import numpy as np
'''
----------------
Model embedding related
----------------
'''
def resize_token_embeddings(model,new_embeddings,new_num_tokens):
    base_model = getattr(model, model.base_model_prefix, model)  # get the base model if needed
    model_embeds = model._resize_token_embeddings(new_num_tokens)
    original_vocab_size = model.config.vocab_size
    assert new_embeddings.shape[0] == new_num_tokens-original_vocab_size #确保数量正确
    old_embeddings = model.get_input_embeddings() 
    new_embeddings.to(old_embeddings.weight.device)
    old_embeddings.weight.data[original_vocab_size:,:] = new_embeddings
    model.set_input_embeddings(old_embeddings)

    model_embeds = model.get_input_embeddings()
    if hasattr(model,'final_logits_bias'):
        model._resize_final_logits_bias(new_num_tokens)

    # Update base model and current model config
    model.config.vocab_size = new_num_tokens
    base_model.vocab_size = new_num_tokens

    # Tie weights again if needed
    model.tie_weights()
    return model


def get_embeddings_for_movie_name(tokenizer,model,saved_path,model_name,data_type):
    movieId = pd.read_csv(args.Movie_Name_Path[data_type])
    movieId = movieId.set_index('movieId').T.to_dict('list')
    new_tensor = None
    new_keys = []   
    for i,(key,value) in tqdm(enumerate(movieId.items()),total=len(movieId)):
        token = tokenizer.tokenize(value[0].lower())
        token_ids = tokenizer.convert_tokens_to_ids(token)
        token_tensor = torch.tensor(token_ids)
        embedding = model.get_context_embeddings(token_tensor)
        mean_tensor = torch.mean(embedding,dim=0).unsqueeze(0)

        new_keys.append('@'+ str(key))

        if i == 0:
            new_tensor = mean_tensor
        else:
            new_tensor = torch.cat((new_tensor,mean_tensor))
    print(new_tensor.size())
    torch.save(new_tensor,os.path.join(saved_path,'movie_embedding.pt'))  

    assert len(new_keys) == args.AT_TOKEN_NUMBER[data_type]
    tokenizer.add_special_tokens({'additional_special_tokens':new_keys})
    return tokenizer,new_tensor

def add_tokens_for_tokenizer(tokenizer,data_type):
    movieId = pd.read_csv(args.Movie_Name_Path[data_type])
    movieId = movieId.set_index('movieId').T.to_dict('list')
    new_keys = []   
    for i,(key,value) in tqdm(enumerate(movieId.items()),total=len(movieId)):
        new_keys.append('@'+ str(key))
    tokenizer.add_special_tokens({'additional_special_tokens':new_keys})
    return tokenizer
    
'''
----------------
faiss related
----------------
'''
def faiss_search(representation,faiss_index,dstore_keys,dstore_vals,dstore_labels,k,device):
    D,I = faiss_index.search(representation.detach().float().cpu().numpy(),k) #[bz,K]
    batch_faiss_vecs = None
    batch_faiss_labels = []
    for i,each in enumerate(I):
        each = np.delete(each, np.where(each == -1))
        faiss_vecs = torch.from_numpy(dstore_vals[each]).to(device) #[k,hidden_size]
        if dstore_labels is not None:
            faiss_labels = dstore_labels[each].tolist() #[k,1]
        if i  == 0:
            batch_faiss_vecs = faiss_vecs.unsqueeze(0)
        else:
            batch_faiss_vecs = torch.cat((batch_faiss_vecs,faiss_vecs.unsqueeze(0)),dim=0) #[bz,k,hidden_size]
        assert batch_faiss_vecs.shape[0] == i+1
        if dstore_labels is not None:
            batch_faiss_labels.append(faiss_labels)
    return {
        'batch_faiss_vecs':batch_faiss_vecs,
        'batch_faiss_labels':batch_faiss_labels  if dstore_labels is not None else None,
    }


def faiss_search_train_retrieval(faiss_index,dstore_keys,dstore_vals,dstore_labels,representation,correct_k_num,fake_k_num,device,groundTruth):
    search_k = correct_k_num + fake_k_num
    ref = None
    logist = None
    D,I = faiss_index.search(representation.detach().cpu().float().numpy(),search_k) 
    for i,each in enumerate(I):
        '''找出正例和负例'''
        label = groundTruth[i] 
        same_label_index = np.where(dstore_labels==label)[0]
        fake_label_index = np.where(dstore_labels!=label)[0]

        '''pretrain_dataset里有的movie_item没有'''
        if same_label_index.shape[0] < correct_k_num:
            # print(label)
            continue

        each = np.delete(each, np.where(each == -1))

        correct_index = np.delete(each,np.where(dstore_labels[each]!= label))
        same_label_index = np.setdiff1d(same_label_index,correct_index) # 新添加的不能是已经出现的
        if correct_index.shape[0] <correct_k_num:
            sample_num = correct_k_num-correct_index.shape[0]
            sample_correct = np.random.choice(same_label_index,sample_num,replace=False)
            correct_index = np.append(correct_index,sample_correct,axis=0)
        else:
            correct_index = np.random.choice(correct_index,correct_k_num,replace=False)
        
        fake_index = np.delete(each,np.where(dstore_labels[each]== label))
        fake_label_index = np.setdiff1d(fake_label_index,fake_index)
        if fake_index.shape[0] <fake_k_num:
            sample_num = fake_k_num-fake_index.shape[0]
            sample_fake = np.random.choice(fake_label_index,sample_num,replace=False)
            fake_index = np.append(fake_index,sample_fake,axis=0)
        else:
            fake_index = np.random.choice(fake_index,fake_k_num,replace=False)

        correct_keys =  dstore_keys[correct_index]
        fake_keys = dstore_keys[fake_index]

        knn_correct_vecs = torch.from_numpy(correct_keys).to(device) #[correct_k_num,hidden_size]
        knn_fake_vecs = torch.from_numpy(fake_keys).to(device) #[fake_k_num,hidden_size]
        knn = torch.cat((knn_correct_vecs,knn_fake_vecs),dim=0).unsqueeze(0) #[1,correct_k_num+fake_k_num,hidden_size]
        # each_weight = torch.matmul(representation[i],torch.from_numpy(dstore_keys[each]).to(device).T)
        knn_weight = torch.matmul(representation[i],knn.squeeze(0).T).unsqueeze(0) #[1,correct_k_num+fake_k_num]
        if ref == None:
            ref = knn
            logist = knn_weight
        else:
            ref = torch.cat((ref,knn),dim=0)
            logist = torch.cat((logist,knn_weight),dim=0)
    if logist is None:
        raise("make bz larger or correct_k_num smaller")
    correct_logist = logist[:,:correct_k_num] #[bz,correct_k_num]
    fake_logist = logist[:,correct_k_num:]#[bz,fake_k_num]
    logist = None
    for each in torch.split(correct_logist,1,dim=1):
        temp_logist = torch.cat((each,fake_logist),dim=1) #[bz,1+fake_k_num]
        if logist is None:
            logist = temp_logist
        else:
            logist = torch.cat((logist,temp_logist),dim=0) #[bz*correct_num,1+fake_num]
    return ref,logist

def test_faiss():
    pass