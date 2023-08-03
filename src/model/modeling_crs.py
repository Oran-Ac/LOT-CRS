
import torch
from torch import nn
from torch.nn import functional as F
from utils.args import AT_TOKEN_NUMBER
class CRSModel(nn.Module):
    def __init__(self,backbone_model,model_type,query_position=None,opt=None,add_knowledge_prompt=False,conv=False):
        super().__init__()
        self.backbone_model = backbone_model
        self.model_type = model_type
        self.query_position = query_position
        self.add_knowledge_prompt = add_knowledge_prompt

        self.faiss_aug =  nn.MultiheadAttention(self.backbone_model.config.hidden_size,1,dropout=0,batch_first =True)
        self.layerNorm = nn.LayerNorm(self.backbone_model.config.hidden_size)
        # self.dense_query = nn.Sequential(
        #     nn.Linear(self.backbone_model.config.hidden_size,self.backbone_model.config.hidden_size),
        #     nn.tanh(),
        #     nn.LayerNorm(self.backbone_model.config.hidden_size)
        # )
        self.rec_loss = nn.CrossEntropyLoss()
        if opt is not None:
            self.movie_ids = opt['movie_ids']
            self.data_type = opt['data_type']
            if 'conv' in opt and conv:
                self.conv_hidden_size = opt['conv']['hidden_size']
                self.conv_num_layers = opt['conv']['num_layers']
                self.conv_num_blocks = opt['conv']['num_blocks']
                self.conv_num_heads = opt['conv']['num_heads']
                self.conv_head_dim = self.conv_hidden_size // self.conv_num_heads
        if self.add_knowledge_prompt:
            self.build_knowledge_prompt()  
        if conv:  
            self.build_conv()

    def build_knowledge_prompt(self):
        # with knowledge
        self.fc_word = nn.Linear(128,self.backbone_model.config.hidden_size) # word_rep:[batch,num,kg_dim],kg_dim = 128,max_len = 256
        self.fc_entity = nn.Linear(128,self.backbone_model.config.hidden_size) # entity_rep:[batch,num,kg_dim],kg_dim = 128,max_len = 256
        self.fc_graph_representation_1 = nn.Sequential(
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,128),
        )
        self.fc_graph_representation_2 = nn.Linear(128,self.backbone_model.config.hidden_size) #here is vocab_size


    def build_conv(self):
        self.faiss_aug_conv = nn.MultiheadAttention(self.backbone_model.config.hidden_size,1,dropout=0,batch_first =True)
        self.conv_token_proj1 = nn.Sequential(
            nn.Linear(self.backbone_model.config.hidden_size, self.backbone_model.config.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(self.backbone_model.config.hidden_size // 2, self.backbone_model.config.hidden_size),
        )
        self.conv_token_proj2 = nn.Linear(self.backbone_model.config.hidden_size, self.conv_hidden_size)

        self.conv_prompt_proj1 = nn.Sequential(
            nn.Linear(self.conv_hidden_size, self.conv_hidden_size // 2),
            nn.ReLU(),
            nn.Linear(self.conv_hidden_size // 2, self.conv_hidden_size),
        )
        self.conv_prompt_proj2 = nn.Linear(self.conv_hidden_size, self.conv_num_layers * self.conv_num_blocks * self.conv_hidden_size)


    def get_query_representation(self,batch):
        outputs = self.backbone_model(
                                        batch['context_batch'],
                                        attention_mask = batch['context_batch_attn'],
                                        output_hidden_states=True
                                )
        representation,last_hidden_states = self.backbone_model.get_representation_for_query(
                                            batch,
                                            position = self.query_position,
                                            model_outputs=outputs,
                                            return_last_hidden_states=True)

        # query_representation = self.dense_query(representation)
        # return query_representation,last_hidden_states
        return representation,last_hidden_states    

    def faiss_aug_recommendation(self,batch,faiss_weight):
        last_hidden_states = batch['last_hidden_states']
        faiss_aug_representation = batch['faiss_aug_representation']
        faiss_aug_representation = faiss_aug_representation.to(last_hidden_states.device)

        attn,attn_output_weights = self.faiss_aug(last_hidden_states,faiss_aug_representation,faiss_aug_representation)
        new_hidden_layer = self.layerNorm(last_hidden_states + attn*faiss_weight)


        representation = self.backbone_model.vocab_head(new_hidden_layer) #[bz,seq_len,vocab_size]

        # get the classification scores
        assert batch['context_batch_mlm_position'].shape[1] == representation.shape[1] #确保引入了knowledge还是正确的
        rec_scores = representation[torch.where(batch['context_batch_mlm_position']>0)] #[bz,vocab_size]
        added_faiss_aug_last_hidden_state = new_hidden_layer[torch.where(batch['context_batch_mlm_position']>0)]

        
        return rec_scores,added_faiss_aug_last_hidden_state
    
    def faiss_aug_conversation(self,batch):
        last_hidden_states = batch['last_hidden_states']
        faiss_aug_representation = batch['faiss_aug_representation']
        faiss_aug_representation = faiss_aug_representation.to(last_hidden_states.device)

        attn,attn_output_weights = self.faiss_aug_conv(last_hidden_states,faiss_aug_representation,faiss_aug_representation)
        new_hidden_layer = self.layerNorm(last_hidden_states + attn)

        token_embeds = self.conv_token_proj1(new_hidden_layer) + new_hidden_layer
        token_embeds = self.conv_token_proj2(token_embeds)
        
        return token_embeds
        
        
        
    
    def get_query_representation_with_knowledge_prompt(self,batch):
        word_representations_embeds = self.fc_word(batch['word_representations'])
        entity_representations_embeds = self.fc_entity(batch['entity_representations'])
        word_attention_mask = batch['word_batch_attn']
        entity_attention_mask = batch['entity_batch_attn']

        knowledge_attention_mask = torch.cat((word_attention_mask,entity_attention_mask),dim=1).to(word_attention_mask.device)
        knowledge_filled_position = torch.zeros(knowledge_attention_mask.shape[0],knowledge_attention_mask.shape[1]).long().to(word_attention_mask.device)
        attention_mask = torch.cat((knowledge_attention_mask,batch['context_batch_attn']),dim=1) #注意embeds也要用这个顺序连接
        position_ids = torch.cat((torch.zeros(knowledge_attention_mask.shape[1]),torch.arange(1,batch['context_batch'].shape[1]+1)),dim=0).long().expand((1,-1)).to(word_attention_mask.device)
        

        inputs_embeds = self.backbone_model.get_context_embeddings(batch['context_batch'])
        inputs_embeds = torch.cat((word_representations_embeds,entity_representations_embeds,inputs_embeds),dim=1)
        assert inputs_embeds.shape[1] == attention_mask.shape[1]
 

        outputs = self.backbone_model(
                                        inputs_embeds=inputs_embeds,
                                        position_ids=position_ids,
                                        attention_mask = attention_mask,
                                        output_hidden_states=True,
                                )
        
        if batch['context_batch_last_position'] is not None: #有的没有eos就没有‘context_batch_last_position’
            context_batch_last_position = torch.cat((knowledge_filled_position,batch['context_batch_last_position']),dim=1)
            batch['context_batch_last_position'] = context_batch_last_position #这样之后调用外部就是正确的长度
        
        if batch['context_batch_mlm_position'] is not None:
            context_batch_mlm_position = torch.cat((knowledge_filled_position,batch['context_batch_mlm_position']),dim=1)
            batch['context_batch_mlm_position'] = context_batch_mlm_position #这样之后调用外部就是正确的长度
            


        representation,last_hidden_states = self.backbone_model.get_representation_for_query(
                                            batch,
                                            position = self.query_position,
                                            model_outputs=outputs,
                                            bias = knowledge_filled_length,
                                            return_last_hidden_states=True)
        if representation is None:
            return last_hidden_states
        
        query_representation = self.dense_query(representation)
        return query_representation,last_hidden_states
    

    def forward(self,batch,mode,faiss_weight=1):
        if mode == 'query_representation':
            if self.add_knowledge_prompt:
                return self.get_query_representation_with_knowledge_prompt(batch)
            else:
                return self.get_query_representation(batch)
        elif mode == 'faiss_aug_recommendation':
            rec_scores,added_faiss_aug_last_hidden_state= self.faiss_aug_recommendation(batch,faiss_weight)
            if self.add_knowledge_prompt:
                graph_representation = self.fc_graph_representation_1(batch['entity_graph_representations']) + batch['entity_graph_representations']
                graph_representation = self.fc_graph_representation_2(graph_representation)
                rec_scores = added_faiss_aug_last_hidden_state @ graph_representation.T #[bz,n_item]
                rec_dbpedia_label_batch = torch.tensor([self.movie_ids[m] for m in batch['rec_dbpedia_movie_label_batch'].tolist()]).long().to(rec_scores.device)
                rec_loss = self.rec_loss(rec_scores,rec_dbpedia_label_batch) # 由于此时为n_item，所以gt也为n_item
                movie_scores = rec_scores[:,self.movie_ids]
            else:
                rec_loss  = self.rec_loss(rec_scores,batch['label_batch']) 
                movie_scores = rec_scores[:,-AT_TOKEN_NUMBER[self.data_type]:] # verblizer only care about the movie token
            return {
                        'movie_scores':movie_scores,
                        'rec_loss':rec_loss
                        }
            

        elif mode == 'pretrain':
            if self.add_knowledge_prompt:
                last_hidden_states = self.get_query_representation_with_knowledge_prompt(batch)
            else:
                last_hidden_states = self.get_query_representation(batch)
            representation = self.backbone_model.vocab_head(last_hidden_states) #[bz,seq_len,vocab_size]
            loss_fct = nn.CrossEntropyLoss()
            masked_lm_loss = loss_fct(representation.view(-1,self.backbone_model.config.vocab_size),batch['labels'].view(-1))
            return masked_lm_loss
        
        elif mode == 'faiss_aug_conversation':
            token_embeds = self.faiss_aug_conversation(batch)
            if self.add_knowledge_prompt:
                raise
            else:
                prompt_embeds = token_embeds
            prompt_embeds = self.conv_prompt_proj1(prompt_embeds) + prompt_embeds
            prompt_embeds = self.conv_prompt_proj2(prompt_embeds)
            batch_size = prompt_embeds.shape[0]
            return prompt_embeds.reshape(
                        batch_size, -1, self.conv_num_layers, self.conv_num_blocks, self.conv_num_heads, self.conv_head_dim
                ).permute(2, 3, 0, 4, 1, 5)  # (n_layer, n_block, batch_size, n_head, prompt_len, head_dim)

        else:
            raise
