import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import RGCNConv,GCNConv
import logging



def edge_to_pyg_format(edge, type='RGCN'):
    if type == 'RGCN':
        edge_sets = torch.as_tensor(edge, dtype=torch.long)
        edge_idx = edge_sets[:, :2].t()
        edge_type = edge_sets[:, 2]
        return edge_idx, edge_type
    elif type == 'GCN':
        edge_set = [[co[0] for co in edge], [co[1] for co in edge]]
        return torch.as_tensor(edge_set, dtype=torch.long)
    else:
        raise NotImplementedError('type {} has not been implemented', type)

# copy and paste
class GateLayer(nn.Module):
    def __init__(self, input_dim):
        super(GateLayer, self).__init__()
        self._norm_layer1 = nn.Linear(input_dim * 2, input_dim)
        self._norm_layer2 = nn.Linear(input_dim, 1)

    def forward(self, input1, input2):
        norm_input = self._norm_layer1(torch.cat([input1, input2], dim=-1))
        gate = torch.sigmoid(self._norm_layer2(norm_input))  # (bs, 1)
        gated_emb = gate * input1 + (1 - gate) * input2  # (bs, dim)
        return gated_emb

# copy and paste
class SelfAttentionSeq(nn.Module):
    def __init__(self, dim, da, alpha=0.2, dropout=0.5):
        super(SelfAttentionSeq, self).__init__()
        self.dim = dim
        self.da = da
        self.alpha = alpha
        self.dropout = dropout
        self.a = nn.Parameter(torch.zeros(size=(self.dim, self.da)), requires_grad=True)
        self.b = nn.Parameter(torch.zeros(size=(self.da, 1)), requires_grad=True)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        nn.init.xavier_uniform_(self.b.data, gain=1.414)

    def forward(self, h, mask=None, return_logits=False):
        """
        For the padding tokens, its corresponding mask is True
        if mask==[1, 1, 1, ...]
        """
        # h: (batch, seq_len, dim), mask: (batch, seq_len)
        e = torch.matmul(torch.tanh(torch.matmul(h, self.a)), self.b)  # (batch, seq_len, 1)
        if mask is not None:
            full_mask = -1e30 * mask.float()
            batch_mask = torch.sum((mask == False), -1).bool().float().unsqueeze(-1)  # for all padding one, the mask=0
            mask = full_mask * batch_mask
            e += mask.unsqueeze(-1)
        attention = F.softmax(e, dim=1)  # (batch, seq_len, 1)
        # (batch, dim)
        if return_logits:
            return torch.matmul(torch.transpose(attention, 1, 2), h).squeeze(1), attention.squeeze(-1)
        else:
            return torch.matmul(torch.transpose(attention, 1, 2), h).squeeze(1)


# only contains the recommendation modules
class KGSF(nn.Module):
    def __init__(self,opt,device,load_path=None):
       super(KGSF, self).__init__()
       self.pretrained_embedding = None
       # vocab
       self.pad_token_idx = opt['vocab']['pad_token_idx']
       self.pad_word_idx = opt['vocab']['pad_word_idx']
       self.pad_entity_idx = opt['vocab']['pad_entity_idx']

       self.movie_ids = opt['movie_ids']
       # graph
       self.n_entity = opt['graph']['n_entity']
       self.n_word = opt['graph']['n_word']
       self.n_relation = opt['graph']['entity_kg']['n_relation']
       self.num_bases = opt['num_bases']
       entity_edges = opt['graph']['entity_kg']['edge']
       self.entity_edge_idx, self.entity_edge_type = edge_to_pyg_format(entity_edges, 'RGCN')
       self.word_edges = opt['graph']['word_kg']['edge']
       self.word_edges = edge_to_pyg_format(self.word_edges, 'GCN').to(device)


       self.entity_edge_idx = self.entity_edge_idx.to(device)
       self.entity_edge_type = self.entity_edge_type.to(device)


       #demension
       self.token_emb_dim = opt['token_emb_dim']
       self.kg_emb_dim = opt['kg_emb_dim']

       self.build_model()
       if load_path is not  None:
        print("[Load the trained kgsf]")
        self.load_model(load_path)

       


    def load_model(self,path):
        check_point= torch.load(path)
        if 'model_state_dict' in check_point: #load from crslab kgsf
            # copyed from crslab
            check_point['model_state_dict']['module.word_encoder.lin.weight'] = check_point['model_state_dict']['module.word_encoder.weight']
            check_point['model_state_dict'].pop('module.word_encoder.weight')
            # to make it better for our experiments
            keeped_key = [ "word_kg_embedding.weight", "entity_encoder.weight", "entity_encoder.comp", "entity_encoder.root", "entity_encoder.bias", "entity_self_attn.a", "entity_self_attn.b", "word_encoder.bias", "word_encoder.lin.weight", "word_self_attn.a", "word_self_attn.b", "gate_layer._norm_layer1.weight", "gate_layer._norm_layer1.bias", "gate_layer._norm_layer2.weight", "gate_layer._norm_layer2.bias", "infomax_norm.weight", "infomax_norm.bias", "infomax_bias.weight", "infomax_bias.bias", "rec_bias.weight", "rec_bias.bias"]
            prefix = 'module'
            for k in keeped_key:
                if prefix+ '.' + k in  check_point['model_state_dict']:
                    check_point[k] = check_point['model_state_dict'][prefix+ '.' + k]
            check_point.pop('model_state_dict')

        self.load_state_dict(check_point,strict=True) 
    def build_model(self):
        self._init_embeddings()
        self._build_kg_layer()
        self._build_infomax_layer()
        self._build_recommendation_layer()

    def _init_embeddings(self):
        
        self.word_kg_embedding = nn.Embedding(self.n_word, self.kg_emb_dim, self.pad_word_idx)
        nn.init.normal_(self.word_kg_embedding.weight, mean=0, std=self.kg_emb_dim ** -0.5)
        nn.init.constant_(self.word_kg_embedding.weight[self.pad_word_idx], 0)

        logging.info('[Finish init embeddings]')

    def _build_kg_layer(self):
            # db encoder
        self.entity_encoder = RGCNConv(self.n_entity, self.kg_emb_dim, self.n_relation, self.num_bases)
        self.entity_self_attn = SelfAttentionSeq(self.kg_emb_dim, self.kg_emb_dim)

        # concept encoder
        self.word_encoder = GCNConv(self.kg_emb_dim, self.kg_emb_dim)
        self.word_self_attn = SelfAttentionSeq(self.kg_emb_dim, self.kg_emb_dim)

        # gate mechanism
        self.gate_layer = GateLayer(self.kg_emb_dim)

        logging.debug('[Finish build kg layer]')

    def _build_infomax_layer(self):
        self.infomax_norm = nn.Linear(self.kg_emb_dim, self.kg_emb_dim)
        self.infomax_bias = nn.Linear(self.kg_emb_dim, self.n_entity)
        self.infomax_loss = nn.MSELoss(reduction='sum')

        logging.debug('[Finish build infomax layer]')

    def _build_recommendation_layer(self):
        self.rec_bias = nn.Linear(self.kg_emb_dim, self.n_entity)
        self.rec_loss = nn.CrossEntropyLoss()

        logging.debug('[Finish build rec layer]')

    def recommend(self, batch,mode):
        """
        context_entities: (batch_size, entity_length)
        context_words: (batch_size, word_length)
        movie: (batch_size)
        """
        context_entities, context_words = batch['entity_batch'],batch['word_batch']

        entity_graph_representations = self.entity_encoder(None, self.entity_edge_idx, self.entity_edge_type)
        word_graph_representations = self.word_encoder(self.word_kg_embedding.weight, self.word_edges)

        entity_padding_mask = context_entities.eq(self.pad_entity_idx)  # (bs, entity_len)
        word_padding_mask = context_words.eq(self.pad_word_idx)  # (bs, word_len)

        entity_representations = entity_graph_representations[context_entities]
        word_representations = word_graph_representations[context_words]

        entity_attn_rep = self.entity_self_attn(entity_representations, entity_padding_mask)
        word_attn_rep = self.word_self_attn(word_representations, word_padding_mask)
        
       
        
        user_rep = self.gate_layer(entity_attn_rep, word_attn_rep)
        rec_scores = F.linear(user_rep, entity_graph_representations, self.rec_bias.bias)  # (bs, #entity)

        # 注意不同mode下的rec_scores不同
        if mode == 'teacher':
            return {
                'rec_scores':rec_scores[:,self.movie_ids], #convert #n_entity to #n_movie
                'scores':rec_scores,
                'user_rep':user_rep,
                'word_representations':word_representations,  #[batch_size,context_words_num,max_len=128]
                'word_padding_mask':word_padding_mask,#[batch_size,context_words_num]
                'entity_representations':entity_representations, #[batch_size,context_entities_num,max_len=128]
                'entity_padding_mask':entity_padding_mask,  #[batch_size,context_entities_num]
                'entity_graph_representations':entity_graph_representations
            }
        elif mode == 'train' or mode == 'test':
            movie = batch['item_batch']
            rec_loss = self.rec_loss(rec_scores, movie)

            entity_labels = batch['entity_labels_batch']
            info_loss_mask = torch.sum(entity_labels)
            if info_loss_mask.item() == 0:
                info_loss = None
            else:
                word_info_rep = self.infomax_norm(word_attn_rep)  # (bs, dim)
                info_predict = F.linear(word_info_rep, entity_graph_representations,
                                        self.infomax_bias.bias)  # (bs, #entity)
                info_loss = self.infomax_loss(info_predict, entity_labels) / info_loss_mask

            return {
                'rec_loss':rec_loss,
                'info_loss':info_loss,
                'rec_scores':rec_scores, #(bs, #entity)
                'rec_movie_scores':rec_scores[:,self.movie_ids], #convert #n_entity to #n_movie
            }
    
    def pretrain(self, batch):
        """
        words: (batch_size, word_length)
        entity_labels: (batch_size, n_entity)
        """
        words, entity_labels = batch['word_batch'],batch['entity_labels_batch']

        loss_mask = torch.sum(entity_labels)
        if loss_mask.item() == 0:
            return None

        entity_graph_representations = self.entity_encoder(None, self.entity_edge_idx, self.entity_edge_type)
        word_graph_representations = self.word_encoder(self.word_kg_embedding.weight, self.word_edges)

        word_representations = word_graph_representations[words]
        word_padding_mask = words.eq(self.pad_word_idx)  # (bs, seq_len)

        word_attn_rep = self.word_self_attn(word_representations, word_padding_mask)
        word_info_rep = self.infomax_norm(word_attn_rep)  # (bs, dim)
        info_predict = F.linear(word_info_rep, entity_graph_representations, self.infomax_bias.bias)  # (bs, #entity)
        loss = self.infomax_loss(info_predict, entity_labels) / loss_mask
        return {
            'info_loss':loss,
        }

    def forward(self,batch,mode):
        if mode == 'teacher_recommend':
            return self.recommend(batch,'teacher')
        elif mode == 'train':
            return self.recommend(batch,'train')
        elif mode == 'test':
            return self.recommend(batch,'test')
        elif mode == 'pretrain':
            return self.pretrain(batch)