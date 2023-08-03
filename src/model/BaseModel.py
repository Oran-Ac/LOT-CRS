from transformers import BertForMaskedLM,BartForConditionalGeneration
"""
python中__init__使用继承:

"""
class BaseCRS():
    def get_last_hidden_states(self,outputs):
        raise NotImplementedError
    def get_representation(self,batch,position,last_hidden_states=None,first_hidden_states=None,bias = 0,return_last_hidden_states=False):
        if last_hidden_states is None:
            last_hidden_states = self.get_last_hidden_states(batch)
        if first_hidden_states is None:
            first_hidden_states = self.get_first_hidden_states(batch)
        if position == 'msk':
            representation = last_hidden_states[torch.where(batch['context_batch_mlm_position']>0)]
        elif position == 'cls':
            representation = last_hidden_states[:,0+bias]#[bz,hidden]]
        elif position == 'last':
            representation = last_hidden_states[torch.where(batch['context_batch_last_position']>0)]
        elif position == 'avg_first_last':
            representation = ((first_hidden_states + last_hidden_states) / 2.0 * batch['context_batch_attn'].unsqueeze(-1)).sum(1) / batch['context_batch_attn'].sum(-1).unsqueeze(-1)
        else:
            return None,last_hidden_states

        if return_last_hidden_states:
            return representation,last_hidden_states
        else:
            return representation,None

    def get_representation_for_query(self,batch,position,model_outputs,bias = 0,return_last_hidden_states=False):
        
        last_hidden_states = self.get_last_hidden_states(model_outputs)
        first_hidden_states = self.get_first_hidden_states(model_outputs)
        return self.get_representation(batch,position,last_hidden_states,first_hidden_states,return_last_hidden_states=return_last_hidden_states)

    def vocab_head(self,representation):
        raise NotImplementedError
    
    def get_context_embeddings(self,context):
        raise NotImplementedError

    

    
    
class BertCRS(BertForMaskedLM,BaseCRS):
    def get_last_hidden_states(self,outputs):
        return outputs['hidden_states'][-1]
    def get_first_hidden_states(self,outputs):
        return outputs['hidden_states'][1]
    def vocab_head(self,representation):
        return self.cls(representation)
    
    def get_context_embeddings(self,context):
        return self.bert.embeddings.word_embeddings(context)



class BartCRS(BartForConditionalGeneration,BaseCRS):
    def get_last_hidden_states(self,outputs):
        return outputs['decoder_hidden_states'][-1]
    def get_first_hidden_states(self,outputs):
        return outputs['encoder_hidden_states'][1]
    def vocab_head(self,representation):
        return self.lm_head(representation) + self.final_logits_bias.to(representation.device)
    def get_context_embeddings(self,context):
        return self.model.encoder.embed_tokens(context)
    
    