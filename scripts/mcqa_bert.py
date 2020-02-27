import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import BertModel, BertPreTrainedModel


class BertForMCQA(BertPreTrainedModel):
    def __init__(self, config):
        super(BertQAJoint, self).__init__(config)

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)

        self.init_weights()

    def forward(self, input_pair_list=None, labels=None)
        logits = []
        for input_pair in input_pair_list:
            input_ids, attention_mask, token_type_ids = input_pair
            outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            pooled_output = outputs[1]
            pooled_output = self.dropout(pooled_output)
            logit = self.classifier(pooled_output)
            logits.append(logit)
        logits = torch.cat(logits, dim=-1)
        
        outputs = (logits,)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            outputs = (loss,) + outputs
            
        return outputs  # (loss), logits
    