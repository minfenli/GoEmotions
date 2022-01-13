import torch.nn as nn
from torch import ones, log, sum, rand_like, cuda
from transformers import BertPreTrainedModel, BertModel

class BertForMultiLabelClassification(BertPreTrainedModel):
    def __init__(self, config, smoothing = 0.0, smoothing_by_norm = False, pos_weight = None, focal_loss = False, gamma=2, alpha=0.6, linear_dropout_prob = 0.5):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.batch_norm = nn.BatchNorm1d(config.hidden_size)
        self.dropout = nn.Dropout(linear_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)
        self.pos_weight = self.init_pos_weight(pos_weight)
        self.sigmoid = nn.Sigmoid()
        self.loss_fct = nn.BCELoss()
        self.loss_fct_logit = nn.BCEWithLogitsLoss(pos_weight = self.pos_weight)
        self.loss_fct_focal = self.init_focal_loss(focal_loss, gamma, alpha, self.pos_weight)
        self.label_smt = LabelSmoothing(smoothing, smoothing_by_norm)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        pooled_output = outputs[1]

        pooled_output = self.batch_norm(pooled_output)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.loss_fct_focal is not None:
                logits = self.sigmoid(logits)
                targets = self.label_smt(labels)
                loss = self.loss_fct_focal(logits, targets)
                #print(loss)
                outputs = (loss,) + outputs
            else:
                targets = self.label_smt(labels)
                loss = self.loss_fct_logit(logits, targets)
                #print(loss)
                #loss = self.loss_fct(self.sigmoid(logits), self.label_smt(labels))
                outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)
    
    def init_pos_weight(self, pos_weight):
        if(pos_weight == None or pos_weight.shape[0] != self.num_labels):
            return None
        return pos_weight.cuda() if cuda.is_available() else pos_weight
    def init_focal_loss(self, focal_loss, gamma, alpha, pos_weight):
        if focal_loss:
            return FocalLoss(gamma, alpha, pos_weight)
        else: 
            focal_loss = None
    
class LabelSmoothing(nn.Module):
    def __init__(self, smoothing=0.0, norm = False):
        super(LabelSmoothing, self).__init__()
        self.smoothing = smoothing
        self.norm = norm

    def forward(self, x):
        if self.norm:
            return x + (1-2*x) * rand_like(x)*self.smoothing
        else:
            return x + (1-2*x) * self.smoothing
        
    
class FocalLoss(nn.Module):
    def __init__(self, gamma, alpha, pos_weight = None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.pos_weight = pos_weight

    def forward(self, x, target):
        if self.pos_weight is not None:
            return sum(self.pos_weight*(- self.alpha * (1-x)**self.gamma *target*log(x) - (1-self.alpha)* x**self.gamma *(1-target)*log(1-x)))
        else:
            return sum(- self.alpha * (1-x)**self.gamma * target * log(x) - (1-self.alpha) * x**self.gamma *(1-target)*log(1-x))