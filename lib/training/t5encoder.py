import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration, PreTrainedModel

class T5EncoderForSequenceClassification(PreTrainedModel):

    def __init__(self, model_name, num_labels):
        super(T5EncoderForSequenceClassification, self).__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.num_labels = num_labels
        self.classifier = nn.Linear(self.model.config.d_model, num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        # Forward pass through the model encoder only
        outputs = self.model.get_encoder()(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_states = outputs.last_hidden_state
        # Classification head
        loss = None
        logits = self.classifier(last_hidden_states[:, 0, :])
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        
        return {
            'logits': logits,
            'last_hidden_states': last_hidden_states,
            'loss': loss
        }
    
class T5EncoderForTokenClassification(PreTrainedModel):

    def __init__(self, model_name, num_labels):
        super(T5EncoderForTokenClassification, self).__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.num_labels = num_labels
        self.classifier = nn.Linear(self.model.config.d_model, num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        # Forward pass through the model encoder only
        outputs = self.model.get_encoder()(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_states = outputs.last_hidden_state
        # Classification head
        logits = self.classifier(last_hidden_states)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return {
            'logits': logits,
            'last_hidden_states': last_hidden_states,
            'loss': loss
        }