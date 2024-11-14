import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration

class T5EncoderForSequenceClassification(nn.Module):

    def __init__(self, model_name, num_labels):
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.num_labels = num_labels
        self.classifier = nn.Linear(self.model.config.d_model, num_labels)

    def forward(self, input_ids, attention_mask):
        # Forward pass through the model encoder only
        outputs = self.model.get_encoder()(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_states = outputs.last_hidden_state
        # Classification head
        logits = self.classifier(last_hidden_states[:, 0, :])
        return logits
    
class T5EncoderForTokenClassification(nn.Module):

    def __init__(self, model_name, num_labels):
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.num_labels = num_labels
        self.classifier = nn.Linear(self.model.config.d_model, num_labels)

    def forward(self, input_ids, attention_mask):
        # Forward pass through the model encoder only
        outputs = self.model.get_encoder()(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_states = outputs.last_hidden_state
        # Classification head
        logits = self.classifier(last_hidden_states)
        return logits