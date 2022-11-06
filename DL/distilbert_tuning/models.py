from transformers import DistilBertModel, DistilBertConfig
import torch.nn as nn

class DBertClassifier(nn.Module):
    def __init__(self, pretrained_name, n_classes=2, layer_dropout=0.2):
        super(DBertClassifier, self).__init__()
        config = DistilBertConfig(dropout=layer_dropout,
                                 output_hidden_states=False)

        self.bert_model = DistilBertModel.from_pretrained(pretrained_name,
                                                          config=config)

        self.classifier = nn.Sequential(
            nn.Linear(768*2, n_classes),
          #  nn.ReLU(),
          #  nn.Linear(256, n_classes),
            nn.Softmax(dim=-1)
        )

    def forward(self, input_ids, attention_mask=None):
        last_hidden_states = self.bert_model(input_ids=input_ids,attention_mask=attention_mask).last_hidden_state[:, 0:2, :]
        last_hidden_states = last_hidden_states.reshape(-1, 768*2)
        final_scores = self.classifier(last_hidden_states)

        return final_scores
