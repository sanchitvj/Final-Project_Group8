import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
import torch.nn.init as init


'''
For pooling refer to
https://www.kaggle.com/code/rhtsingh/utilizing-transformer-representations-efficiently
'''
class MeanPooling(nn.Module):
    def __init__(self, backbone_config):
        super(MeanPooling, self).__init__()
        self.output_dim = backbone_config.hidden_size

    def forward(self, inputs, backbone_outputs):
        attention_mask = inputs["attention_mask"]
        last_hidden_state = backbone_outputs[0]

        # below operation is done to expand att mask: [batch_size, max_len] -> [batch_size, max_len, hidden_size]
        # https://www.kaggle.com/code/rhtsingh/utilizing-transformer-representations-efficiently?scriptVersionId=108281817&cellId=7
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        # multiplied to only include real tokens (value 1) and exclude padding tokens (value 0)
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        # summing along max_len dim to ignore padding token
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)  # avoiding division by zero
        mean_embeddings = sum_embeddings / sum_mask  # averaging
        return mean_embeddings


class LSTMPooling(nn.Module):
    def __init__(self, backbone_config, pooling_config, is_lstm=True):
        super(LSTMPooling, self).__init__()

        self.num_hidden_layers = backbone_config.num_hidden_layers
        self.hidden_size = backbone_config.hidden_size
        self.hidden_lstm_size = pooling_config.hidden_size
        self.dropout_rate = pooling_config.dropout_rate
        self.bidirectional = pooling_config.bidirectional

        self.is_lstm = is_lstm
        self.output_dim = pooling_config.hidden_size*2 if self.bidirectional else pooling_config.hidden_size

        if self.is_lstm:
            self.lstm = nn.LSTM(self.hidden_size,
                                self.hidden_lstm_size,
                                bidirectional=self.bidirectional,
                                batch_first=True)
        else:
            self.lstm = nn.GRU(self.hidden_size,
                               self.hidden_lstm_size,
                               bidirectional=self.bidirectional,
                               batch_first=True)

        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, inputs, backbone_outputs):
        all_hidden_states = torch.stack(backbone_outputs[1])  # ????????????????????????????????????

        # take only [CLS] token into account; it is 1st token aggregating all the info from entire sequence
        hidden_states = torch.stack([all_hidden_states[layer_i][:, 0].squeeze()
                                     for layer_i in range(1, self.num_hidden_layers + 1)], dim=-1)
        # [batch_size, num_hidden_layers, hidden_size]
        hidden_states = hidden_states.view(-1, self.num_hidden_layers, self.hidden_size)
        out, _ = self.lstm(hidden_states, None)
        out = self.dropout(out[:, -1, :])
        return out


class CustomModel(nn.Module):
    def __init__(self, cfg, backbone_config):
        super(CustomModel, self).__init__()
        self.cfg = cfg
        self.backbone_config = backbone_config

        # if self.cfg.model.use_pretrained:
        self.backbone = AutoModel.from_pretrained(self.cfg.model.model_type, config=self.backbone_config)

        # Adjust the embedding size of the transformer model
        self.backbone.resize_token_embeddings(len(self.cfg.tokenizer))

        # Define a pooling layer and a fully connected layer
        self.pooling_layer = get_pooling_layer(self.cfg, self.backbone_config)
        self.classifier = nn.Linear(self.pooling_layer.output_dim, len(self.cfg.general.target_labels))

        # Initialize weights of the classifier
        self.initialize_weights(self.classifier)

    def initialize_weights(self, module, init_type='normal'):
        if isinstance(module, nn.Linear):
            if init_type == 'xavier':
                init.xavier_uniform_(module.weight)
            elif init_type == 'normal':
                init.normal_(module.weight, mean=0.0, std=self.backbone_config.initializer_range)
            if module.bias is not None:
                init.zeros_(module.bias)

    def forward(self, inputs):
        backbone_outputs = self.backbone(**inputs)
        pooled_features = self.pooling_layer(inputs, backbone_outputs)
        final_output = self.classifier(pooled_features)
        return final_output
