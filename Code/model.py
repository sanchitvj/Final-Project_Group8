import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from torch.utils.checkpoint import checkpoint


class CustomModel(nn.Module):
    def __init__(self, cfg, backbone_config):
        super().__init__()
        self.cfg = cfg
        self.backbone_config = backbone_config

        if self.cfg.model.pretrained_backbone:
            self.backbone = AutoModel.from_pretrained(cfg.model.backbone_type, config=self.backbone_config)
        else:
            self.backbone = AutoModel.from_config(self.backbone_config)

        self.backbone.resize_token_embeddings(len(cfg.tokenizer))
        self.pool = get_pooling_layer(cfg, backbone_config)
        self.fc = nn.Linear(self.pool.output_dim, len(self.cfg.general.target_columns))

        # self._init_weights(self.fc)

    def forward(self, inputs):
        outputs = self.backbone(**inputs)
        feature = self.pool(inputs, outputs)
        output = self.fc(feature)
        return output


