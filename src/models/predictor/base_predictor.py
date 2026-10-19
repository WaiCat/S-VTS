
import torch
import torch.nn as nn

from ..modules.loss_layer import LossLayer


class BasePredictor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.loss_layer = LossLayer(config)

    def get_logits(self, fused_features):
        raise NotImplementedError("get_logits function is not implemented")
    
    
    def fuse_features(self, text_features, visual_features):
        """Paper Eq. (4): m_i = h_i^v \u2295 h_i^t."""
        if self.config.fuse_type != "cat_t_v":
            raise ValueError("paper implementation requires fuse_type='cat_t_v'")
        return torch.cat((visual_features, text_features), dim=1)
