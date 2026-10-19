
from .ma_moe_encoder import MergeAttentionMoEEncoder


def get_cross_encoder(config):
    if config.cross_encoder_type == "ma_moe":
        return MergeAttentionMoEEncoder(config)
    raise ValueError("paper implementation requires cross_encoder_type='ma_moe'")
