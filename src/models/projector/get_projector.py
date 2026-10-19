
from .linear_projector import LinearProjector


def get_projector(config):
    return LinearProjector(config)
    
