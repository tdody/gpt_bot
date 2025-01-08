from typing import Any
import torch.nn as nn


# Define the model class
class Model(nn.Module):
    def __init__(self, config: dict[str, Any]):
        super().__init__()
        self.config = config

    def forward(self, x):
        raise NotImplementedError

    def backward(self, x):
        raise NotImplementedError


# Bigram model
class BigramModel(Model):
    def __init__(self, config: dict[str, Any]):
        super().__init__(config)

    def forward(self, x):
        pass

    def backward(self, x):
        pass


# Trigram model
class TrigramModel(Model):
    def __init__(self, config: dict[str, Any]):
        super().__init__(config)

    def forward(self, x):
        pass

    def backward(self, x):
        pass
