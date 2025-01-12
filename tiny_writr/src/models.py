from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class ModelConfig:
    vocab_size: int


# Define the model class
class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

    def forward(
        self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """ "
        Forward pass of the model.

        Args:
            idx (torch.Tensor): The input tensor.
            targets (Optional[torch.Tensor]): The target tensor.

        Returns:
            logits, loss: The logits and the loss stored as tensors.
        """
        raise NotImplementedError

    def backward(self, x):
        raise NotImplementedError


# Bigram model
class BigramModel(Model):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        n = config.vocab_size
        self.logits = nn.Parameter(torch.zeros((n, n)))

    def forward(
        self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.logits[idx]

        loss: Optional[torch.Tensor] = None

        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        return logits, loss

    def backward(self, x):
        # No backward pass needed for this model
        pass


# Trigram model
class TrigramModel(Model):
    def __init__(self, config: dict[str, Any]):
        super().__init__(config)

    def forward(self, x):
        pass

    def backward(self, x):
        pass
