import torch.nn as nn

# Define the model class
class Model(nn.Module):
    def __init__(self, config: dict[str, any]):
        super().__init__()
        self.config = config

    def forward(self, )
        raise NotImplementedError

    def backward(self, ):
        raise NotImplementedError

# Bigram model
class BigramModel(Model):
    def __init__(self, config: dict[str, any]):
        super().__init__(config)

    def forward(self, x):
       pass

    def backward(self, x):
        pass