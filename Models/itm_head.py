import torch.nn as nn

class ITMHead(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        return self.fc(x)
