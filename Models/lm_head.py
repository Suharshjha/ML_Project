import torch.nn as nn

class LMHead(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, 30522)  # BERT vocab

    def forward(self, x):
        return self.fc(x)
