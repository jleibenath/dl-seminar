import torch.nn as nn

DROPOUT = 0.0


class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(5184, 20),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(20, 10),
            nn.ReLU(),
            nn.Linear(10, 2),
        )

    def forward(self, x):
        return self.classifier(x)
