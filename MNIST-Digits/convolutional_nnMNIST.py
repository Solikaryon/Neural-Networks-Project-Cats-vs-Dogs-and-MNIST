import torch
import torch.nn as nn
import torch.nn.functional as F

class MinPool2d(nn.Module):
    def __init__(self, kernel_size=2, stride=2, padding=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
    def forward(self, x):
        return -F.max_pool2d(-x, self.kernel_size, self.stride, self.padding)

class CNNMNIST(nn.Module):
    """CNN sencilla para clasificación de dígitos MNIST (10 clases).
    Usa logits (sin softmax) para compatibilidad con CrossEntropyLoss.
    """
    def __init__(self, apply_softmax: bool = False, pool_type: str = "max"):
        super().__init__()
        self.apply_softmax = apply_softmax
        self.pool_type = pool_type.lower()
        def make_pool():
            return nn.MaxPool2d(2,2) if self.pool_type == "max" else MinPool2d(2,2)
        # Entrada: (B,1,28,28)
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)   # -> (32,28,28)
        self.pool1 = make_pool()                       # -> (32,14,14)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)  # -> (64,14,14)
        self.pool2 = make_pool()                       # -> (64,7,7)
        self.fc1 = nn.Linear(64*7*7, 128)
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(-1, 64*7*7)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        if self.apply_softmax:
            x = torch.softmax(x, dim=1)
        return x
