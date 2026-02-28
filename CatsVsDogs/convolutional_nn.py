# We import the necessary modules from PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F

class MinPool2d(nn.Module):
    """Implementación de min-pooling usando max-pooling sobre el negativo."""
    def __init__(self, kernel_size=2, stride=2, padding=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
    def forward(self, x):
        return -F.max_pool2d(-x, self.kernel_size, self.stride, self.padding)

class CNN(nn.Module):
    """
    Red Convolucional simple para clasificación binaria (Cat vs Dog).

    Nota sobre la activación final:
    - Con `nn.BCEWithLogitsLoss` NO se aplica `sigmoid` en `forward`, porque la loss
        internamente hace la operación estable numéricamente.
    - Si en el futuro se cambiara a `nn.BCELoss` (que espera probabilidades en [0,1])
        o se quisiera obtener directamente probabilidades, se puede usar el parámetro
        `apply_sigmoid=True` al instanciar la red para que aplique `torch.sigmoid` sobre
        los logits antes de retornarlos.
    """

    def __init__(self, apply_sigmoid: bool = False, pool_type: str = "max"):
        """
        Constructor: Define capas y permite opcionalmente aplicar sigmoid final.
        :param apply_sigmoid: Si True, la salida del forward estará ya en [0,1].
                                                    Mantener False cuando se use BCEWithLogitsLoss.
        """
        super(CNN, self).__init__()
        self.apply_sigmoid = apply_sigmoid
        self.pool_type = pool_type.lower()

        def make_pool():
            return nn.MaxPool2d(kernel_size=2, stride=2) if self.pool_type == "max" else MinPool2d(kernel_size=2, stride=2)

        # --- Convolutional Block 1 ---
        # Input shape: (Batch_Size, 3, 150, 150)
        # 3 input channels (RGB), 32 output channels, 3x3 kernel, 1 padding
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        # Output after conv1: (Batch_Size, 32, 150, 150)
        self.pool1 = make_pool() # 2x2 pooling (max o min)
        # Output after pool1: (Batch_Size, 32, 75, 75)

        # --- Convolutional Block 2 ---
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        # Output after conv2: (Batch_Size, 64, 75, 75)
        self.pool2 = make_pool()
        # Output after pool2: (Batch_Size, 64, 37, 37)

        # --- Convolutional Block 3 ---
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        # Output after conv3: (Batch_Size, 128, 37, 37)
        self.pool3 = make_pool()
        # Output after pool3: (Batch_Size, 128, 18, 18) 
        # (37 / 2 = 18.5 -> rounds down to 18)

        # --- Fully-Connected (Classifier) Block ---
        # We must flatten the 128x18x18 feature map
        self.fc1 = nn.Linear(in_features=128 * 18 * 18, out_features=512)
        self.dropout = nn.Dropout(0.5) # Dropout layer to prevent overfitting
        self.fc2 = nn.Linear(in_features=512, out_features=1) # 1 output for binary (Cat vs Dog)

    def forward(self, x):
        """
        Defines the forward pass: how data flows through the layers.
        """
        # Pass through Conv Block 1
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        # Pass through Conv Block 2
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        # Pass through Conv Block 3
        x = self.conv3(x)
        x = F.relu(x)
        x = self.pool3(x)
        
        # Flatten the output for the linear layers
        # The -1 automatically calculates the batch size
        x = x.view(-1, 128 * 18 * 18) 
        
        # Pass through FC Block
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)  # Logits sin activación

        # Sigmoid solo si el usuario lo solicita (p.ej. para inferencia directa
        # o si cambiara la función de pérdida a BCELoss).
        if self.apply_sigmoid:
            x = torch.sigmoid(x)
        return x