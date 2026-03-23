import torch
import torch.nn as nn
import torch.nn.functional as F

# This is the EXACT class definition from your training notebook.
# This file tells Django what the "shape" of your model is.

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=38):
        super(SimpleCNN, self).__init__()
        # Input: 3 channels (RGB), 128x128 pixels

        # Conv Block 1
        # 128x128 -> 124x124 -> 62x62
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Conv Block 2
        # 62x62 -> 58x58 -> 29x29
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=0)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Conv Block 3
        # 29x29 -> 25x25 -> 12x12
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=0)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Calculate the flattened size
        # L1: (128 - 5)/1 + 1 = 124. Pool -> 124/2 = 62
        # L2: (62 - 5)/1 + 1 = 58.  Pool -> 58/2 = 29
        # L3: (29 - 5)/1 + 1 = 25.  Pool -> floor(25/2) = 12
        self.flat_features = 64 * 12 * 12

        # Fully Connected Layers
        self.fc1 = nn.Linear(in_features=self.flat_features, out_features=256)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=256, out_features=num_classes) # num_classes is 38

    def forward(self, x):
        # Conv Block 1
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        # Conv Block 2
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        # Conv Block 3
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        
        # Flatten
        x = x.view(-1, self.flat_features)
        
        # FC Layers
        x = self.fc1(x)
        x = self.relu4(x)
        x = self.fc2(x)
        
        return x
