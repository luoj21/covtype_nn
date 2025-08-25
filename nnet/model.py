import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicNN(nn.Module):
    
    def __init__(self, input_dim, num_classes):
        super(BasicNN, self).__init__()

        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 128)
        self.dropout1 = nn.Dropout(0.1)
        self.fc4 = nn.Linear(128, 32)
        self.dropout2 = nn.Dropout(0.2)
        self.out = nn.Linear(32, num_classes)


    def forward(self, X):
        """Forward Pass"""
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = F.relu(self.fc3(X))
        X = self.dropout1(X)
        X = F.relu(self.fc4(X))
        X = self.dropout2(X)
        X = self.out(X)  
        
        return X