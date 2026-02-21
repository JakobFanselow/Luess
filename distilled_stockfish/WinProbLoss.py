import torch
import torch.nn as nn

class WinProbLoss(nn.Module):
    def __init__(self, scaling_factor=0.6):
        super().__init__()
        self.k = scaling_factor
        self.mse = nn.MSELoss()

    def forward(self, predictions, targets):
        pred_win_prob = torch.sigmoid(self.k * (predictions / 100.0))
        target_win_prob = torch.sigmoid(self.k * (targets / 100.0))
        
        return self.mse(pred_win_prob, target_win_prob)