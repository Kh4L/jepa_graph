"""
Loss functions for Graph JEPA training.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class JEPALoss(nn.Module):
    """Standard JEPA loss for graph representation learning."""
    
    def __init__(
        self,
        loss_type: str = "mse",
        normalize: bool = True,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.loss_type = loss_type
        self.normalize = normalize
        self.temperature = temperature
    
    def forward(
        self,
        predicted: Tensor,
        target: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        if self.normalize:
            predicted = F.normalize(predicted, dim=-1)
            target = F.normalize(target, dim=-1)
        
        if self.loss_type == "mse":
            loss = F.mse_loss(predicted, target, reduction="none")
            loss = loss.mean(dim=-1)
        elif self.loss_type == "cosine":
            cos_sim = F.cosine_similarity(predicted, target, dim=-1)
            loss = 1 - cos_sim
        elif self.loss_type == "smooth_l1":
            loss = F.smooth_l1_loss(predicted, target, reduction="none")
            loss = loss.mean(dim=-1)
        elif self.loss_type == "huber":
            loss = F.huber_loss(predicted, target, reduction="none")
            loss = loss.mean(dim=-1)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        if mask is not None:
            loss = loss * mask.float()
            loss = loss.sum() / mask.float().sum().clamp(min=1)
        else:
            loss = loss.mean()
        
        return loss / self.temperature


class VICRegLoss(nn.Module):
    """VICReg-style loss for preventing embedding collapse."""
    
    def __init__(
        self,
        sim_coeff: float = 25.0,
        var_coeff: float = 25.0,
        cov_coeff: float = 1.0,
        gamma: float = 1.0,
        epsilon: float = 1e-4,
    ):
        super().__init__()
        self.sim_coeff = sim_coeff
        self.var_coeff = var_coeff
        self.cov_coeff = cov_coeff
        self.gamma = gamma
        self.epsilon = epsilon
    
    def forward(
        self,
        predicted: Tensor,
        target: Tensor,
    ) -> Tuple[Tensor, dict]:
        if predicted.dim() == 3:
            predicted = predicted.reshape(-1, predicted.size(-1))
            target = target.reshape(-1, target.size(-1))
        
        N, D = predicted.shape
        
        sim_loss = F.mse_loss(predicted, target)
        
        pred_std = predicted.std(dim=0)
        target_std = target.std(dim=0)
        
        var_loss = (
            F.relu(self.gamma - pred_std).mean() +
            F.relu(self.gamma - target_std).mean()
        )
        
        pred_centered = predicted - predicted.mean(dim=0)
        target_centered = target - target.mean(dim=0)
        
        pred_cov = (pred_centered.T @ pred_centered) / (N - 1)
        target_cov = (target_centered.T @ target_centered) / (N - 1)
        
        pred_cov_loss = self._off_diagonal(pred_cov).pow(2).sum() / D
        target_cov_loss = self._off_diagonal(target_cov).pow(2).sum() / D
        cov_loss = pred_cov_loss + target_cov_loss
        
        total_loss = (
            self.sim_coeff * sim_loss +
            self.var_coeff * var_loss +
            self.cov_coeff * cov_loss
        )
        
        return total_loss, {
            "sim_loss": sim_loss.item(),
            "var_loss": var_loss.item(),
            "cov_loss": cov_loss.item(),
        }
    
    @staticmethod
    def _off_diagonal(matrix: Tensor) -> Tensor:
        n = matrix.size(0)
        return matrix.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class CombinedLoss(nn.Module):
    """Combines multiple loss functions with configurable weights."""
    
    def __init__(
        self,
        jepa_loss: Optional[JEPALoss] = None,
        vicreg_loss: Optional[VICRegLoss] = None,
        jepa_weight: float = 1.0,
        vicreg_weight: float = 0.0,
    ):
        super().__init__()
        
        self.jepa_loss = jepa_loss or JEPALoss()
        self.vicreg_loss = vicreg_loss
        
        self.jepa_weight = jepa_weight
        self.vicreg_weight = vicreg_weight
    
    def forward(
        self,
        predicted: Tensor,
        target: Tensor,
        **kwargs,
    ) -> Tuple[Tensor, dict]:
        losses = {}
        total_loss = 0.0
        
        if self.jepa_weight > 0:
            jepa = self.jepa_loss(predicted, target)
            losses["jepa"] = jepa.item()
            total_loss = total_loss + self.jepa_weight * jepa
        
        if self.vicreg_weight > 0 and self.vicreg_loss is not None:
            vicreg, vicreg_dict = self.vicreg_loss(predicted, target)
            losses["vicreg"] = vicreg.item()
            losses.update({f"vicreg_{k}": v for k, v in vicreg_dict.items()})
            total_loss = total_loss + self.vicreg_weight * vicreg
        
        return total_loss, losses
