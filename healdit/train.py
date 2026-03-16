from __future__ import annotations

from tqdm import tqdm
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
from hydra.utils import instantiate

if TYPE_CHECKING:
    from typing import Dict, List, Optional, Any, Union

    import torch.optim as optim
    from torch.utils.data import DataLoader

    from healdit.schemas.config import TrainParams


def train(
        model: nn.Module, 
        loader: DataLoader, 
        params: TrainParams
    ) -> Dict[str, List[float]]:
    
    optim = params.optimiser(params=model.parameters())
    
    history: Dict[str, List[Any]] = {
        "grad_norm": [],
        "total_kl": [],
        "kl_levels": [],
        "recon_loss": [],
        "elbo": [],
        "skip_occurred": []
    }

    optim.zero_grad()

    total_update_attempts = 0
    total_skips = 0
    
    for epoch in range(params.epochs):
        print(f"\n--- Epoch {epoch+1}/{params.epochs} ---")
        for step, x in enumerate(tqdm(loader)):
            x = x.to(params.device)
            
            _, elbo, metrics = vae_loss(x, model, params.criterion)
            
            (elbo / params.accumulation_steps).backward()
            is_update_step = not ((step + 1) % params.accumulation_steps) or (step + 1) == len(loader)
            
            if is_update_step:
                total_update_attempts += 1

                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 
                    max_norm=params.max_norm
                ).item()

                if params.gradient_threshold is not None and grad_norm >= params.gradient_threshold:
                    total_skips += 1
                    history["skip_occurred"].append(1)
                else:
                    optim.step()
                    history["skip_occurred"].append(0)
                
                optim.zero_grad()
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 
                    max_norm=float('inf')
                ).item()
            
            history["grad_norm"].append(grad_norm)
            history["total_kl"].append(metrics["total_kl"])
            history["kl_levels"].append(metrics["kl_levels"]) 
            history["recon_loss"].append(metrics["recon_loss"])
            history["elbo"].append(elbo.item())
            
            if torch.isnan(elbo) or np.isnan(grad_norm):
                print(f"\nCRITICAL: NaN detected at Epoch {epoch+1}, Step {step}.")
                return history 

    skip_pct = (total_skips / total_update_attempts) if total_update_attempts > 0 else 0
    history["skip_percentage"] = skip_pct

    return history

def vae_loss(
        x: torch.Tensor, 
        model: nn.Module, 
        crit: nn.Module, 
        beta: float = 10.0
    ) -> Dict:
    y, decoder_kl = model(x)
    
    rl = crit(x, y).mean(dim=(1, 2)) 
    
    kl_per_level = []
    rpp = torch.zeros_like(rl)
    
    for kl_list in decoder_kl:
        level_sum = torch.stack([k.sum(dim=list(range(1, k.dim()))) for k in kl_list]).sum(dim=0)
        level_sum /= np.prod(x.shape[1:])
        rpp += level_sum
        kl_per_level.append(level_sum.mean().item())
        
    elbo = (rpp + rl * beta).mean()
    
    metrics = {
        "recon_loss": rl.mean().item(),
        "total_kl": rpp.mean().item(),
        "kl_levels": kl_per_level 
    }
    return y, elbo, metrics

