import math
import typing as ty

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from model.lib.temporal_embeddings import TemporalEmbeddings


class MLP_Temporal(nn.Module):
    def __init__(
        self,
        *,
        d_in: int,
        d_out: int, 
        t_mean: float,
        t_std: float,
        d_layers: ty.List[int],    
        dropout: float,
        temporal_embeddings: Optional[dict],
        ) -> None:
        super().__init__()
        self.dropout = dropout
        self.temporal_embeddings = TemporalEmbeddings(t_mean, t_std, **temporal_embeddings)
        self.d_out = d_out
        self.d_in = d_in + self.temporal_embeddings.out_dim
        self.layers = nn.ModuleList(
            [
                nn.Linear(d_layers[i - 1] if i else self.d_in, x)
                for i, x in enumerate(d_layers)
            ]
        )
        self.head = nn.Linear(d_layers[-1] if d_layers else self.d_in, self.d_out)


    def forward(self, x, x_cat, idx):
        if self.temporal_embeddings.out_dim:
            idx = self.temporal_embeddings(idx).flatten(1)
            x = torch.cat([x, idx], dim=-1)
        for layer in self.layers:
            x = layer(x)
            x = F.relu(x)
            if self.dropout:
                x = F.dropout(x, self.dropout, self.training)
        logit = self.head(x)        
        if self.d_out == 1:
            logit = logit.squeeze(-1)
        return logit