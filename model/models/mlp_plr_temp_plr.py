import math
import typing as ty
from model.lib.tabr.utils import make_module
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from model.lib.temporal_embeddings import TemporalEmbeddings_PLR

# %%
class MLP_Temp_PLR(nn.Module):
    def __init__(
        self,
        *,
        d_in: int,
        d_num:int,
        d_out: int, 
        t_mean: float,
        t_std: float,
        d_layers: ty.List[int],    
        dropout: float,
        num_embeddings: Optional[dict],
        temporal_embeddings: Optional[dict],
        ) -> None:
        super().__init__()
        self.dropout = dropout
        self.d_out = d_out
        self.d_num = d_num + temporal_embeddings['d_embedding']
        self.d_in = d_in + temporal_embeddings['d_embedding'] if num_embeddings is None else self.d_num * num_embeddings['d_embedding'] + d_in - d_num
        self.layers = nn.ModuleList(
            [
                nn.Linear(d_layers[i - 1] if i else self.d_in, x)
                for i, x in enumerate(d_layers)
            ]
        )
        self.head = nn.Linear(d_layers[-1] if d_layers else self.d_in, self.d_out)
        self.num_embeddings = (
            None
            if num_embeddings is None
            else make_module(num_embeddings, n_features=self.d_num)
        )
        self.temporal_embeddings = TemporalEmbeddings_PLR(t_mean, t_std, **temporal_embeddings)

    def forward(self, x_num, x_cat, idx):
        idx = self.temporal_embeddings(idx).flatten(1)
        x_num = torch.cat([x_num, idx], dim=-1)
        if self.num_embeddings is not None and self.d_num > 0:
            x_num = self.num_embeddings(x_num).flatten(1)
        if x_num is not None and x_cat is not None:
            x = torch.cat([x_num, x_cat], dim=-1)
        elif x_num is not None:
            x = x_num
        elif x_cat is not None:
            x = x_cat
            x = x.to(torch.float32)
        for layer in self.layers:
            x = layer(x)
            x = F.relu(x)
            if self.dropout:
                x = F.dropout(x, self.dropout, self.training)
        logit = self.head(x)        
        if self.d_out == 1:
            logit = logit.squeeze(-1)
        return logit