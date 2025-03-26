import torch
import torch.nn as nn
import torch.nn.functional as F
from model.lib.tabr.utils import make_module
from typing import Optional

from model.models.modernNCA import Residual_block
from model.lib.temporal_embeddings import TemporalEmbeddings


class ModernNCA_Temporal(nn.Module):
    def __init__(
        self,
        *,
        d_in: int,
        d_num: int,
        d_out: int,
        t_mean: float,
        t_std: float,
        dim: int,
        dropout: float,
        d_block: int,
        n_blocks: int,
        num_embeddings: Optional[dict],
        temporal_embeddings: Optional[dict],
        temperature: float=1.0,
        sample_rate: float=0.8
        ) -> None:

        super().__init__()
        self.temporal_embeddings = TemporalEmbeddings(t_mean, t_std, **temporal_embeddings)
        self.d_out = d_out
        self.input_d_num = d_num
        self.d_num = d_num + self.temporal_embeddings.out_dim
        self.d_in = d_in + self.temporal_embeddings.out_dim if num_embeddings is None else self.d_num * num_embeddings['d_embedding'] + d_in - d_num
        self.dim = dim
        self.dropout = dropout
        self.d_block = d_block
        self.n_blocks = n_blocks
        self.T = temperature
        self.sample_rate = sample_rate
        if self.n_blocks > 0:
            self.post_encoder=nn.Sequential()
            for i in range(n_blocks):
                name = f"ResidualBlock{i}"
                self.post_encoder.add_module(name, self.make_layer())
            self.post_encoder.add_module('bn', nn.BatchNorm1d(self.dim))
        self.encoder = nn.Linear(self.d_in, self.dim)
        self.num_embeddings = (
            None
            if num_embeddings is None
            else make_module(num_embeddings, n_features=self.d_num)
        )

    def make_layer(self):
        block = Residual_block(self.dim, self.d_block, self.dropout)
        return block

    def forward(self, x, y, idx, candidate_x, candidate_y, candidate_idx, is_train):
        if is_train:
            data_size = candidate_x.shape[0]
            retrival_size = int(data_size * self.sample_rate)
            sample_idx = torch.randperm(data_size)[:retrival_size]
            candidate_x = candidate_x[sample_idx]
            candidate_y = candidate_y[sample_idx]
            candidate_idx = candidate_idx[sample_idx]
        
        x_num, x_cat = x[:, :self.input_d_num], x[:, self.input_d_num:]
        candidate_x_num, candidate_x_cat = candidate_x[:, :self.input_d_num], candidate_x[:, self.input_d_num:]
        
        if self.temporal_embeddings.out_dim:
            idx = self.temporal_embeddings(idx).flatten(1)
            candidate_idx = self.temporal_embeddings(candidate_idx).flatten(1)
            x_num = torch.cat([x_num, idx], dim=-1)
            candidate_x_num = torch.cat([candidate_x_num, candidate_idx], dim=-1)
        
        if self.num_embeddings is not None and self.d_num > 0:
            x_num = self.num_embeddings(x_num).flatten(1)
            candidate_x_num = self.num_embeddings(candidate_x_num).flatten(1)
        
        x = torch.cat([x_num, x_cat], dim=-1)
        candidate_x = torch.cat([candidate_x_num, candidate_x_cat], dim=-1)
        
        if self.n_blocks > 0:
            candidate_x = self.post_encoder(self.encoder(candidate_x))
            x = self.post_encoder(self.encoder(x))
        else:
            candidate_x = self.encoder(candidate_x)
            x = self.encoder(x)
        if is_train:
            assert y is not None
            candidate_x = torch.cat([x, candidate_x])
            candidate_y = torch.cat([y, candidate_y])
        else:
            assert y is None
        
        if self.d_out > 1:
            candidate_y = F.one_hot(candidate_y, self.d_out).float()
        elif len(candidate_y.shape) == 1:
            candidate_y=candidate_y.unsqueeze(-1).float()

        distances = torch.cdist(x, candidate_x, p=2)
        distances = distances / self.T
        
        if is_train:
            distances = distances.clone().fill_diagonal_(torch.inf)
        distances = F.softmax(-distances, dim=-1)
        logits = torch.mm(distances, candidate_y)
        if self.d_out > 1:
            eps = 1e-7
            logits = torch.log(logits + eps)
        return logits.squeeze(-1)