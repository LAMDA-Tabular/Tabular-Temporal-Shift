import math
import typing as ty
import torch
import torch.nn as nn

from model.models.dcn2 import CrossLayer
from model.lib.temporal_embeddings import TemporalEmbeddings


class DCNv2_Temporal(nn.Module):
    def __init__(
        self,
        *,
        d_in: int,
        t_mean: float,
        t_std: float,
        d: int,
        n_hidden_layers: int,
        n_cross_layers: int,
        hidden_dropout: float,
        cross_dropout: float,
        d_out: int,
        stacked: bool,
        categories: ty.Optional[ty.List[int]],
        d_embedding: int,
        temporal_embeddings: ty.Optional[dict],
    ) -> None:
        super().__init__()

        self.temporal_embeddings = TemporalEmbeddings(t_mean, t_std, **temporal_embeddings)
        if categories is not None:
            d_in += len(categories) * d_embedding + self.temporal_embeddings.out_dim
            category_offsets = torch.tensor([0] + categories[:-1]).cumsum(0)
            self.register_buffer('category_offsets', category_offsets)
            self.category_embeddings = nn.Embedding(sum(categories), d_embedding)
            nn.init.kaiming_uniform_(self.category_embeddings.weight, a=math.sqrt(5))

        self.first_linear = nn.Linear(d_in, d)
        self.last_linear = nn.Linear(d if stacked else 2 * d, d_out)

        deep_layers = sum(
            [
                [nn.Linear(d, d), nn.ReLU(True), nn.Dropout(hidden_dropout)]
                for _ in range(n_hidden_layers)
            ],
            [],
        )
        cross_layers = [CrossLayer(d, cross_dropout) for _ in range(n_cross_layers)]

        self.deep_layers = nn.Sequential(*deep_layers)
        self.cross_layers = nn.ModuleList(cross_layers)
        self.stacked = stacked
        

    def forward(self, x_num, x_cat, idx):
        x = []
        if x_num is not None:
            x.append(x_num)
        if self.temporal_embeddings.out_dim:
            x.append(self.temporal_embeddings(idx).flatten(1))
        if x_cat is not None:
            x.append(
                self.category_embeddings(x_cat + self.category_offsets[None]).view(
                    x_cat.size(0), -1
                )
            )
        x = torch.cat(x, dim=-1)
        x = self.first_linear(x)
        x_cross = x
        for cross_layer in self.cross_layers:
            x_cross = cross_layer(x, x_cross)

        if self.stacked:
            return self.last_linear(self.deep_layers(x_cross)).squeeze(1)
        else:
            return self.last_linear(
                torch.cat([x_cross, self.deep_layers(x)], dim=1)
            ).squeeze(1)