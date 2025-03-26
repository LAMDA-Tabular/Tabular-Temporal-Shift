import torch
import torch.nn as nn
from torch import Tensor


class FourierEmbeddings(nn.Module):
    def __init__(
        self, period: float, order: int
    ) -> None:
        super().__init__()
        self.frequencies = torch.arange(1, order + 1, dtype=torch.float32) / period
        self.frequencies = nn.Parameter(self.frequencies)

    def forward(self, x: Tensor) -> Tensor:
        assert x.ndim == 1
        x = 2 * torch.pi * self.frequencies[None] * x[..., None]
        x = torch.cat([torch.cos(x), torch.sin(x)], dim=-1)
        return x


class TemporalEmbeddings(nn.Module):
    def __init__(
        self,
        t_mean: float,
        t_std: float,
        order: list[int],
        trend: bool,
        d_embedding: int,
    ) -> None:
        super().__init__()
        self.order = order
        self.trend = trend
        self.periodicity = sum(order) > 0
        self.out_dim = (d_embedding if self.periodicity else 0) + (1 if self.trend else 0)
        
        if self.trend:
            self.t_mean = t_mean
            self.t_std = t_std
        
        if self.periodicity:
            assert len(order) == 4, "The length of orders must be 4, corresponding to (year, month, week, day)"
            self.embeddings = nn.ModuleList([                                   # period priors
                FourierEmbeddings(31557600.0, order[0]) if order[0] else None,  # year
                FourierEmbeddings(2629800.0, order[1]) if order[1] else None,   # month
                FourierEmbeddings(604800.0, order[2]) if order[2] else None,    # week
                FourierEmbeddings(86400.0, order[3]) if order[3] else None,     # day
            ])
            self.embeddings = nn.ModuleList([embedding for embedding in self.embeddings if embedding is not None])
            self.linear = nn.Linear(2 * sum(order), d_embedding)
            self.relu = nn.ReLU()

    def forward(self, x):
        x_trend = (x[..., None] - self.t_mean) / self.t_std if self.trend else None
        if self.periodicity:
            x = torch.cat([module(x) for module in self.embeddings], dim=-1)
            x = self.linear(x)
            x = self.relu(x)
            x = torch.cat([x, x_trend], dim=-1) if x_trend is not None else x
        else:
            assert x_trend is not None, "forwards() will not be called if self.out_dim == 0"
            x = x_trend
        return x


class TimestampNorm(nn.Module):
    def __init__(
        self,
        t_mean: float,
        t_std: float,
    ) -> None:
        super().__init__()
        
        self.t_mean = t_mean
        self.t_std = t_std

    def forward(self, x):
        x = (x - self.t_mean) / self.t_std
        return x


class PeriodicEmbeddings(nn.Module):
    """ The 'PeriodicEmbeddings' from 'On Embeddings for Numerical Features in Tabular Deep Learning'.
    """
    def __init__(
        self, n_frequencies: int, frequency_scale: float
    ) -> None:
        super().__init__()
        self.frequencies = nn.Parameter(
            torch.normal(0.0, frequency_scale, (1, n_frequencies))
        )

    def forward(self, x: Tensor) -> Tensor:
        assert x.ndim == 1
        x = 2 * torch.pi * self.frequencies * x[..., None]
        x = torch.cat([torch.cos(x), torch.sin(x)], dim=-1)
        return x


class TemporalEmbeddings_PLR(nn.Sequential):
    def __init__(
        self,
        t_mean: float,
        t_std: float,
        n_frequencies: int,
        frequency_scale: float,
        d_embedding: int,
    ) -> None:
        super().__init__(
            TimestampNorm(t_mean, t_std),
            PeriodicEmbeddings(n_frequencies, frequency_scale),
            nn.Linear(2 * n_frequencies, d_embedding),
            nn.ReLU(),
        )