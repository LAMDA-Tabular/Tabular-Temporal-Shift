import torch
import torch.nn as nn
import torch.nn.functional as F
import typing as ty
from typing import Optional
from torch import Tensor
import math
import delu
import faiss
import faiss.contrib.torch_utils

from model.lib.tabr.utils import make_module
from model.lib.temporal_embeddings import TemporalEmbeddings

class TabR_Temporal(nn.Module):
    def __init__(
        self,
        *,
        #
        n_num_features: int,
        n_cat_features: int,
        n_classes: Optional[int],
        t_mean: float,
        t_std: float,
        #
        num_embeddings: Optional[dict],
        temporal_embeddings: Optional[dict],
        d_main: int,
        d_multiplier: float,
        encoder_n_blocks: int,
        predictor_n_blocks: int,
        mixer_normalization,
        context_dropout: float,
        dropout0: float,
        dropout1,
        normalization: str,
        activation: str,
        #
        # The following options should be used only when truly needed.
        memory_efficient: bool = True,
        candidate_encoding_batch_size: Optional[int] = 4096,
    ) -> None:
        if not memory_efficient:
            assert candidate_encoding_batch_size is None
        if mixer_normalization == 'auto':
            mixer_normalization = encoder_n_blocks > 0
        if encoder_n_blocks == 0:
            assert not mixer_normalization
        
        super().__init__()
        self.temporal_embeddings = TemporalEmbeddings(t_mean, t_std, **temporal_embeddings)
        if dropout1 == 'dropout0':
            dropout1 = dropout0
        n_num_features += self.temporal_embeddings.out_dim
        self.n_classes = n_classes

        self.num_embeddings = (
            None
            if num_embeddings is None
            else make_module(num_embeddings, n_features=n_num_features)
        )

        # >>> E
        d_in = (
            n_num_features
            * (1 if num_embeddings is None else num_embeddings['d_embedding'])
            + n_cat_features
        )
        d_block = int(d_main * d_multiplier)
        Normalization = getattr(nn, normalization)
        Activation = getattr(nn, activation)

        def make_block(prenorm: bool) -> nn.Sequential:
            return nn.Sequential(
                *([Normalization(d_main)] if prenorm else []),
                nn.Linear(d_main, d_block),
                Activation(),
                nn.Dropout(dropout0),
                nn.Linear(d_block, d_main),
                nn.Dropout(dropout1),
            )

        self.linear = nn.Linear(d_in, d_main)
        self.blocks0 = nn.ModuleList(
            [make_block(i > 0) for i in range(encoder_n_blocks)]
        )

        # >>> R
        self.normalization = Normalization(d_main) if mixer_normalization else None
        self.label_encoder = (
            nn.Linear(1, d_main)
            if n_classes == 1
            else nn.Sequential(
                nn.Embedding(n_classes, d_main), delu.nn.Lambda(lambda x: x.squeeze(-2))
            )
        )
        self.K = nn.Linear(d_main, d_main)
        self.T = nn.Sequential(
            nn.Linear(d_main, d_block),
            Activation(),
            nn.Dropout(dropout0),
            nn.Linear(d_block, d_main, bias=False),
        )
        self.dropout = nn.Dropout(context_dropout)

        # >>> P
        self.blocks1 = nn.ModuleList(
            [make_block(True) for _ in range(predictor_n_blocks)]
        )
        self.head = nn.Sequential(
            Normalization(d_main),
            Activation(),
            nn.Linear(d_main, n_classes),
        )

        # >>>
        self.search_index = None
        self.memory_efficient = memory_efficient
        self.candidate_encoding_batch_size = candidate_encoding_batch_size
        self.reset_parameters()

    def reset_parameters(self):
        if isinstance(self.label_encoder, nn.Linear):
            bound = 1 / math.sqrt(2.0)
            nn.init.uniform_(self.label_encoder.weight, -bound, bound)  # type: ignore[code]  # noqa: E501
            nn.init.uniform_(self.label_encoder.bias, -bound, bound)  # type: ignore[code]  # noqa: E501
        else:
            assert isinstance(self.label_encoder[0], nn.Embedding)
            nn.init.uniform_(self.label_encoder[0].weight, -1.0, 1.0)  # type: ignore[code]  # noqa: E501

    def _encode(self, x_num, x_cat, idx) :
        x = []
        if self.temporal_embeddings.out_dim:
            idx = self.temporal_embeddings(idx).flatten(1)
            x_num = torch.cat([x_num, idx], dim=-1)
        if x_num is None:
            self.num_embeddings = None
        else:
            x.append(
                x_num
                if self.num_embeddings is None
                else self.num_embeddings(x_num).flatten(1)
            )
        if x_cat is not None:
            x.append(x_cat)
        x = torch.cat(x, dim=1)

        x = x.float()
        x = self.linear(x)
        for block in self.blocks0:
            x = x + block(x)
        k = self.K(x if self.normalization is None else self.normalization(x))
        
        return x, k

    def forward(
        self,
        *,
        x_num: Tensor, 
        x_cat: ty.Optional[Tensor],
        y: Optional[Tensor],
        idx: Tensor,
        candidate_x_num: ty.Optional[Tensor],
        candidate_x_cat: ty.Optional[Tensor],
        candidate_y: Tensor,
        candidate_idx: Tensor,
        context_size: int,
        is_train: bool,
    ) -> Tensor:
        # >>>
        with torch.set_grad_enabled(
            torch.is_grad_enabled() and not self.memory_efficient
        ):
            candidate_k = (
                self._encode(candidate_x_num, candidate_x_cat, candidate_idx)[1]
                if self.candidate_encoding_batch_size is None
                else torch.cat(
                    [
                        self._encode(x_num_, x_cat_, idx_)[1]
                        for x_num_, x_cat_, idx_ in delu.iter_batches(
                            (candidate_x_num, candidate_x_cat, candidate_idx), self.candidate_encoding_batch_size
                        )
                    ]
                )
            )
        x, k = self._encode(x_num, x_cat, idx)
        if is_train:
            assert y is not None
            candidate_k = torch.cat([k, candidate_k])
            candidate_y = torch.cat([y, candidate_y])
        else:
            assert y is None
        
        batch_size, d_main = k.shape
        device = k.device
        with torch.no_grad():
            if self.search_index is None:
                self.search_index = (
                    faiss.GpuIndexFlatL2(faiss.StandardGpuResources(), d_main)
                    if device.type == 'cuda'
                    else faiss.IndexFlatL2(d_main)
                )
            
            self.search_index.reset()
            self.search_index.add(candidate_k.to(torch.float32))  # type: ignore[code]
            distances: Tensor
            context_idx: Tensor
            distances, context_idx = self.search_index.search(  # type: ignore[code]
                k.to(torch.float32), context_size + (1 if is_train else 0)
            )
            if is_train:
                distances[
                    context_idx == torch.arange(batch_size, device=device)[:, None]
                ] = torch.inf
                # Not the most elegant solution to remove the argmax, but anyway.
                context_idx = context_idx.gather(-1, distances.argsort()[:, :-1])
        if self.memory_efficient and torch.is_grad_enabled():
            assert is_train
            # Repeating the same computation,
            # but now only for the context objects and with autograd on.
            context_k = self._encode(
                torch.cat([x_num, candidate_x_num])[
                        context_idx
                    ].flatten(0, 1),
                torch.cat([x_cat, candidate_x_cat])[
                        context_idx
                    ].flatten(0, 1),
                torch.cat([idx, candidate_idx])[
                        context_idx
                    ].flatten(0, 1)
            )[1].reshape(batch_size, context_size, -1)
        else:
            context_k = candidate_k[context_idx]

        similarities = (
            -k.square().sum(-1, keepdim=True)
            + (2 * (k[..., None, :] @ context_k.transpose(-1, -2))).squeeze(-2)
            - context_k.square().sum(-1)
        )
        probs = F.softmax(similarities, dim=-1)
        probs = self.dropout(probs)
        
        if self.n_classes > 1:
            context_y_emb = self.label_encoder(candidate_y[context_idx][..., None].long())
        else:
            context_y_emb = self.label_encoder(candidate_y[context_idx][..., None])
            if len(context_y_emb.shape) == 4:
                context_y_emb = context_y_emb[:,:,0,:]
        values = context_y_emb + self.T(k[:, None] - context_k)
        context_x = (probs[:, None] @ values).squeeze(1)
        x = x + context_x

        # >>>
        for block in self.blocks1:
            x = x + block(x)
        x = self.head(x)
        return x