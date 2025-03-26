import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Literal
from torch import Tensor
import delu
from model.lib.tabm.tabm import _init_scaling_by_sections
from model.lib.tabm.deep import ElementwiseAffineEnsemble, make_efficient_ensemble, OneHotEncoding0d
from model.lib.tabr.utils import make_module, make_module1, MLP # , ResNet
from model.models.tabm import _get_first_input_scaling
from model.lib.temporal_embeddings import TemporalEmbeddings
    
class TabM_Temporal(nn.Module):
    def __init__(
        self,
        *,
        n_num_features: int,
        cat_cardinalities: list[int],
        n_classes: None | int,
        t_mean: float,
        t_std: float,
        # bins: None | list[Tensor],
        backbone: dict,
        num_embeddings: None | dict = None,
        temporal_embeddings: None | dict = None,
        arch_type: Literal[
            # Active
            'vanilla', # Simple MLP
            'tabm',  # BatchEnsemble + separate heads + better initialization
            'tabm-mini',  # Minimal: * weight
            # BatchEnsemble
            'tabm-naive'
        ],
        k: None | int = None,
    ) -> None:
        # >>> Validate arguments.
        assert n_num_features >= 0
        assert n_num_features or cat_cardinalities
        if arch_type == 'vanilla':
            assert k is None
        else:
            assert k is not None
            assert k > 0
        if cat_cardinalities is None:
            cat_cardinalities = []
        super().__init__()

        # >>> Continuous (numerical) features
        scaling_init_sections = []
        self.temporal_embeddings = TemporalEmbeddings(t_mean, t_std, **temporal_embeddings)
        n_num_features += self.temporal_embeddings.out_dim

        if num_embeddings is None:
            self.num_module = None
            d_num = n_num_features
            scaling_init_sections.extend(1 for _ in range(n_num_features))

        else:
            self.num_module = make_module(
                num_embeddings, n_features=n_num_features
            )
            d_num = n_num_features * num_embeddings['d_embedding']
            scaling_init_sections.extend(
                num_embeddings['d_embedding'] for _ in range(n_num_features)
            )

        # >>> Categorical features
        self.cat_module = (
            OneHotEncoding0d(cat_cardinalities) if cat_cardinalities else None
        )
        scaling_init_sections.extend(cat_cardinalities)
        d_cat = sum(cat_cardinalities)

        # >>> Backbone
        d_flat = d_num + d_cat
        self.affine_ensemble = None
        self.backbone = make_module1(d_in=d_flat,**backbone)

        if arch_type != 'vanilla':
            assert k is not None
            scaling_init = (
                'random-signs'
                if num_embeddings is None
                else 'normal'
            )

            if arch_type == 'tabm-mini':
                # The minimal possible efficient ensemble.
                self.affine_ensemble = ElementwiseAffineEnsemble(
                    k,
                    d_flat,
                    weight=True,
                    bias=False,
                    weight_init=(
                        'random-signs'
                        if num_embeddings is None
                        else 'normal'
                    ),
                )
                _init_scaling_by_sections(
                    self.affine_ensemble.weight,  # type: ignore[code]
                    scaling_init,
                    scaling_init_sections,
                )

            elif arch_type == 'tabm-naive':
                # The original BatchEnsemble.
                make_efficient_ensemble(
                    self.backbone,
                    k=k,
                    ensemble_scaling_in=True,
                    ensemble_scaling_out=True,
                    ensemble_bias=True,
                    scaling_init='random-signs',
                )
            elif arch_type == 'tabm':
                # Like BatchEnsemble, but all scalings, except for the first one,
                # are initialized with ones.
                make_efficient_ensemble(
                    self.backbone,
                    k=k,
                    ensemble_scaling_in=True,
                    ensemble_scaling_out=True,
                    ensemble_bias=True,
                    scaling_init='ones',
                )
                _init_scaling_by_sections(
                    _get_first_input_scaling(self.backbone).r,  # type: ignore[code]
                    scaling_init,
                    scaling_init_sections,
                )

            else:
                raise ValueError(f'Unknown arch_type: {arch_type}')

        # >>> Output
        d_block = backbone['d_layers'][-1]
        self.d_out = 1 if n_classes is None else n_classes
        self.output = (
            nn.Linear(d_block, self.d_out)
            if arch_type == 'vanilla'
            else delu.nn.NLinear(k, d_block, self.d_out)  # type: ignore[code]
        )

        # >>>
        self.arch_type = arch_type
        self.k = k

    def forward(
        self, x_num: None | Tensor = None, x_cat: None | Tensor = None, idx: Tensor = None
    ) -> Tensor:
        x = []
        if self.temporal_embeddings.out_dim:
            idx = self.temporal_embeddings(idx).flatten(1)
            x_num = torch.cat([x_num, idx], dim=-1)
        if x_num is not None:
            x.append(x_num if self.num_module is None else self.num_module(x_num))
        if x_cat is None:
            assert self.cat_module is None
        else:
            assert self.cat_module is not None
            x.append(self.cat_module(x_cat))
        x = torch.column_stack([x_.flatten(1, -1) for x_ in x])
        if x.dtype == torch.int64:
            x = x.float()

        if self.k is not None:
            x = x[:, None].expand(-1, self.k, -1)  # (B, D) -> (B, K, D)
            if self.affine_ensemble is not None:
                x = self.affine_ensemble(x)
        else:
            assert self.affine_ensemble is None

        x = self.backbone(x)
        x = self.output(x)
        if self.k is None:
            x = x[:, None]
        if self.d_out == 1:
            x = x.squeeze(-1)
        return x