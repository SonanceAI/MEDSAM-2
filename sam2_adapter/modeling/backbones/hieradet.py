# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from sam2.modeling.backbones.hieradet import MultiScaleBlock as MultiScaleBlockOriginal
from sam2.modeling.backbones.hieradet import Hiera as HieraOriginal
from sam2.modeling.backbones.hieradet import do_pool
from sam2.modeling.backbones.utils import window_partition, window_unpartition


class Adapter(nn.Module):
    def __init__(self, input_dim: int, mlp_ratio=0.25, act_layer=nn.GELU, skip_connect=True):
        super().__init__()
        self.skip_connect = skip_connect
        num_hidden_features = int(input_dim * mlp_ratio)
        self.adapter_block = nn.Sequential(
            nn.Linear(input_dim, num_hidden_features),
            act_layer(),
            nn.Linear(num_hidden_features, input_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xs = self.adapter_block(x)
        if self.skip_connect:
            return x+xs
        return xs


class Adaptered(nn.Module):
    def __init__(self, orig_layer: nn.Module, input_dim: int):
        super().__init__()
        self.orig_layer = orig_layer
        self.adapter = Adapter(input_dim)

    def forward(self, *x):
        orig_out = self.orig_layer(*x)
        return self.adapter(orig_out)


class MultiScaleBlock(MultiScaleBlockOriginal):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mlp_adapter = Adapter(self.dim_out, skip_connect=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x  # B, H, W, C
        x = self.norm1(x)

        # Skip connection
        if self.dim != self.dim_out:
            shortcut = do_pool(self.proj(x), self.pool)

        # Window partition
        window_size = self.window_size
        if window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, window_size)

        # Window Attention + Q Pooling (if stage change)
        x = self.attn(x)
        if self.q_stride:
            # Shapes have changed due to Q pooling
            window_size = self.window_size // self.q_stride[0]
            H, W = shortcut.shape[1:3]

            pad_h = (window_size - H % window_size) % window_size
            pad_w = (window_size - W % window_size) % window_size
            pad_hw = (H + pad_h, W + pad_w)

        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, window_size, pad_hw, (H, W))

        x = shortcut + self.drop_path(x)
        # MLP
        xn = self.norm2(x)
        x = x + self.drop_path(self.mlp(xn))
        x = x + 0.5 * self.mlp_adapter(xn)

        return x


class Hiera(HieraOriginal):
    def __init__(
        self,
        embed_dim: int = 96,  # initial embed dim
        num_heads: int = 1,  # initial number of heads
        drop_path_rate: float = 0.0,  # stochastic depth
        q_pool: int = 3,  # number of q_pool stages
        q_stride: Tuple[int, int] = (2, 2),  # downsample stride bet. stages
        stages: Tuple[int, ...] = (2, 3, 16, 3),  # blocks per stage
        dim_mul: float = 2.0,  # dim_mul factor at stage shift
        head_mul: float = 2.0,  # head_mul factor at stage shift
        window_pos_embed_bkg_spatial_size: Tuple[int, int] = (14, 14),
        # window size per stage, when not using global att.
        window_spec: Tuple[int, ...] = (
            8,
            4,
            14,
            7,
        ),
        # global attn in these blocks
        global_att_blocks: Tuple[int, ...] = (
            12,
            16,
            20,
        ),
        return_interm_layers=True  # return feats from every stage
    ):
        super().__init__(embed_dim=embed_dim, num_heads=num_heads, drop_path_rate=drop_path_rate, q_pool=q_pool,
                         q_stride=q_stride, stages=stages, dim_mul=dim_mul, head_mul=head_mul,
                         window_pos_embed_bkg_spatial_size=window_pos_embed_bkg_spatial_size, window_spec=window_spec,
                         global_att_blocks=global_att_blocks, return_interm_layers=return_interm_layers)
        self.patch_adapter = Adapter(embed_dim)
        # add adapters to blocks
        for i, blk in enumerate(self.blocks):
            if blk.window_size == 0:
                self.blocks[i] = MultiScaleBlock(
                    dim=blk.dim,
                    dim_out=blk.dim_out,
                    num_heads=blk.attn.num_heads,
                    q_stride=blk.q_stride,
                    window_size=blk.window_size,
                )
                self.blocks[i].drop_path = blk.drop_path

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.patch_embed(x)
        x = self.patch_adapter(x)
        # x: (B, H, W, C)

        # Add pos embed
        x = x + self._get_pos_embed(x.shape[1:3])

        outputs = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if (i == self.stage_ends[-1]) or (
                i in self.stage_ends and self.return_interm_layers
            ):
                feats = x.permute(0, 3, 1, 2)
                outputs.append(feats)

        return outputs
