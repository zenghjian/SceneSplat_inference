"""
Point Structure for Pointcept - Minimal Version
"""

import torch
import spconv.pytorch as spconv
from addict import Dict
from typing import List

# Use the existing serialization module
from scenesplat.serialization import encode
from . import (
    offset2batch,
    batch2offset,
    offset2bincount,
    bincount2offset,
)


class Point(Dict):
    """
    Point Structure of Pointcept - Minimal Version
    
    A Point (point cloud) in Pointcept is a dictionary that contains various properties of
    a batched point cloud. Essential properties:
    
    - "coord": original coordinate of point cloud;
    - "grid_coord": grid coordinate for specific grid size;
    - "feat": feature of point cloud, default input of model;
    - "batch": batch indices;
    - "offset": offset for batching;
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # If one of "offset" or "batch" do not exist, generate by the existing one
        if "batch" not in self.keys() and "offset" in self.keys():
            self["batch"] = offset2batch(self.offset)
        elif "offset" not in self.keys() and "batch" in self.keys():
            self["offset"] = batch2offset(self.batch)

    def serialization(self, order="z", depth=None, shuffle_orders=False):
        """
        Point Cloud Serialization
        
        relies on ["grid_coord" or "coord" + "grid_size", "batch", "feat"]
        """
        self["order"] = order
        assert "batch" in self.keys()
        
        if "grid_coord" not in self.keys():
            assert {"grid_size", "coord"}.issubset(self.keys())
            self["grid_coord"] = torch.div(
                self.coord - self.coord.min(0)[0], self.grid_size, rounding_mode="trunc"
            ).int()

        if depth is None:
            # Adaptive measure the depth of serialization cube (length = 2 ^ depth)
            depth = int(self.grid_coord.max() + 1).bit_length()
        self["serialized_depth"] = depth
        
        # Maximum bit length for serialization code is 63 (int64)
        assert depth * 3 + len(self.offset).bit_length() <= 63
        assert depth <= 16  # Depth limitation for point position

        # Handle multiple orders
        if isinstance(order, str):
            order = [order]
        
        self["serialized_code"] = []
        self["serialized_order"] = []
        self["serialized_inverse"] = []
        
        for o in order:
            code = encode(self.grid_coord, self.batch, depth, o)
            order_indices = torch.argsort(code)
            inverse_indices = torch.zeros_like(order_indices).scatter_(
                dim=0, index=order_indices, src=torch.arange(len(order_indices), device=order_indices.device)
            )
            
            self["serialized_code"].append(code)
            self["serialized_order"].append(order_indices)
            self["serialized_inverse"].append(inverse_indices)
        
        # Shuffle orders if requested
        if shuffle_orders and len(self["serialized_order"]) > 1:
            indices = torch.randperm(len(self["serialized_order"]))
            self["serialized_code"] = [self["serialized_code"][i] for i in indices]
            self["serialized_order"] = [self["serialized_order"][i] for i in indices]
            self["serialized_inverse"] = [self["serialized_inverse"][i] for i in indices]

    def sparsify(self):
        """
        Sparsify point cloud to SpConv format
        """
        device = self.coord.device
        if "sparse_shape" not in self.keys():
            sparse_shape = (self.grid_coord.max(0)[0] + 1).tolist()
            self["sparse_shape"] = sparse_shape
        
        if "sparse_conv_feat" not in self.keys():
            sparse_conv_feat = spconv.SparseConvTensor(
                features=self.feat,
                indices=torch.cat([self.batch.unsqueeze(1), self.grid_coord], dim=1),
                spatial_shape=self.sparse_shape,
                batch_size=self.batch.max().item() + 1,
            )
            self["sparse_conv_feat"] = sparse_conv_feat 