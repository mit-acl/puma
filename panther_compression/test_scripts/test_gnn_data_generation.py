#!/usr/bin/env python3

import argparse
import os.path as osp

import torch
import torch.nn.functional as F
from torch.nn import ReLU
from tqdm import tqdm

import torch_geometric.transforms as T
from torch_geometric.datasets import OGB_MAG
from torch_geometric.loader import HGTLoader, NeighborLoader
from torch_geometric.nn import Linear, SAGEConv, Sequential, to_hetero

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

path = osp.join(osp.dirname(osp.realpath(__file__)), '../../data/OGB')
transform = T.ToUndirected(merge=True)
dataset = OGB_MAG(path, preprocess='metapath2vec', transform=transform)

# Already send node features/labels to GPU for faster access during sampling:
data = dataset[0].to(device, 'x', 'y')

# print data
print(data)