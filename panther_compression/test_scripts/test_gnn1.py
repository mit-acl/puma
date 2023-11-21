# Udemy class Section 1: workshop 1

#!/usr/bin/env python3

import torch
from torch_geometric.data import Data
import networkx as nx
from torch_geometric.utils.convert import to_networkx
import matplotlib.pyplot as plt

" Define a graph "

# a graph with 4 nodes
edge_list = torch.tensor([
                         [0, 0, 0, 1, 2, 2, 3, 3],  # source nodes
                         [1, 2, 3, 0, 0, 3, 2, 0],  # target nodes
                         ],dtype=torch.long)

# 6 Features for each node (4x6 - # of nodes x # of features)
node_features = torch.tensor([
                             [1, 3, 4, 5, 2, 1], # Features of Node 0
                             [1, 3, 3, 6, 4, 3], # Features of Node 1
                             [2, 6, 3, 6, 3, 4], # Features of Node 2
                             [1, 8, 1, 6, 1, 1], # Features of Node 3
                             ], dtype=torch.long)

# 1 weight for each edge
edge_weight = torch.tensor([
                           [35.], # weight for nodes (0,1)
                           [48.], # weight for nodes (0,2)
                           [12.], # weight for nodes (0,3)
                           [10.], # weight for nodes (1,0)
                           [70.], # weight for nodes (2,0)
                           [15.], # weight for nodes (2,3)
                           [8.], # weight for nodes (3,2)
                           [6.], # weight for nodes (3,0)
                           ], dtype=torch.float)

# make a data object to store graph information
data = Data(x=node_features, edge_index=edge_list, edge_attr=edge_weight)

" Print the graph info "
print("number of nodes: ", data.num_nodes)
print("number of edges: ", data.num_edges)
print("number of features per node: ", data.num_node_features)
print("number of weights per edge: ", data.num_edge_features)

" Plot the graph "

G = to_networkx(data)
nx.draw(G, with_labels=True)
plt.show()