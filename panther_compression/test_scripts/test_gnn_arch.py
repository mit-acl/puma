#!/usr/bin/env python3

import torch as th
from torch_geometric.data import HeteroData
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils.convert import to_networkx

data = HeteroData()

# add nodes
data["current_state"].x = th.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]).double() # [number of "current_state" nodes, size of feature vector]
data["goal_state"].x = th.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]).double() # [number of "goal_state" nodes, size of feature vector]
data["observation"].x = th.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]).double() # [number of "observation" nodes, size of feature vector]

# add edges
data["current_state", "dist_state_to_goal", "goal_state"].edge_index = th.tensor([
                                                                            [0],  # idx of source nodes (current state)
                                                                            [0],  # idx of target nodes (goal state)
                                                                            ],dtype=th.int64)
data["current_state", "dist_state_to_observation", "observation"].edge_index = th.tensor([
                                                                            [0],  # idx of source nodes (current state)
                                                                            [0],  # idx of target nodes (observation)
                                                                            ],dtype=th.int64)
data["goal_state", "dist_goal_to_state", "current_state"].edge_index = th.tensor([
                                                                            [0],  # idx of source nodes (goal state)
                                                                            [0],  # idx of target nodes (current state)
                                                                            ],dtype=th.int64)
data["observation", "dist_observation_to_state", "current_state"].edge_index = th.tensor([
                                                                            [0],  # idx of source nodes (observation)
                                                                            [0],  # idx of target nodes (current state)
                                                                            ],dtype=th.int64)
data["observation", "dist_observation_to_goal", "goal_state"].edge_index = th.tensor([
                                                                            [0],  # idx of source nodes (observation)
                                                                            [0],  # idx of target nodes (goal state)
                                                                            ],dtype=th.int64)
data["goal_state", "dist_goal_to_observation", "observation"].edge_index = th.tensor([
                                                                            [0],  # idx of source nodes (goal state)
                                                                            [0],  # idx of target nodes (observation)
                                                                            ],dtype=th.int64)

# add edge weights
data["current_state", "dist_state_to_goal", "goal_state"].edge_attr = 1.0
data["current_state", "dist_state_to_observation", "observation"].edge_attr = 1.0
data["observation", "dist_observation_to_goal", "goal_state"].edge_attr = 1.0
# make it undirected
data["goal_state", "dist_goal_to_state", "current_state"].edge_attr = 1.0
data["observation", "dist_observation_to_state", "current_state"].edge_attr = 1.0
data["goal_state", "dist_goal_to_observation", "observation"].edge_attr = 1.0

# visualize the graph using networkx

G = to_networkx(data.to_homogeneous())
nx.draw(G, with_labels=True)
plt.show()

