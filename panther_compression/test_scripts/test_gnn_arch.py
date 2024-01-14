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
data["observation"].x = th.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]).double() # [number of "observation" nodes, size of feature vector]

# add edges
data["current_state", "dist_state_to_goal", "goal_state"].edge_index = th.tensor([
                                                                            [0],  # idx of source nodes (current state)
                                                                            [0],  # idx of target nodes (goal state)
                                                                            ],dtype=th.int64)
data["goal_state", "dist_goal_to_state", "current_state"].edge_index = th.tensor([
                                                                            [0],  # idx of source nodes (goal state)
                                                                            [0],  # idx of target nodes (current state)
                                                                            ],dtype=th.int64)
data["current_state", "dist_state_to_observation", "observation"].edge_index = th.tensor([
                                                                            [0, 0],  # idx of source nodes (current state)
                                                                            [0, 1],  # idx of target nodes (observation)
                                                                            ],dtype=th.int64)
data["observation", "dist_observation_to_state", "current_state"].edge_index = th.tensor([
                                                                            [0, 0],  # idx of source nodes (observation)
                                                                            [0, 1],  # idx of target nodes (current state)
                                                                            ],dtype=th.int64)
data["observation", "dist_observation_to_goal", "goal_state"].edge_index = th.tensor([
                                                                            [0, 1],  # idx of source nodes (observation)
                                                                            [0, 0],  # idx of target nodes (goal state)
                                                                            ],dtype=th.int64)
data["goal_state", "dist_goal_to_observation", "observation"].edge_index = th.tensor([
                                                                            [0, 0],  # idx of source nodes (goal state)
                                                                            [0, 1],  # idx of target nodes (observation)
                                                                            ],dtype=th.int64)
data["observation", "dist_observation_to_observation", "observation"].edge_index = th.tensor([
                                                                            [0, 1],  # idx of source nodes (observation)
                                                                            [1, 0],  # idx of target nodes (observation)
                                                                            ],dtype=th.int64)

# add edge weights
data["current_state", "dist_state_to_goal", "goal_state"].edge_attr = th.tensor([1.0])
data["current_state", "dist_state_to_observation", "observation"].edge_attr = th.tensor([1.0, 2.0])
data["observation", "dist_observation_to_goal", "goal_state"].edge_attr = th.tensor([1.0, 2.0])
# make it undirected
data["goal_state", "dist_goal_to_state", "current_state"].edge_attr = th.tensor([1.0])
data["observation", "dist_observation_to_state", "current_state"].edge_attr = th.tensor([2.0, 1.0])
data["goal_state", "dist_goal_to_observation", "observation"].edge_attr = th.tensor([2.0, 1.0])
data["observation", "dist_observation_to_observation", "observation"].edge_attr = th.tensor([1.0, 1.0])

# print the graph
print(data)

# visualize the graph using networkx
G = to_networkx(data.to_homogeneous())

# add labels
labels = {}
for i in range(data["current_state"].x.shape[0]):
    labels[i] = "current_state_" + str(i)
for i in range(data["goal_state"].x.shape[0]):
    labels[i + data["current_state"].x.shape[0]] = "goal_state_" + str(i)
for i in range(data["observation"].x.shape[0]):
    labels[i + data["current_state"].x.shape[0] + data["goal_state"].x.shape[0]] = "observation_" + str(i)

# draw the graph
plt.figure(figsize=(10, 10))
pos = nx.spring_layout(G)
nx.draw(G, pos, labels=labels, with_labels=True, font_weight='bold')
plt.show()
