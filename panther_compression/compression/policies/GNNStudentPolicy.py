from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import gym
import torch as th
import time
from statistics import mean
from torch import nn
import numpy as np
from stable_baselines3.common.distributions import SquashedDiagGaussianDistribution
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    FlattenExtractor,
    create_mlp,
)

from compression.utils.other import ActionManager, ObservationManager, ObstaclesManager, getPANTHERparamsAsCppStruct, readPANTHERparams
from colorama import init, Fore, Back, Style

from torch_geometric.data import Data, HeteroData
from torch_geometric.nn import GATConv, to_hetero, HeteroConv, Linear, HGTConv
import networkx as nx
from torch_geometric.utils.convert import to_networkx
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader

import warnings

# CAP the standard deviation of the actor
LOG_STD_MAX = 20
LOG_STD_MIN = -20

def create_empty_gnn_data():
    
        """
        This function creates an empty dataset for GNN
        """
    
        data = HeteroData()
    
        # add nodes
        data["current_state"].x = th.tensor([], dtype=th.double)
        data["goal_state"].x = th.tensor([], dtype=th.double)
        data["observation"].x = th.tensor([], dtype=th.double)
    
        # add edges
        data["current_state", "dist_to_goal_state", "goal_state"].edge_index = th.tensor([
                                                                                        [],  # idx of source nodes (current state)
                                                                                        [],  # idx of target nodes (goal state)
                                                                                        ],dtype=th.int64)
        data["current_state", "dist_to_observation", "observation"].edge_index = th.tensor([
                                                                                        [],  # idx of source nodes (current state)
                                                                                        [],  # idx of target nodes (observation)
                                                                                        ],dtype=th.int64)
        data["goal_state", "dist_to_goal_state", "current_state"].edge_index = th.tensor([
                                                                                        [],  # idx of source nodes (goal state)
                                                                                        [],  # idx of target nodes (current state)
                                                                                        ],dtype=th.int64)
        data["observation", "dist_to_observation", "current_state"].edge_index = th.tensor([
                                                                                        [],  # idx of source nodes (observation)
                                                                                        [],  # idx of target nodes (current state)
                                                                                        ],dtype=th.int64)
    
        # add edge weights
        data["current_state", "dist_to_goal_state", "goal_state"].edge_attr = th.tensor([], dtype=th.double)
        data["current_state", "dist_to_observation", "observation"].edge_attr = th.tensor([], dtype=th.double)
        data["goal_state", "dist_to_goal_state", "current_state"].edge_attr = th.tensor([], dtype=th.double)
        data["observation", "dist_to_observation", "current_state"].edge_attr = th.tensor([], dtype=th.double)
    
        return data

class GNNStudentPolicy(BasePolicy):
    """
    Actor network (policy) for Dagger, taken from SAC of stable baselines3.
    https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/sac/policies.py#L26

    :param observation_space: Obervation space
    :param action_space: Action space
    :param net_arch: Network architecture
    :param features_extractor: Network to extract features (a CNN when using images, a nn.Flatten() layer otherwise)
    :param features_dim: Number of features
    :param activation_fn: Activation function
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        gnn_data: HeteroData,
        gnn_hidden_channels: int,
        gnn_num_layers: int,
        gnn_num_heads: int,
        num_linear_layers: int,
        linear_hidden_channels: int,
        out_channels: int,
        num_of_trajs_per_replan: int,
        net_arch: [List[int]] = [64, 64],
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        if optimizer_kwargs is None:
            optimizer_kwargs = {}
            # Small values to avoid NaN in Adam optimizer
            if optimizer_class == th.optim.Adam:
                optimizer_kwargs["eps"] = 1e-5

        super(GNNStudentPolicy, self).__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor = features_extractor_class(observation_space),
            features_extractor_kwargs = features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=False,
        )

        " **** Save arguments to re-create object at loading time **** "

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.name=Style.BRIGHT+Fore.WHITE+"  [Stu]"+Style.RESET_ALL
        self.om=ObservationManager()
        self.am=ActionManager()
        self.obsm=ObstaclesManager()
        par = getPANTHERparamsAsCppStruct()
        self.features_dim=self.om.getObservationSize()
        self.agent_input_dim = self.om.getAgentObservationSize()
        self.lstm_each_obstacle_dim = self.obsm.getSizeEachObstacle()
        self.use_bn = par.use_bn
        self.features_extractor_class = features_extractor_class
        self.use_num_obses = True

        " **** GNN related params **** "

        self.gnn_data = gnn_data
        self.gnn_hidden_channels = gnn_hidden_channels
        self.gnn_num_layers = gnn_num_layers
        self.gnn_num_heads = gnn_num_heads
        self.num_linear_layers = num_linear_layers
        self.linear_hidden_channels = linear_hidden_channels
        self.out_channels = out_channels
        self.num_of_trajs_per_replan = num_of_trajs_per_replan

        " **** GNN **** "

        self.lin_dict = th.nn.ModuleDict()
        for node_type in self.gnn_data.node_types:
            self.lin_dict[node_type] = Linear(-1, self.gnn_hidden_channels)

        self.convs = th.nn.ModuleList()
        for _ in range(self.gnn_num_layers):
            conv = HGTConv(self.gnn_hidden_channels, self.gnn_hidden_channels, self.gnn_data.metadata(), self.gnn_num_heads, group='sum')
            self.convs.append(conv)

        # add linear layers (num_linear_layers) times
        self.lins = th.nn.ModuleList()
        for _ in range(num_linear_layers-1):
            self.lins.append(Linear(-1, linear_hidden_channels)) 
        self.lins.append(Linear(linear_hidden_channels, out_channels*num_of_trajs_per_replan)) 

        # Normalize the output
        self.tanh = nn.Tanh()

        self.double() # convert all the parameters to doublew

    # def _get_data(self) -> Dict[str, Any]:
    #     data = super()._get_data()

    #     data.update(
    #         dict(
    #             net_arch=self.net_arch,
    #             features_dim=self.features_dim,
    #             activation_fn=self.activation_fn,
    #             features_extractor=self.features_extractor,
    #         )
    #     )
    #     return data

    " **** GNN forward function **** "

    def forward(self, x_dict, edge_index_dict):
        
        for node_type, x in x_dict.items():
            x_dict[node_type] = self.lin_dict[node_type](x).relu_()

        for conv in self.convs:

            x_dict = conv(x_dict, edge_index_dict)
    
        # extract the global embedding
        latent = th.cat([x for _, x in sorted(x_dict.items())], dim=-1)

        # add linear layers
        for lin in self.lins:
            latent = lin(latent)
        
        before_shape = latent.shape
        # reshape latent to be [num_of_trajs_per_replan, out_channels]
        output = th.reshape(latent, (before_shape[0],) + (self.num_of_trajs_per_replan, self.out_channels))

        # normalize the output
        output = self.tanh(output)

        return output

    def _predict(self, obs_n:th.Tensor, deterministic=True) -> th.Tensor:

        data = self.generate_data(obs_n, self.device)
        action = self.forward(data.x_dict, data.edge_index_dict)
        self.am.assertActionIsNormalized(action.cpu().numpy().reshape(self.am.getActionShape()), self.name)
        return action

    def predictSeveral(self, obs_n):
                       
        acts=[]
        for i in range(len(obs_n)):
            self.i_index = i
            acts.append(self.predict(obs_n[i,:], deterministic=True)[0].reshape(self.am.getActionShape()))
        acts=np.stack(acts, axis=0)
        return acts

    def generate_data(self, f_obs_n, device):

        """
        This function generates a dataset for GNN
        """

        " ********************* GET NODES ********************* "

        # nodes you need for GNN
        # 0. current state
        # 1. goal state
        # 2. observation
        # In the current setting f_obs is a realative state from the current state so we pass f_v, f_z, yaw_dot to the current state node

        if type(f_obs_n) is np.ndarray:
            warnings.warn("f_obs_n is a numpy array - converting it to a torch tensor")
            f_obs_n = th.tensor(f_obs_n, dtype=th.double).to(device)

        feature_vector_for_current_state = f_obs_n[0,0,0:7].clone().detach().unsqueeze(0).to('cpu')
        feature_vector_for_goal = f_obs_n[0,0,7:10].clone().detach().unsqueeze(0).to('cpu')
        feature_vector_for_obs = f_obs_n[0,0,10:].clone().detach().unsqueeze(0).to('cpu')
        dist_current_state_goal = th.tensor([np.linalg.norm(feature_vector_for_goal[:3])], dtype=th.double).to(device)
        dist_current_state_obs = th.tensor([np.linalg.norm(feature_vector_for_obs[:3])], dtype=th.double).to(device)

        " ********************* MAKE A DATA OBJECT FOR HETEROGENEUS GRAPH ********************* "

        data = HeteroData()

        # add nodes
        data["current_state"].x = feature_vector_for_current_state.double() # [number of "current_state" nodes, size of feature vector]
        data["goal_state"].x = feature_vector_for_goal.double() # [number of "goal_state" nodes, size of feature vector]
        data["observation"].x = feature_vector_for_obs.double() # [number of "observation" nodes, size of feature vector]

        # add edges
        data["current_state", "dist_to_goal_state", "goal_state"].edge_index = th.tensor([
                                                                                        [0],  # idx of source nodes (current state)
                                                                                        [0],  # idx of target nodes (goal state)
                                                                                        ],dtype=th.int64)
        data["current_state", "dist_to_observation", "observation"].edge_index = th.tensor([
                                                                                        [0],  # idx of source nodes (current state)
                                                                                        [0],  # idx of target nodes (observation)
                                                                                        ],dtype=th.int64)
        data["goal_state", "dist_to_goal_state", "current_state"].edge_index = th.tensor([
                                                                                        [],  # idx of source nodes (goal state)
                                                                                        [],  # idx of target nodes (current state)
                                                                                        ],dtype=th.int64)
        data["observation", "dist_to_observation", "current_state"].edge_index = th.tensor([
                                                                                        [],  # idx of source nodes (observation)
                                                                                        [],  # idx of target nodes (current state)
                                                                                        ],dtype=th.int64)

        # add edge weights
        data["current_state", "dist_to_goal_state", "goal_state"].edge_attr = dist_current_state_goal
        data["current_state", "dist_to_observation", "observation"].edge_attr = dist_current_state_obs
        # make it undirected
        data["goal_state", "dist_to_goal_state", "current_state"].edge_attr = dist_current_state_goal
        data["observation", "dist_to_observation", "current_state"].edge_attr = dist_current_state_obs

        # convert the data to the device
        data = data.to(device)

        " ********************* RETURN ********************* "

        return data