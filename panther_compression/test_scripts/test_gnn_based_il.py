#!/usr/bin/env python3

# HGT based imitation learning

import sys
import os
from os.path import exists
import math 
import gym
from gym import spaces
import numpy as np
from numpy import load
import torch as th
import matplotlib.pyplot as plt
import argparse

import rosbag
import rospy
import rospkg

from geometry_msgs.msg import PointStamped, TransformStamped, PoseStamped, Vector3, Quaternion, Pose
from visualization_msgs.msg import Marker, MarkerArray
from tf2_msgs.msg import TFMessage
from nav_msgs.msg import Path

from mpl_toolkits.mplot3d import Axes3D
import yaml
from scipy.optimize import linear_sum_assignment

import time
from statistics import mean
import copy
from random import random, shuffle
from colorama import init, Fore, Back, Style
import py_panther
from joblib import Parallel, delayed
import multiprocessing

from torch_geometric.data import Data, HeteroData
from torch_geometric.nn import GATConv, to_hetero, HeteroConv, Linear, HGTConv
import networkx as nx
from torch_geometric.utils.convert import to_networkx
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1', 'True'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0', 'False'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# args
parser = argparse.ArgumentParser()
parser.add_argument('-v', '--visualize_traj', default=True, help='visualize trajectory', type=str2bool)
parser.add_argument('-t', '--train', default=False, help='train the model', type=str2bool)
parser.add_argument('-l', '--load_f_obs_ns', default=False, help='load f_obs_ns', type=str2bool)
parser.add_argument('-p', '--train-only-pos', default=False, help='train only pos', type=str2bool)
parser.add_argument('-e', '--evaluate-performance', default=False, help='evaluate performance', type=str2bool)

IS_VISUALIZE_TRAJ = parser.parse_args().visualize_traj
IS_TRAIN_MODEL = parser.parse_args().train
IS_LOAD_F_OBS_NS = parser.parse_args().load_f_obs_ns
IS_TRAIN_ONLY_POS = parser.parse_args().train_only_pos
IS_EVALUATE_PERFORMANCE = parser.parse_args().evaluate_performance

if IS_VISUALIZE_TRAJ:

    from imitation.data import rollout, types
    from compression.policies.ExpertPolicy import ExpertPolicy
    from compression.policies.StudentPolicy import StudentPolicy
    from imitation.util import util
    from compression.utils.other import ObstaclesManager, ObservationManager, ActionManager, \
                                        CostComputer, State, GTermManager, getPANTHERparamsAsCppStruct, \
                                        computeTotalTime, posAccelYaw2TfMatrix, ExpertDidntSucceed, \
                                        TfMatrix2RosQuatAndVector3, TfMatrix2RosPose, \
                                        MyClampedUniformBSpline, listOf3dVectors2numpy3Xmatrix, getZeroState
    from imitation.algorithms import bc

if IS_EVALUATE_PERFORMANCE:

    from compression.utils.other import ObstaclesManager, ObservationManager, ActionManager, \
                                        CostComputer, State, GTermManager, getPANTHERparamsAsCppStruct, \
                                        computeTotalTime, posAccelYaw2TfMatrix, ExpertDidntSucceed, \
                                        TfMatrix2RosQuatAndVector3, TfMatrix2RosPose, \
                                        MyClampedUniformBSpline, listOf3dVectors2numpy3Xmatrix, getZeroState


" ********************* CLASS DEFINITION ********************* "

# class HGT(th.nn.Module):

#     def __init__(self, hidden_channels, out_channels, num_heads, num_layers, data):
#         super().__init__()

#         self.lin_dict = th.nn.ModuleDict()
#         for node_type in data.node_types:
#             self.lin_dict[node_type] = Linear(-1, hidden_channels)

#         self.convs = th.nn.ModuleList()
#         for _ in range(num_layers):
#             conv = HGTConv(hidden_channels, hidden_channels, data.metadata(),
#                            num_heads, group='sum')
#             self.convs.append(conv)

#         self.lin = Linear(-1, out_channels)

#         self.double() # convert all the parameters to double

#     def forward(self, x_dict, edge_index_dict):
#         for node_type, x in x_dict.items():
#             x_dict[node_type] = self.lin_dict[node_type](x).relu_()

#         for conv in self.convs:
#             x_dict = conv(x_dict, edge_index_dict)
    
#         # extract the global embedding
#         x = th.cat([x for _, x in sorted(x_dict.items())], dim=-1)
#         x = self.lin(x)

#         return x

class HGT(th.nn.Module):

    def __init__(self, hidden_channels, out_channels, num_heads, num_layers, group, num_linear_layers, linear_hidden_channels, num_of_trajs_per_replan, data):
        super().__init__()

        self.num_of_trajs_per_replan = num_of_trajs_per_replan
        self.out_channels = out_channels

        self.lin_dict = th.nn.ModuleDict()
        for node_type in data.node_types:
            self.lin_dict[node_type] = Linear(-1, hidden_channels)

        self.convs = th.nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(hidden_channels, hidden_channels, data.metadata(),
                           num_heads, group=group)
            self.convs.append(conv)

        # add linear layers (num_linear_layers) times
        self.lins = th.nn.ModuleList()
        for _ in range(num_linear_layers-1):
            self.lins.append(Linear(-1, linear_hidden_channels)) 
        self.lins.append(Linear(linear_hidden_channels, out_channels*num_of_trajs_per_replan)) 

        self.double() # convert all the parameters to doublew

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

        return output

" ********************* FUNCTION DEFINITION ********************* "

def print_graph_info_and_draw_graph(data):

    " ********************* PRINT OUT THE GRAPH ********************* "

    print("number of nodes: ", data.num_nodes)
    print("number of edges: ", data.num_edges)
    print("number of features per node: ", data.num_node_features)
    print("number of weights per edge: ", data.num_edge_features)

    G = to_networkx(data)
    nx.draw(G, with_labels=True)
    plt.show()

def train(model, batch, optimizer):

    """
    This function trains the model and return the loss
    """
    model.train() # set the model.training to be True
    optimizer.zero_grad() # reset the gradient of all variables
    predicted_traj = model(batch.x_dict, batch.edge_index_dict)

    # since predicted_traj is a tensor of size [batch_size, traj_size], we need to reshape true_traj to be [batch_size, traj_size] by repeating true_traj
    # true_traj = true_traj.repeat(predicted_traj.shape[0], 1)

    # compute the loss 
    # loss = F.mse_loss(predicted_traj, batch.true_traj[:,0,:]) # just use the first traj to compute the loss

    # number of ctrlpoints for pos = 3 * (num_seg_pos + deg_pos - 5) = 3 * (7 + 3 - 5) = 15
    # number of ctrlpoints for yaw = (num_seg_yaw + deg_yaw - 3) = (7 + 2 - 3) = 6
    # the last element is time = 1
    # the total number of ctrlpoints = 15 + 6 + 1 = 22
    loss, stats_dict = calculate_loss(predicted_traj, batch.true_traj, 15, 6, 1, device)
    
    # compute the gradient
    loss.backward()

    # update the parameters
    optimizer.step()

    return loss

@th.no_grad()
def test(model, data):

    """
    This function tests the model and return the loss
    """

    model.eval() # set the model.training to be False
    predicted_traj = model(data.x_dict, data.edge_index_dict) # Log prob of all data

    # since predicted_traj is a tensor of size [batch_size, traj_size], we need to reshape true_traj to be [batch_size, traj_size] by repeating true_traj
    # true_traj = true_traj.repeat(predicted_traj.shape[0], 1)

    # compute the loss
    # loss, stats_dict = calculate_loss(predicted_traj, data.true_traj, 15, 6, 1, device)
    # using caluate_loss function requires the graduent to be True but we don't wanna do that
    # so we just use the mse loss for the first traj (therefore it's not really accurate)
    loss = F.mse_loss(predicted_traj[:,0,:], data.true_traj[:,0,:])

    return loss

def printFailedOpt(info):
    print(" Called optimizer--> "+Style.BRIGHT+Fore.RED +"Failed"+ Style.RESET_ALL+". "+ info)

def printSucessOpt(info):
    print(" Called optimizer--> "+Style.BRIGHT+Fore.GREEN +"Success"+ Style.RESET_ALL+". "+ info)

def generate_trajectory():

    """
    This function generate trajectory and return f_obs_n and f_traj
    """

    ##
    ## params
    ##

    ENV_NAME = 	"my-environment-v1" # you shouldn't change the name
    num_envs = 	1
    seed = 	1

    ##
    ## get vectorized environment
    ##

    venv = gym.make(ENV_NAME)
    venv.seed(seed)

    ##
    ## initial (current) state
    ##

    current_state = getZeroState()

    ##
    ## get obsearvation
    ##

    w_obstacles = venv.obsm.getFutureWPosStaticObstacles()
    f_obs = venv.om.get_fObservationFrom_w_stateAnd_w_gtermAnd_w_obstacles(current_state, venv.gm.get_w_GTermPos(), w_obstacles)
    f_obs_n = venv.om.normalizeObservation(f_obs)

    ##
    ## compare it to master's trajectory
    ##

    par_v_max = [10.0, 10.0, 10.0]
    par_a_max = [15.0, 15.0, 15.0]
    par_factor_alloc = 1.0

    my_SolverIpopt=py_panther.SolverIpopt(getPANTHERparamsAsCppStruct())
    init_state=venv.om.getInit_f_StateFromObservation(f_obs)       
    final_state=venv.om.getFinal_f_StateFromObservation(f_obs)        
    total_time=computeTotalTime(init_state, final_state, par_v_max, par_a_max, par_factor_alloc)
    ExpertPolicy.my_SolverIpopt.setInitStateFinalStateInitTFinalT(init_state, final_state, 0.0, total_time)
    ExpertPolicy.my_SolverIpopt.setFocusOnObstacle(True)
    obstacles=venv.om.getObstaclesForCasadi(f_obs)
    ExpertPolicy.my_SolverIpopt.setObstaclesForOpt(obstacles)
    succeed=ExpertPolicy.my_SolverIpopt.optimize(True)
    info=ExpertPolicy.my_SolverIpopt.getInfoLastOpt()

    ##
    ## Print results
    ##

    if not succeed:
        printFailedOpt(info)
    else:
        printSucessOpt(info)

    best_solutions=ExpertPolicy.my_SolverIpopt.getBestSolutions()
    actions=venv.am.solsOrGuesses2action(best_solutions)
    actions_normalized=venv.am.normalizeAction(actions)
    index_smallest_augmented_cost=venv.cost_computer.getIndexBestTraj(f_obs_n, actions_normalized)
    f_traj=venv.am.getTrajFromAction(actions, index_smallest_augmented_cost)

    return f_obs_n, f_traj

def generate_dataset(f_obs_ns, true_trajs, device):

    """
    This function generates a dataset for GNN
    """

    " ********************* INITIALIZE DATASET ********************* "

    dataset = []

    assert len(f_obs_ns) == len(true_trajs), "the length of f_obs_ns and true_trajs should be the same"

    for i in range(len(f_obs_ns)):

        " ********************* GET NODES ********************* "

        # nodes you need for GNN
        # 0. current state
        # 1. goal state
        # 2. observation
        # In the current setting f_obs is a realative state from the current state so we pass f_v, f_z, yaw_dot to the current state node

        feature_vector_for_current_state = f_obs_ns[i][0:7].clone().detach().unsqueeze(0).to('cpu')
        feature_vector_for_goal = f_obs_ns[i][7:10].clone().detach().unsqueeze(0).to('cpu')
        feature_vector_for_obs = f_obs_ns[i][10:].clone().detach().unsqueeze(0).to('cpu')
        dist_current_state_goal = th.tensor([np.linalg.norm(feature_vector_for_goal[0][:3])], dtype=th.double).to(device)
        dist_current_state_obs = th.tensor([np.linalg.norm(feature_vector_for_obs[0][:3])], dtype=th.double).to(device)

        " ********************* MAKE A DATA OBJECT FOR HETEROGENEUS GRAPH ********************* "

        data = HeteroData()

        # add nodes
        data["current_state"].x = feature_vector_for_current_state # [number of "current_state" nodes, size of feature vector]
        data["goal_state"].x = feature_vector_for_goal # [number of "goal_state" nodes, size of feature vector]
        data["observation"].x = feature_vector_for_obs # [number of "observation" nodes, size of feature vector]

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

        # add ground truth trajectory
        if not IS_TRAIN_ONLY_POS:
            data.true_traj = true_trajs[i].clone().detach().unsqueeze(0).to(device)
        else:
            columns = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,-1]
            data.true_traj = true_trajs[i][:,columns].clone().detach().unsqueeze(0).to(device)

        # convert the data to the device
        data = data.to(device)
        # append data to the dataset
        dataset.append(data)

    " ********************* RETURN ********************* "

    return dataset

def visualize_trajectory(model, data, f_obs_n, true_trajs):

    am = ActionManager() # get action manager
    om = ObservationManager() # get observation manager

    predicted_traj_n = model(data.x_dict, data.edge_index_dict) # get the predicted trajectory
    predicted_traj_n = predicted_traj_n.squeeze().detach().numpy() # convert it to numpy array

    if IS_TRAIN_ONLY_POS:
        new_predicted_traj_n = np.zeros((predicted_traj_n.shape[0], 22))
        new_predicted_traj_n[:,:21] = np.concatenate((predicted_traj_n[:,:-1], np.zeros((predicted_traj_n.shape[0], 6))), axis=1)
        for i in range(new_predicted_traj_n.shape[0]):
            new_predicted_traj_n[i,21] = predicted_traj_n[i,-1]
        predicted_traj_n = new_predicted_traj_n.copy()

    true_trajs = true_trajs.squeeze().detach().numpy() # convert it to numpy array

    # plot 
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(211, projection='3d')

    assert predicted_traj_n.shape[0] == true_trajs.shape[0], "the number of predicted trajectories and true trajectories should be the same"

    # get the predicted trajectory
    for idx in range(predicted_traj_n.shape[0]):

        pred_traj = am.denormalizeTraj(predicted_traj_n[idx])
        true_traj = am.denormalizeTraj(true_trajs[idx])

        # add one dimension to the numpy trajectory
        pred_traj = np.expand_dims(pred_traj, axis=0)
        true_traj = np.expand_dims(true_traj, axis=0)

        # convert the trajectory to a b-spline
        start_state = getZeroState()
        w_posBS_pred, w_yawBS_pred = am.f_trajAnd_w_State2wBS(pred_traj, start_state)
        w_posBS_true, w_yawBS_true = am.f_trajAnd_w_State2wBS(true_traj, start_state)
        num_vectors_pos = 100
        num_vectors_yaw = 10
        time_pred = np.linspace(w_posBS_pred.getT0(), w_posBS_pred.getTf(), num_vectors_pos)
        time_yaw_pred = np.linspace(w_yawBS_pred.getT0(), w_yawBS_pred.getTf(), num_vectors_yaw)
        time_true = np.linspace(w_posBS_true.getT0(), w_posBS_true.getTf(), num_vectors_pos)
        time_yaw_true = np.linspace(w_yawBS_true.getT0(), w_yawBS_true.getTf(), num_vectors_yaw)

        # plot the predicted trajectory
        if idx == 0:
            ax.plot(w_posBS_true.pos_bs[0](time_true), w_posBS_true.pos_bs[1](time_true), w_posBS_true.pos_bs[2](time_true), lw=4, alpha=0.7, label='Expert', c='green')
            ax.plot(w_posBS_pred.pos_bs[0](time_pred), w_posBS_pred.pos_bs[1](time_pred), w_posBS_pred.pos_bs[2](time_pred), lw=4, alpha=0.7, label='GNN', c='orange')

            # plot the start and goal position
            ax.scatter(w_posBS_true.pos_bs[0](w_posBS_true.getT0()), w_posBS_true.pos_bs[1](w_posBS_true.getT0()), w_posBS_true.pos_bs[2](w_posBS_true.getT0()), s=100, c='green', marker='o', label='Start')
            ax.scatter(w_posBS_true.pos_bs[0](w_posBS_true.getTf()), w_posBS_true.pos_bs[1](w_posBS_true.getTf()), w_posBS_true.pos_bs[2](w_posBS_true.getTf()), s=100, c='red', marker='o', label='Goal')

        else:
            ax.plot(w_posBS_true.pos_bs[0](time_true), w_posBS_true.pos_bs[1](time_true), w_posBS_true.pos_bs[2](time_true), lw=4, alpha=0.7, c='green')
            ax.plot(w_posBS_pred.pos_bs[0](time_pred), w_posBS_pred.pos_bs[1](time_pred), w_posBS_pred.pos_bs[2](time_pred), lw=4, alpha=0.7, c='orange')

    # plot the obstacles
    f_obs = om.denormalizeObservation(f_obs_n)
    
    # get w pos of the obstacles
    w_obs_poses = []
    p0 = start_state.w_pos[0] # careful here: we assume the agent pos is at the origin - as the agent moves we expect the obstacles shift accordingly
    for i in range(int(len(f_obs[0][10:-3])/3)):
        w_obs_poses.append((f_obs[0][3*i+10:3*i+3+10].clone().detach().numpy() - p0.T).tolist())
    bbox = f_obs[0][-3:].clone().detach().numpy()

    for idx, w_obs_pos in enumerate(w_obs_poses):
        # obstacle's position
        # bbox (8 points)
        p1 = [w_obs_pos[0] + bbox[0]/2, w_obs_pos[1] + bbox[1]/2, w_obs_pos[2] + bbox[2]/2]
        p2 = [w_obs_pos[0] + bbox[0]/2, w_obs_pos[1] + bbox[1]/2, w_obs_pos[2] - bbox[2]/2]
        p3 = [w_obs_pos[0] + bbox[0]/2, w_obs_pos[1] - bbox[1]/2, w_obs_pos[2] + bbox[2]/2]
        p4 = [w_obs_pos[0] + bbox[0]/2, w_obs_pos[1] - bbox[1]/2, w_obs_pos[2] - bbox[2]/2]
        p5 = [w_obs_pos[0] - bbox[0]/2, w_obs_pos[1] + bbox[1]/2, w_obs_pos[2] + bbox[2]/2]
        p6 = [w_obs_pos[0] - bbox[0]/2, w_obs_pos[1] + bbox[1]/2, w_obs_pos[2] - bbox[2]/2]
        p7 = [w_obs_pos[0] - bbox[0]/2, w_obs_pos[1] - bbox[1]/2, w_obs_pos[2] + bbox[2]/2]
        p8 = [w_obs_pos[0] - bbox[0]/2, w_obs_pos[1] - bbox[1]/2, w_obs_pos[2] - bbox[2]/2]
        # bbox lines (12 lines)
        if idx == 0:
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], lw=2, alpha=0.7, c='blue', label='Obstacle')
        else:
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], lw=2, alpha=0.7, c='blue')
        ax.plot([p1[0], p3[0]], [p1[1], p3[1]], [p1[2], p3[2]], lw=2, alpha=0.7, c='blue')
        ax.plot([p1[0], p5[0]], [p1[1], p5[1]], [p1[2], p5[2]], lw=2, alpha=0.7, c='blue')
        ax.plot([p2[0], p4[0]], [p2[1], p4[1]], [p2[2], p4[2]], lw=2, alpha=0.7, c='blue')
        ax.plot([p2[0], p6[0]], [p2[1], p6[1]], [p2[2], p6[2]], lw=2, alpha=0.7, c='blue')
        ax.plot([p3[0], p4[0]], [p3[1], p4[1]], [p3[2], p4[2]], lw=2, alpha=0.7, c='blue')
        ax.plot([p3[0], p7[0]], [p3[1], p7[1]], [p3[2], p7[2]], lw=2, alpha=0.7, c='blue')
        ax.plot([p4[0], p8[0]], [p4[1], p8[1]], [p4[2], p8[2]], lw=2, alpha=0.7, c='blue')
        ax.plot([p5[0], p6[0]], [p5[1], p6[1]], [p5[2], p6[2]], lw=2, alpha=0.7, c='blue')
        ax.plot([p5[0], p7[0]], [p5[1], p7[1]], [p5[2], p7[2]], lw=2, alpha=0.7, c='blue')
        ax.plot([p6[0], p8[0]], [p6[1], p8[1]], [p6[2], p8[2]], lw=2, alpha=0.7, c='blue')
        ax.plot([p7[0], p8[0]], [p7[1], p8[1]], [p7[2], p8[2]], lw=2, alpha=0.7, c='blue')

    ax.grid(True)
    ax.legend(loc='best')
    ax.set_xlim(-2, 15)
    ax.set_ylim(-4, 4)
    ax.set_zlim(-5, 5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_aspect('equal')
    # plt.show()

    fig.savefig('test_pos.png')

    # plot the yaw trajectory
    ax = fig.add_subplot(212)

    for idx in range(predicted_traj_n.shape[0]):

        pred_traj = am.denormalizeTraj(predicted_traj_n[idx])
        true_traj = am.denormalizeTraj(true_trajs[idx])

        # add one dimension to the numpy trajectory
        pred_traj = np.expand_dims(pred_traj, axis=0)
        true_traj = np.expand_dims(true_traj, axis=0)

        # convert the trajectory to a b-spline
        start_state = getZeroState()
        w_posBS_pred, w_yawBS_pred = am.f_trajAnd_w_State2wBS(pred_traj, start_state)
        w_posBS_true, w_yawBS_true = am.f_trajAnd_w_State2wBS(true_traj, start_state)

        if idx == 0:
            ax.plot(time_yaw_true, w_yawBS_true.pos_bs[0](time_yaw_true), lw=4, alpha=0.7, label='Expert', c='green')
            ax.plot(time_yaw_pred, w_yawBS_pred.pos_bs[0](time_yaw_pred), lw=4, alpha=0.7, label='GNN', c='orange')
        else:
            ax.plot(time_yaw_true, w_yawBS_true.pos_bs[0](time_yaw_true), lw=4, alpha=0.7, c='green')
            ax.plot(time_yaw_pred, w_yawBS_pred.pos_bs[0](time_yaw_pred), lw=4, alpha=0.7, c='orange')

    ax.grid(True)
    ax.legend(loc='best')
    ax.set_xlabel('Time')
    ax.set_ylabel('Yaw')
    # fig.savefig('/home/kota/Research/deep-panther_ws/src/deep_panther/panther/matlab/figures/test_yaw.png')
    plt.show()

def calculate_loss(pred_traj, true_traj, traj_size_pos_ctrl_pts, traj_size_yaw_ctrl_pts, yaw_loss_weight, device):

    #Expert --> i
    #Student --> j
    num_of_traj_per_action=list(pred_traj.shape)[1] #pred_traj.shape is [batch size, num_traj_action, size_traj]
    num_of_elements_per_traj=list(pred_traj.shape)[2] #pred_traj.shape is [batch size, num_traj_action, size_traj]
    batch_size=list(pred_traj.shape)[0] #pred_traj.shape is [batch size, num_of_traj_per_action, size_traj]

    distance_matrix= th.zeros(batch_size, num_of_traj_per_action, num_of_traj_per_action, device=device)
    distance_pos_matrix= th.zeros(batch_size, num_of_traj_per_action, num_of_traj_per_action, device=device) 
    distance_yaw_matrix= th.zeros(batch_size, num_of_traj_per_action, num_of_traj_per_action, device=device) 
    distance_time_matrix= th.zeros(batch_size, num_of_traj_per_action, num_of_traj_per_action, device=device)
    distance_pos_matrix_within_expert= th.zeros(batch_size, num_of_traj_per_action, num_of_traj_per_action, device=device)

    for i in range(num_of_traj_per_action):
        for j in range(num_of_traj_per_action):

            # All the elements
            expert_i=       true_traj[:,i,:].float() #All the elements
            student_j=      pred_traj[:,j,:].float() #All the elements

            # Pos
            expert_pos_i=   true_traj[:,i,:traj_size_pos_ctrl_pts].float()
            student_pos_j=  pred_traj[:,j,:traj_size_pos_ctrl_pts].float()

            # Yaw
            expert_yaw_i=   true_traj[:,i,traj_size_pos_ctrl_pts:(traj_size_pos_ctrl_pts+traj_size_yaw_ctrl_pts-1)].float()
            student_yaw_j=  pred_traj[:,j,traj_size_pos_ctrl_pts:(traj_size_pos_ctrl_pts+traj_size_yaw_ctrl_pts-1)].float()

            # Time
            expert_time_i=       true_traj[:,i,-1:].float() #Time. Note: Is you use only -1 (instead of -1:), then distance_time_matrix will have required_grad to false
            student_time_j=      pred_traj[:,j,-1:].float() #Time. Note: Is you use only -1 (instead of -1:), then distance_time_matrix will have required_grad to false

            # Distance matrices
            distance_matrix[:,i,j]=th.mean(th.nn.MSELoss(reduction='none')(expert_i, student_j), dim=1)
            distance_pos_matrix[:,i,j]=th.mean(th.nn.MSELoss(reduction='none')(expert_pos_i, student_pos_j), dim=1)
            distance_yaw_matrix[:,i,j]=th.mean(th.nn.MSELoss(reduction='none')(expert_yaw_i, student_yaw_j), dim=1)
            distance_time_matrix[:,i,j]=th.mean(th.nn.MSELoss(reduction='none')(expert_time_i, student_time_j), dim=1)

            #This is simply to delete the trajs from the expert that are repeated
            expert_pos_j=   true_traj[:,j,0:traj_size_pos_ctrl_pts].float()
            distance_pos_matrix_within_expert[:,i,j]=th.mean(th.nn.MSELoss(reduction='none')(expert_pos_i, expert_pos_j), dim=1)

    is_repeated=th.zeros(batch_size, num_of_traj_per_action, dtype=th.bool, device=device)

    for i in range(num_of_traj_per_action):
        for j in range(i+1, num_of_traj_per_action):
            is_repeated[:,j]=th.logical_or(is_repeated[:,j], th.lt(distance_pos_matrix_within_expert[:,i,j], 1e-7))

    assert distance_matrix.requires_grad==True
    assert distance_pos_matrix.requires_grad==True
    assert distance_yaw_matrix.requires_grad==True
    assert distance_time_matrix.requires_grad==True

    #Option 1: Solve assignment problem
    A_matrix=th.ones(batch_size, num_of_traj_per_action, num_of_traj_per_action, device=device)

    if(num_of_traj_per_action>1):

        #Option 1: Solve assignment problem
        A_matrix=th.zeros(batch_size, num_of_traj_per_action, num_of_traj_per_action, device=device)

        for index_batch in range(batch_size):         

            # cost_matrix_numpy=distance_pos_matrix_numpy[index_batch,:,:];
            cost_matrix=distance_pos_matrix[index_batch,:,:]
            map2RealRows=np.array(range(num_of_traj_per_action))
            map2RealCols=np.array(range(num_of_traj_per_action))

            rows_to_delete=[]
            for i in range(num_of_traj_per_action): #for each row (expert traj)
                if(is_repeated[index_batch,i]==True): 
                    #Delete that row
                    rows_to_delete.append(i)

            cost_matrix=cost_matrix[is_repeated[index_batch,:]==False]   #np.delete(cost_matrix_numpy, rows_to_delete, axis=0)
            cost_matrix_numpy=cost_matrix.cpu().detach().numpy()
            distance_pos_matrix_batch_tmp=distance_pos_matrix[index_batch,:,:].clone()
            distance_pos_matrix_batch_tmp[is_repeated[index_batch,:],:] = float('inf')  #Set the ones that are repeated to infinity

            ### RWTAc: This version ensures that the columns sum up to one (This is what https://arxiv.org/pdf/2110.05113.pdf does, see Eq.6)
            minimum_per_column, row_indexes =th.min(distance_pos_matrix_batch_tmp[:,:], 0) #Select the minimum values
            col_indexes=th.arange(0, distance_pos_matrix_batch_tmp.shape[1], dtype=th.int64)
            minimum_per_row, col_indexes =th.min(distance_pos_matrix_batch_tmp[:,:], dim=1) #Select the minimum values
            row_indexes=th.arange(0, distance_pos_matrix_batch_tmp.shape[0], dtype=th.int64)

            # Solve assignment problem                                       
            row_indexes, col_indexes = linear_sum_assignment(cost_matrix_numpy)
            for row_index, col_index in zip(row_indexes, col_indexes):
                A_matrix[index_batch, map2RealRows[row_index], map2RealCols[col_index]]=1
            
    num_nonzero_A=th.count_nonzero(A_matrix) #This is the same as the number of distinct trajectories produced by the expert

    pos_loss=th.sum(A_matrix*distance_pos_matrix)/num_nonzero_A
    yaw_loss=th.sum(A_matrix*distance_yaw_matrix)/num_nonzero_A
    time_loss=th.sum(A_matrix*distance_time_matrix)/num_nonzero_A

    assert (distance_matrix.shape)[0]==batch_size, "Wrong shape!"
    assert (distance_matrix.shape)[1]==num_of_traj_per_action, "Wrong shape!"
    assert pos_loss.requires_grad==True
    assert yaw_loss.requires_grad==True
    assert time_loss.requires_grad==True

    loss_Hungarian = time_loss
    loss_Hungarian += pos_loss
    if not IS_TRAIN_ONLY_POS:
        loss_Hungarian += yaw_loss
    loss = loss_Hungarian

    # stats dict
    stats_dict = dict(
        loss_Hungarian=loss_Hungarian.item(),
        pos_loss=pos_loss.item(),
        yaw_loss=yaw_loss_weight*yaw_loss.item(),
        time_loss=time_loss.item(),
    )

    return loss, stats_dict

def evaluate_performance(model, data, f_obs_n, true_traj):

    cc = CostComputer() # get cost computer
    am = ActionManager() # get action manager
    om = ObservationManager() # get observation manager

    predicted_traj_n = model(data.x_dict, data.edge_index_dict) # get the predicted trajectory
    predicted_traj_n = predicted_traj_n.detach().numpy() # convert it to numpy array
    costs_and_violations_of_action = cc.getCostsAndViolationsOfActionFromObsnAndActionn(f_obs_n, predicted_traj_n)

    w_init_state = getZeroState() # get the initial state

    total_cost = 0
    total_aug_cost = 0
    max_cost_idx = 0
    max_aug_cost_idx = 0

    for i in range(predicted_traj_n.shape[0]): #For each row of action

        my_solOrGuess= am.f_trajAnd_w_State2w_ppSolOrGuess(predicted_traj_n[i], w_init_state)
        my_solOrGuess.cost = costs_and_violations_of_action.costs[i]
        my_solOrGuess.obst_avoidance_violation = costs_and_violations_of_action.obst_avoidance_violations[i]
        my_solOrGuess.dyn_lim_violation = costs_and_violations_of_action.dyn_lim_violations[i]
        my_solOrGuess.aug_cost = cc.computeAugmentedCost(my_solOrGuess.cost, my_solOrGuess.obst_avoidance_violation, my_solOrGuess.dyn_lim_violation)

        total_cost += my_solOrGuess.cost
        total_aug_cost += my_solOrGuess.aug_cost

        if my_solOrGuess.cost > costs_and_violations_of_action.costs[max_cost_idx]:
            max_cost_idx = i
        if my_solOrGuess.aug_cost > costs_and_violations_of_action.aug_costs[max_aug_cost_idx]:
            max_aug_cost_idx = i
    
    avg_cost = total_cost/predicted_traj_n.shape[0]
    avg_aug_cost = total_aug_cost/predicted_traj_n.shape[0]

    print("Student")
    print("average cost: ", avg_cost)
    print("average augmented cost: ", avg_aug_cost)
    print("max cost: ", costs_and_violations_of_action.costs[max_cost_idx])
    print("max augmented cost: ", costs_and_violations_of_action.aug_costs[max_aug_cost_idx])

    for i in range(true_traj.shape[0]):

        my_solOrGuess= am.f_trajAnd_w_State2w_ppSolOrGuess(true_traj[i], w_init_state)
        my_solOrGuess.cost = costs_and_violations_of_action.costs[i]
        my_solOrGuess.obst_avoidance_violation = costs_and_violations_of_action.obst_avoidance_violations[i]
        my_solOrGuess.dyn_lim_violation = costs_and_violations_of_action.dyn_lim_violations[i]
        my_solOrGuess.aug_cost = cc.computeAugmentedCost(my_solOrGuess.cost, my_solOrGuess.obst_avoidance_violation, my_solOrGuess.dyn_lim_violation)

        total_cost += my_solOrGuess.cost
        total_aug_cost += my_solOrGuess.aug_cost

        if my_solOrGuess.cost > costs_and_violations_of_action.costs[max_cost_idx]:
            max_cost_idx = i
        if my_solOrGuess.aug_cost > costs_and_violations_of_action.aug_costs[max_aug_cost_idx]:
            max_aug_cost_idx = i
    
    avg_cost = total_cost/true_traj.shape[0]
    avg_aug_cost = total_aug_cost/true_traj.shape[0]

    print("Expert")
    print("average cost: ", avg_cost)
    print("average augmented cost: ", avg_aug_cost)
    print("max cost: ", costs_and_violations_of_action.costs[max_cost_idx])
    print("max augmented cost: ", costs_and_violations_of_action.aug_costs[max_aug_cost_idx])

    exit()



def get_latest_model(dir_idx):
    files = os.listdir(f"models/dir-{dir_idx}/")
    files = [file for file in files if file.endswith('.pt')]
    files.sort()
    file = f"models/dir-{dir_idx}/"+files[-1]
    model = th.load(file).to('cpu')
    return model

def main(dataset, device, hidden_channels, num_heads, num_layers, group, num_linear_layers, linear_hidden_channels, num_of_trajs_per_replan, folder_name):

    """
    Main function
    """

    " ********************* DATA CONSTRUCTION ********************* "

    TRAIN_SPLIT = 0.8
    EVALUATION_SPLIT = 0.1
    TEST_SPLIT = 0.1
    
    train_dataset = dataset[:int(TRAIN_SPLIT*len(dataset))]
    eval_dataset = dataset[int((TRAIN_SPLIT+TEST_SPLIT)*len(dataset)):]
    test_dataset = dataset[int(TRAIN_SPLIT*len(dataset)):int((TRAIN_SPLIT+TEST_SPLIT)*len(dataset))]

    " ********************* CONSTRUCT THE MODEL ********************* "

    data = dataset[0] # for HGT model initialization purpose
    out_chnnels = data.true_traj.shape[-1] if not IS_TRAIN_ONLY_POS else 22 - 6

    model = HGT(hidden_channels=hidden_channels, out_channels=out_chnnels, 
                num_heads=num_heads, num_layers=num_layers, group=group, num_linear_layers=num_linear_layers, 
                linear_hidden_channels=linear_hidden_channels, num_of_trajs_per_replan=num_of_trajs_per_replan, data=data)
    model = model.to(device)
    optimizer = th.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    data = data.to(device)
    with th.no_grad():
        out = model(data.x_dict, data.edge_index_dict)
    
    " ********************* CREATE DATA LOADER ********************* "

    # data loader
    BATCH_SIZE = 64
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=True)

    " ********************* TRAINING ********************* "

    EPOCH_SIZE = 50

    # Epoch loop
    tic = time.time()
    converge_count = 0
    overfitting_count = 0
    previous_total_training_loss = 0
    total_training_loss = 0
    total_eval_loss = 0
    total_testing_loss = 0
    for epoch in range(1, EPOCH_SIZE):

        print(f"\nEpoch: {epoch}/{EPOCH_SIZE}")

        # Batch loop
        total_training_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            batch = batch.to(device)
            loss = train(model, batch, optimizer)
            total_training_loss += loss.item()
            if batch_idx % 100 == 0:
                print(f"Batch: {batch_idx}/{len(train_loader)}")

        print(f"Training Loss: {total_training_loss/len(train_dataset)}")

        # evaluate the model every 10 epochs
        if epoch % 10 == 0:
            total_eval_loss = 0
            for batch in eval_loader:
                batch = batch.to(device)
                loss = test(model, batch)
                total_eval_loss += loss.item()

            print("Evaluation Loss: ", total_eval_loss/len(eval_dataset))

        # time elapsed
        toc = time.time()
        print(f"Time elapsed: {toc-tic} seconds")

        th.save(model, folder_name + "model_" + time.strftime("%Y%m%d-%H%M%S") + ".pt")

        # terminanal conditions

        # if total_training_loss has converged, break
        if abs(total_training_loss - previous_total_training_loss) < 1e-5 and epoch > EPOCH_SIZE/2:
            converge_count += 1
            if converge_count == 10:
                print("the training loss has converged")
                break
        else:
            converge_count = 0
        
        # if the training loss is less than the evaluation loss, break
        if total_training_loss/len(train_dataset) < total_eval_loss/len(eval_dataset) and epoch > EPOCH_SIZE/2:
            overfitting_count += 1
            if overfitting_count == 10:
                print("the training loss is less than the evaluation loss")
                print(f"training loss: {total_training_loss/len(train_dataset)}, evaluation loss: {total_eval_loss/len(eval_dataset)}")
                break
        else:
            overfitting_count = 0

        previous_total_training_loss = total_training_loss

    " ********************* TESTING ********************* "

    total_testing_loss = 0

    for data in test_dataset:

        data = data.to(device)
        loss = test(model, data)
        total_testing_loss += loss.item()
    
    print("Testing Loss: ", total_testing_loss/len(test_dataset))

    " ********************* LOG ********************* "

    with open(folder_name + "log.txt", "a") as f:
        f.write(f"hidden_channels: {hidden_channels}, num_heads: {num_heads}, num_layers: {num_layers}, group: {group}, num_linear_layers: {num_linear_layers}\n")
        f.write(f"training loss: {total_training_loss/len(train_dataset)}, evaluation loss: {total_eval_loss/len(eval_dataset)}, testing loss: {total_testing_loss/len(test_dataset)}\n")
        f.write(f"training time: {toc-tic} seconds\n\n")

if __name__ == "__main__":

    device = th.device('cuda' if th.cuda.is_available() else 'cpu')

    if IS_LOAD_F_OBS_NS:

        print("LOADING f_obs_ns")
        f_obs_ns = th.load("f_obs_ns.pt")

    else: 
     
        " ********************* LOAD DATA ********************* "

        # list npz files in the directory
        dir = "../evals/tmp_dagger/2/demos/round-000/"
        # dir = "../evals_tmp/tmp_dagger/2/demos/round-000/"
        files = os.listdir(dir)
        files = [dir + file for file in files if file.endswith('.npz')]

        # loop over files
        obs_data = th.tensor([]).to(device)
        traj_data = th.tensor([]).to(device)
        for idx, file in enumerate(files):
            
            data = np.load(file) # load data
            obs = data['obs'][:-1] # remove the last observation (since it is the observation of the goal state)
            acts = data['acts'] # actions

            # append to the data
            obs_data = th.cat((obs_data, th.tensor(obs).to(device)), 0).to(device)
            traj_data = th.cat((traj_data, th.tensor(acts).to(device)), 0).to(device)
        
        " ********************* LOOP TO COLLECT f_obs_n and true_traj ********************* "

        # data structure
        # obs_data: [number of demonstrations, number of observations=1, size of observation]
        # traj_data: [number of demonstrations, number of trajectories=10, size of action]

        print("GENERATING f_obs_ns")

        f_obs_ns = obs_data[0].clone().detach().to(device) # To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
        true_trajs = traj_data[0].clone().detach().unsqueeze(0).to(device)

        dataset_size = obs_data.shape[0] if IS_TRAIN_MODEL else 10
        for i in range(1, dataset_size):

            print(f"\rdata index: {i+1}/{obs_data.shape[0]}", end="")

            # get f_obs_n and f_traj
            f_obs_ns = th.cat((f_obs_ns, obs_data[i].clone().detach()), 0)
            true_trajs = th.cat((true_trajs, traj_data[i].clone().detach().unsqueeze(0)), 0)

        th.save(f_obs_ns, "f_obs_ns.pt")

    " ********************* GENERATE DATASET ********************* "
    
    print("GENERATING DATASET")
    dataset = generate_dataset(f_obs_ns, true_trajs, device)

    " ********************* MAIN (TRAIN & TEST) ********************* "

    print("TRAINING AND TESTING")
    if IS_TRAIN_MODEL:

        # hyper parameters loop
        hidden_channels_list = [512]
        num_heads_list = [4]
        num_layers_list = [4, 8]
        group_list = ['sum', 'mean'] # 'mean'
        num_linear_layers_list = [2, 4]
        linear_hidden_channels_list = [512]
        num_of_trajs_per_replan = 10
        num_of_all_the_combinations = len(hidden_channels_list)*len(num_heads_list)*len(num_layers_list)*len(group_list)*len(num_linear_layers_list)*len(linear_hidden_channels_list)

        # for loop
        params_list = [(hidden_channels, num_heads, num_layers, group, num_linear_layers, linear_hidden_channels) for hidden_channels in hidden_channels_list for num_heads in num_heads_list for num_layers in num_layers_list for group in group_list for num_linear_layers in num_linear_layers_list for linear_hidden_channels in linear_hidden_channels_list]
        for dir_idx, [hidden_channels, num_heads, num_layers, group, num_linear_layers, linear_hidden_channels] in enumerate(params_list):

            print(f"\n\nhidden_channels: {hidden_channels}, num_heads: {num_heads}, num_layers: {num_layers}, group: {group}, num_linear_layers: {num_linear_layers}, linear_hidden_channels: {linear_hidden_channels}\n")

            # create a folder to save the model
            if not os.path.exists("models/"):
                os.makedirs("models/")
            folder_name = f"models/dir-{dir_idx}/"
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)
            
            # log the meta data
            with open(f"models/meta-log.txt", "a") as f:
                f.write("*********************\n")
                f.write(f"date: {time.strftime('%Y%m%d-%H%M%S')}\n")
                f.write(f"dir: {dir_idx}/{num_of_all_the_combinations}\n")
                f.write(f"hidden_channels: {hidden_channels} num_heads: {num_heads}, num_layers: {num_layers}, group: {group}, num_linear_layers: {num_linear_layers}, linear_hidden_channels: {linear_hidden_channels}\n")
                f.write("*********************\n")

            # train and test
            main(dataset, device, hidden_channels, num_heads, num_layers, group, num_linear_layers, linear_hidden_channels, num_of_trajs_per_replan, folder_name)

    " ********************* VISUALIZE ********************* "

    if IS_VISUALIZE_TRAJ:
        print("VISUALIZING TRAJECTORY")
        for data_idx, data in enumerate(dataset):
            data = data.to('cpu')
            true_traj = true_trajs[data_idx].clone().detach().unsqueeze(0).to('cpu')
            f_obs_n = f_obs_ns[data_idx].clone().detach().unsqueeze(0).to('cpu')
            # load the latest model (grab the latest .pt file in the models directory)
            dirs = os.listdir("models/")
            for dir_idx in range(len(dirs)-1):
                visualize_trajectory(get_latest_model(dir_idx), data, f_obs_n, true_traj)
    
    if IS_EVALUATE_PERFORMANCE:
        print("EVALUATING PERFORMANCE")
        for data_idx, data in enumerate(dataset):
            data = data.to('cpu')
            true_traj = true_trajs[data_idx].clone().detach().unsqueeze(0).to('cpu')
            f_obs_n = f_obs_ns[data_idx].clone().detach().unsqueeze(0).to('cpu')
            dirs = os.listdir("models/")
            for dir_idx in range(len(dirs)-1):
                evaluate_performance(get_latest_model(dir_idx), data, f_obs_n, true_traj)