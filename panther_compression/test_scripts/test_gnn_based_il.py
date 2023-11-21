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

# args
parser = argparse.ArgumentParser()
parser.add_argument('-v', '--visualize_traj', default=True, help='visualize trajectory')
parser.add_argument('-t', '--train', default=False, help='train the model')
parser.add_argument('-l', '--load_f_obs_ns', default=False, help='load f_obs_ns')
IS_VISUALIZE_TRAJ = parser.parse_args().visualize_traj
IS_TRAIN_MODEL = parser.parse_args().train
IS_LOAD_F_OBS_NS = parser.parse_args().load_f_obs_ns

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

" ********************* CLASS DEFINITION ********************* "

class HGT(th.nn.Module):

    def __init__(self, hidden_channels, out_channels, num_heads, num_layers, data):
        super().__init__()

        self.lin_dict = th.nn.ModuleDict()
        for node_type in data.node_types:
            self.lin_dict[node_type] = Linear(-1, hidden_channels)

        self.convs = th.nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(hidden_channels, hidden_channels, data.metadata(),
                           num_heads, group='sum')
            self.convs.append(conv)

        self.lin = Linear(-1, out_channels)

        self.double() # convert all the parameters to double

    def forward(self, x_dict, edge_index_dict):
        for node_type, x in x_dict.items():
            x_dict[node_type] = self.lin_dict[node_type](x).relu_()

        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
    
        # extract the global embedding
        x = th.cat([x for _, x in sorted(x_dict.items())], dim=-1)
        x = self.lin(x)

        return x

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

def train(model, batch, optimizer, true_traj):

    """
    This function trains the model and return the loss
    """
    model.train() # set the model.training to be True
    optimizer.zero_grad() # reset the gradient of all variables
    predicted_traj = model(batch.x_dict, batch.edge_index_dict)

    # compute the loss 
    loss = F.mse_loss(predicted_traj, true_traj)

    # compute the gradient
    loss.backward()

    # update the parameters
    optimizer.step()

    return loss

@th.no_grad()
def test(model, data, true_traj):

    """
    This function tests the model and return the loss
    """

    model.eval() # set the model.training to be False
    predicted_traj = model(data.x_dict, data.edge_index_dict) # Log prob of all data

    # compute the loss
    loss = F.mse_loss(predicted_traj, true_traj)

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

def generate_dataset(f_obs_ns, device):

    """
    This function generates a dataset for GNN
    """

    " ********************* INITIALIZE DATASET ********************* "

    dataset = []

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
        dist_current_state_goal = th.tensor([np.linalg.norm(feature_vector_for_obs[0][:3])], dtype=th.double).to(device)
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
        # data["current_state", "dist_to_observation", "observation"].edge_attr = dist_current_state_obs
        # make it undirected
        # data["goal_state", "dist_to_goal_state", "current_state"].edge_attr = dist_current_state_goal
        # data["observation", "dist_to_observation", "current_state"].edge_attr = dist_current_state_obs

        # convert the data to the device
        data = data.to(device)
        # append data to the dataset
        dataset.append(data)

    " ********************* RETURN ********************* "

    return dataset

def visualize_trajectory(model, data, f_obs_n, true_traj):

    am = ActionManager() # get action manager
    om = ObservationManager() # get observation manager

    predicted_traj_n = model(data.x_dict, data.edge_index_dict) # get the predicted trajectory
    predicted_traj_n = predicted_traj_n.detach().numpy() # convert it to numpy array

    # get the predicted trajectory
    predicted_traj = am.denormalizeTraj(predicted_traj_n)
    true_traj = am.denormalizeTraj(true_traj.detach().numpy())

    # convert the trajectory to a b-spline
    start_state = getZeroState()
    w_posBS_pred, w_yawBS_pred = am.f_trajAnd_w_State2wBS(predicted_traj, start_state)
    w_posBS_true, w_yawBS_true = am.f_trajAnd_w_State2wBS(true_traj, start_state)
    num_vectors_pos = 100
    num_vectors_yaw = 10
    time = np.linspace(w_posBS_pred.getT0(), w_posBS_pred.getTf(), num_vectors_pos)
    time_yaw = np.linspace(w_yawBS_pred.getT0(), w_yawBS_pred.getTf(), num_vectors_yaw)

    # plot 
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # plot the predicted trajectory
    ax.plot(w_posBS_true.pos_bs[0](time), w_posBS_true.pos_bs[1](time), w_posBS_true.pos_bs[2](time), lw=4, alpha=0.7, label='Expert')
    ax.plot(w_posBS_pred.pos_bs[0](time), w_posBS_pred.pos_bs[1](time), w_posBS_pred.pos_bs[2](time), lw=4, alpha=0.7, label='GNN')
    
    # plot the start and goal position
    ax.scatter(w_posBS_true.pos_bs[0](w_posBS_true.getT0()), w_posBS_true.pos_bs[1](w_posBS_true.getT0()), w_posBS_true.pos_bs[2](w_posBS_true.getT0()), s=100, c='green', marker='o', label='Start')
    ax.scatter(w_posBS_true.pos_bs[0](w_posBS_true.getTf()), w_posBS_true.pos_bs[1](w_posBS_true.getTf()), w_posBS_true.pos_bs[2](w_posBS_true.getTf()), s=100, c='red', marker='o', label='Goal')
    
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
        if idx == 0:
            ax.scatter(w_obs_pos[0], w_obs_pos[1], w_obs_pos[2], s=100, c='blue', marker='o', label='Obstacle')
        else:
            ax.scatter(w_obs_pos[0], w_obs_pos[1], w_obs_pos[2], s=100, c='blue', marker='o')
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
    ax.set_ylim(-2, 15)
    ax.set_zlim(-2, 15)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_aspect('equal')
    # fig.savefig('/home/kota/Research/deep-panther_ws/src/deep_panther/panther/matlab/figures/test_pos.png')
    plt.show()

    # plot the yaw trajectory
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(time_yaw, w_yawBS_true.pos_bs[0](time_yaw), lw=4, alpha=0.7, label='Expert')
    ax.plot(time_yaw, w_yawBS_pred.pos_bs[0](time_yaw), lw=4, alpha=0.7, label='GNN')
    ax.grid(True)
    ax.legend(loc='best')
    ax.set_xlabel('Time')
    ax.set_ylabel('Yaw')
    # fig.savefig('/home/kota/Research/deep-panther_ws/src/deep_panther/panther/matlab/figures/test_yaw.png')
    plt.show()


def main(dataset, true_trajs, device):

    """
    Main function
    """

    " ********************* DATA CONSTRUCTION ********************* "

    TRAIN_TEST_SPLIT = 0.8
    train_dataset = dataset[:int(TRAIN_TEST_SPLIT*len(dataset))]
    test_dataset = dataset[int(TRAIN_TEST_SPLIT*len(dataset)):]

    " ********************* CONSTRUCT THE MODEL ********************* "

    data = dataset[0] # for HGT model initialization purpose
    true_trajs = true_trajs.to(device)
    traj_size = len(true_trajs[0]) # for HGT model initialization purpose
    model = HGT(hidden_channels=64, out_channels=traj_size, num_heads=2, num_layers=2, data=data)
    model = model.to(device)
    optimizer = th.optim.Adam(model.parameters(), lr=0.2, weight_decay=5e-4)
    data = data.to(device)
    with th.no_grad():
        out = model(data.x_dict, data.edge_index_dict)
    
    " ********************* CREATE DATA LOADER ********************* "

    # data loader
    BATCH_SIZE = 2**6
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    " ********************* TRAINING ********************* "

    EPOCH_SIZE = 100

    # Epoch loop
    for epoch in range(1, EPOCH_SIZE):

        # Batch loop
        for batch_idx, batch in enumerate(train_loader):
            batch = batch.to(device)
            loss = train(model, batch, optimizer, true_trajs[batch_idx])

            if batch_idx % 50 == 0:
                print(f"\rEpoch: {epoch}/{EPOCH_SIZE} Batch num: {batch_idx}/{len(train_loader)} Training Loss: {loss.item()}", end="\n")

        # save the model
        th.save(model, "model.pt")

    " ********************* TESTING ********************* "

    total_loss = 0

    for data_idx, data in enumerate(test_dataset):

        data = data.to(device)
        true_traj = true_trajs[data_idx].clone().detach().unsqueeze(0).to(device)

        loss = test(model, data, true_traj)
        total_loss += loss.item()
    
    print("Testing Loss: ", total_loss/len(test_dataset))

if __name__ == "__main__":

    device = th.device('cuda' if th.cuda.is_available() else 'cpu')

    if IS_LOAD_F_OBS_NS:

        print("LOADING f_obs_ns")
        f_obs_ns = th.load("f_obs_ns.pt")

    else: 
     
        " ********************* LOAD DATA ********************* "

        # list npz files in the directory
        dir = "../evals/tmp_dagger/2/demos/round-000/"
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
        true_trajs = traj_data[0][0].clone().detach().unsqueeze(0).to(device) #TODO we just choose the first trajectory for now

        dataset_size = obs_data.shape[0] if IS_TRAIN_MODEL else 1
        for i in range(1, dataset_size):

            print(f"\rdata index: {i+1}/{obs_data.shape[0]}", end="")

            # get f_obs_n and f_traj
            f_obs_ns = th.cat((f_obs_ns, obs_data[i].clone().detach()), 0)
            true_trajs = th.cat((true_trajs, traj_data[i][0].clone().detach().unsqueeze(0)), 0)

        th.save(f_obs_ns, "f_obs_ns.pt")

    " ********************* GENERATE DATASET ********************* "
    
    print("\nGENERATING DATASET")
    dataset = generate_dataset(f_obs_ns, device)

    " ********************* MAIN (TRAIN & TEST) ********************* "

    print("TRAINING AND TESTING")
    if IS_TRAIN_MODEL:
        main(dataset, true_trajs, device)

    " ********************* VISUALIZE ********************* "

    if IS_VISUALIZE_TRAJ:
        for data_idx, data in enumerate(dataset):
            data = data.to('cpu')
            true_traj = true_trajs[data_idx].clone().detach().unsqueeze(0).to('cpu')
            f_obs_n = f_obs_ns[data_idx].clone().detach().unsqueeze(0).to('cpu')
            model = th.load("model.pt").to('cpu')
            visualize_trajectory(model, data, f_obs_n, true_traj)