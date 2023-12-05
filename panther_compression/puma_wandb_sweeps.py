#!/usr/bin/env python3

# HGT based imitation learning using wandb sweep
# ref: https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch/Organizing_Hyperparameter_Sweeps_in_PyTorch_with_W%26B.ipynb

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

from mpl_toolkits.mplot3d import Axes3D
import yaml
from scipy.optimize import linear_sum_assignment

import time
from statistics import mean
import copy
from random import random, shuffle
from colorama import init, Fore, Back, Style
# import py_panther
from joblib import Parallel, delayed

from torch_geometric.data import Data, HeteroData
from torch_geometric.nn import GATConv, to_hetero, HeteroConv, Linear, HGTConv
import networkx as nx
from torch_geometric.utils.convert import to_networkx
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
import wandb
import pprint

wandb.login()

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1', 'True'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0', 'False'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

" ********************* CLASS DEFINITION ********************* "

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
        # latent = th.cat([x for _, x in sorted(x_dict.items())], dim=-1)

        # get current state node
        latent = x_dict["current_state"]

        # add linear layers
        for lin in self.lins:
            latent = lin(latent)
        
        before_shape = latent.shape
        # reshape latent to be [num_of_trajs_per_replan, out_channels]
        output = th.reshape(latent, (before_shape[0],) + (self.num_of_trajs_per_replan, self.out_channels))

        return output

" ********************* FUNCTION DEFINITION ********************* "

" ********************* BUILD DATASET ********************* "

def build_dataset(batch_size, device):

    " ********************* LOAD DATA ********************* "

    # list npz files in the directory
    dirs = "../evals/tmp_dagger/2/demos/"
    # dirs = "../evals_tmp/tmp_dagger/2/demos/"
    # loop over dirs
    obs_data = th.tensor([]).to(device)
    traj_data = th.tensor([]).to(device)
    # list dirs in dirs
    dirs = subfolders = [ f.path for f in os.scandir(dirs) if f.is_dir() ]
    for dir in dirs:

        print("READING DATA FROM: ", dir, "\n")

        files = os.listdir(dir)
        files = [dir + '/' + file for file in files if file.endswith('.npz')]

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

    # dataset_size = obs_data.shape[0]
    dataset_size = 1000
    for i in range(1, dataset_size):

        print(f"\rdata index: {i+1}/{obs_data.shape[0]}", end="")

        # get f_obs_n and f_traj
        f_obs_ns = th.cat((f_obs_ns, obs_data[i].clone().detach()), 0)
        true_trajs = th.cat((true_trajs, traj_data[i].clone().detach().unsqueeze(0)), 0)

    th.save(f_obs_ns, "f_obs_ns.pt")

    " ********************* GENERATE DATASET ********************* "
    
    print("GENERATING DATASET")
    dataset = generate_dataset(f_obs_ns, true_trajs, device)
    
    " ********************* CREATE DATA LOADER ********************* "

    # data loader
    train_loader = DataLoader(dataset, batch_size, shuffle=True)

    return train_loader

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
        dist_goal_obs = th.tensor([np.linalg.norm(feature_vector_for_goal[0][:3] - feature_vector_for_obs[0][:3])], dtype=th.double).to(device)

        " ********************* MAKE A DATA OBJECT FOR HETEROGENEUS GRAPH ********************* "

        data = HeteroData()

        # add nodes
        data["current_state"].x = feature_vector_for_current_state # [number of "current_state" nodes, size of feature vector]
        data["goal_state"].x = feature_vector_for_goal # [number of "goal_state" nodes, size of feature vector]
        data["observation"].x = feature_vector_for_obs # [number of "observation" nodes, size of feature vector]

        # add edges
        data["current_state", "dist_current_state_to_goal_state", "goal_state"].edge_index = th.tensor([
                                                                                        [0],  # idx of source nodes (current state)
                                                                                        [0],  # idx of target nodes (goal state)
                                                                                        ],dtype=th.int64)
        data["current_state", "dist_current_state_to_observation", "observation"].edge_index = th.tensor([
                                                                                        [0],  # idx of source nodes (current state)
                                                                                        [0],  # idx of target nodes (observation)
                                                                                        ],dtype=th.int64)
        data["observation", "dist_obs_to_goal", "goal_state"].edge_index = th.tensor([
                                                                                        [0],  # idx of source nodes (observation)
                                                                                        [0],  # idx of target nodes (goal state)
                                                                                        ],dtype=th.int64)
        data["observation", "dist_observation_to_current_state", "current_state"].edge_index = th.tensor([
                                                                                        [0],  # idx of source nodes (observation)
                                                                                        [0],  # idx of target nodes (current state)
                                                                                        ],dtype=th.int64)
        data["goal_state", "dist_goal_state_to_current_state", "current_state"].edge_index = th.tensor([
                                                                                        [0],  # idx of source nodes (goal state)
                                                                                        [0],  # idx of target nodes (current state)
                                                                                        ],dtype=th.int64)
        data["goal_state", "dist_to_obs", "observation"].edge_index = th.tensor([
                                                                                        [0],  # idx of source nodes (goal state)
                                                                                        [0],  # idx of target nodes (observation)
                                                                                        ],dtype=th.int64)

        # add edge weights
        data["current_state", "dist_current_state_to_goal_state", "goal_state"].edge_attr = dist_current_state_goal
        data["current_state", "dist_current_state_to_observation", "observation"].edge_attr = dist_current_state_obs
        data["observation", "dist_obs_to_goal", "goal_state"].edge_attr = dist_goal_obs
        # make it undirected
        data["observation", "dist_observation_to_current_state", "current_state"].edge_attr = dist_current_state_obs
        data["goal_state", "dist_goal_state_to_current_state", "current_state"].edge_attr = dist_current_state_goal
        data["goal_state", "dist_goal_to_obs", "observation"].edge_attr = dist_goal_obs

        # add ground truth trajectory
        data.true_traj = true_trajs[i].clone().detach().unsqueeze(0).to(device)

        # convert the data to the device
        data = data.to(device)
        # append data to the dataset
        dataset.append(data)

    " ********************* RETURN ********************* "

    return dataset

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

" ********************* TRAIN EPOCH ********************* "

def train_epoch(model, loader, optimizer):

    """
    This function trains the model and return the loss
    """

    cumu_loss = 0

    for batch in loader:

        model.train() # set the model.training to be True
        optimizer.zero_grad() # reset the gradient of all variables
        predicted_traj = model(batch.x_dict, batch.edge_index_dict)

        loss, stats_dict = calculate_loss(predicted_traj, batch.true_traj, 15, 6, 1, device)
        
        # compute the gradient
        loss.backward()

        # update the parameters
        optimizer.step()

        # wandb log
        wandb.log({"batch_loss": loss})
        wandb.log({"batch_pos_loss": stats_dict["pos_loss"]})
        wandb.log({"batch_yaw_loss": stats_dict["yaw_loss"]})
        wandb.log({"batch_time_loss": stats_dict["time_loss"]})

        cumu_loss += loss

    return cumu_loss / len(loader)

def train(config=None):

    """
    Main function
    """

    " ********************* TRAINING ********************* "

    with wandb.init(config=config):
        config = wandb.config

        # hyper parameters
        hidden_channels = config.hidden_channels
        num_heads = config.num_heads
        num_layers = config.num_layers
        group = config.group
        num_linear_layers = config.num_linear_layers
        linear_hidden_channels = config.linear_hidden_channels
        epochs = config.epochs
        batch_size = config.batch_size
        num_of_trajs_per_replan = 10

        # build dataset
        train_loader = build_dataset(batch_size, device)

        # model 
        model = HGT(hidden_channels=hidden_channels, out_channels=22, 
                num_heads=num_heads, num_layers=num_layers, group=group, num_linear_layers=num_linear_layers, 
                linear_hidden_channels=linear_hidden_channels, num_of_trajs_per_replan=num_of_trajs_per_replan, data=train_loader.dataset[0])
        model = model.to(device)

        # optimizer
        optimizer = th.optim.Adam(model.parameters(), lr=1e-3)

        # train
        for epoch in range(epochs):

            print(f"\nEpoch: {epoch}/{epochs}")

            # Batch loop
            avg_loss = train_epoch(model, train_loader, optimizer)
            wandb.log({"loss": avg_loss, "epoch": epoch})

" ********************* MAIN  ********************* "
# REF: https://www.google.com/search?q=how+to+run+same+sweep+on+multiple+machine+weights+%26+biases&oq=how+to+run+same+sweep+on+multiple+machine+weights+%26+biases&gs_lcrp=EgZjaHJvbWUyBggAEEUYOdIBCTE2NTgzajBqN6gCALACAA&sourceid=chrome&ie=UTF-8#fpstate=ive&vld=cid:d9189e0a,vid:WZvG6hwxUEw,st:0

if __name__ == "__main__":

    # args
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--count', type=int, default=100, help='count for wandb sweep')
    args = parser.parse_args()

    device = th.device('cuda' if th.cuda.is_available() else 'cpu')

    # default_config
    default_config = {
        'hidden_channels': 64,
        'num_heads': 2,
        'num_layers': 2,
        'group': "mean",
        'num_linear_layers': 2,
        'linear_hidden_channels': 64,
        'batch_size': 32,
        'epochs': 20,
    }

    # wandb
    train(default_config)
