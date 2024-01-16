#!/usr/bin/env python3

# diffusion policy import
import torch
import os
import time
import numpy as np
import argparse
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler

# torch import
import torch as th
from torch.utils.data import Dataset

# gnn import
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader as GNNDataLoader

# visualization import
import matplotlib.pyplot as plt
from tqdm import tqdm

# calculate loss
from scipy.optimize import linear_sum_assignment

# network utils import
from network_utils import ConditionalUnet1D

# dataset for dictornary

class DictDataset(Dataset):

    def __init__(self, obs, acts):
        self.obs = obs
        self.acts = acts
        self.reward = None

    def __getitem__(self, index):
        if self.reward is None:
            return {'obs': self.obs[index], 'acts': self.acts[index]}
        else:
            return {'obs': self.obs[index], 'acts': self.acts[index], 'reward': self.reward[index]}
    
    def __len__(self):
        return len(self.obs)
    
    def add_reward(self, reward):
        self.reward = reward

def get_dataloader_training(dataset_training, **kwargs):

    """
    This function generates a dataloader for training
    """

    # unpack kwargs
    batch_size = kwargs.get('batch_size')
    en_network_type = kwargs.get('en_network_type')
    device = kwargs.get('device')

    if en_network_type == 'gnn':
        # create dataloader for GNN
        dataloader_training = GNNDataLoader(
            dataset_training,
            batch_size=batch_size,                                      # if batch_size is less than 256, then CPU is faster than GPU on my computer
            shuffle=False if str(device)=='cuda' else True,             # shuffle True causes error Expected a 'cuda' str(device) type for generator but found 'cpu' https://github.com/dbolya/yolact/issues/664#issuecomment-878241658
            num_workers=0 if str(device)=='cuda' else 16 ,              # if we wanna use cuda, need to set num_workers=0
            pin_memory=False if str(device)=='cuda' else True,          # accelerate cpu-gpu transfer
            persistent_workers=False if str(device)=='cuda' else True,  # if we wanna use cuda, need to set False
        )
    elif device == th.device('cpu'):
        dataloader_training = th.utils.data.DataLoader(
            dataset_training,
            batch_size=batch_size,
            num_workers=16,
            shuffle=True,
            # accelerate cpu-gpu transfer
            pin_memory=True,
            # don't kill worker process afte each epoch
            persistent_workers=True
        )
    else:
        dataloader_training = th.utils.data.DataLoader(
            dataset_training,
            batch_size=batch_size,
        )

    return dataloader_training

def str2bool(v):
    """
    This function converts string to boolean (mainly used for argparse)
    """

    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1', 'True'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0', 'False'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def setup_diffusion(**kwargs):

    # unpack kwargs
    action_dim = kwargs.get('action_dim')
    obs_dim = kwargs.get('obs_dim')
    obs_horizon = kwargs.get('obs_horizon')
    num_trajs = kwargs.get('num_trajs')
    num_diffusion_iters = kwargs.get('num_diffusion_iters')
    en_network_type = kwargs.get('en_network_type')
    use_gnn = en_network_type == 'gnn'
    device = kwargs.get('device')
    scheduler_type = kwargs.get('scheduler_type')
    dataset = kwargs['datasets_loader']['dataset_training']

    # create network object
    policy = ConditionalUnet1D(**kwargs)

    # example inputs
    noised_action = torch.randn((1, num_trajs, action_dim)).to(device)
    obs = torch.zeros((1, 1, obs_dim)).to(device)

    # naction's num_trajs needs to be a multiple of 2 so let's make it 8 for now
    # noised_action = noised_action[:, :8, :] # (B, num_trajs, action_dim)

    # example diffusion iteration
    diffusion_iter = torch.zeros((1,)).to(device)

    # the noise prediction network
    # takes noisy action, diffusion iteration and observation as input
    # predicts the noise added to action
    # you need to run policy to initialize the network https://stackoverflow.com/questions/75550160/how-to-set-requires-grad-to-false-freeze-pytorch-lazy-layers
    x_dict = dataset[0].x_dict if use_gnn else None
    edge_index_dict = dataset[0].edge_index_dict if use_gnn else None

    _ = policy(
        sample=noised_action,
        timestep=diffusion_iter,
        global_cond=obs,
        x_dict=x_dict,
        edge_index_dict=edge_index_dict)

    # for this demo, we use DDPMScheduler
    if scheduler_type == 'ddpm':
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=num_diffusion_iters,
            # the choise of beta schedule has big impact on performance
            # we found squared cosine works the best
            beta_schedule='squaredcos_cap_v2',
            # clip output to [-1,1] to improve stability
            clip_sample=True,
            # our network predicts noise (instead of denoised action)
            prediction_type='epsilon'
        )
    elif scheduler_type == 'ddim':
        noise_scheduler = DDIMScheduler(
            num_train_timesteps=num_diffusion_iters,
            # the choise of beta schedule has big impact on performance
            # we found squared cosine works the best
            beta_schedule='squaredcos_cap_v2',
            # clip output to [-1,1] to improve stability
            clip_sample=True,
            # our network predicts noise (instead of denoised action)
            prediction_type='epsilon'
        )
    elif scheduler_type == 'dpm-multistep':
        # DPMSolverMultistepScheduler
        noise_scheduler = DPMSolverMultistepScheduler(
            num_train_timesteps=num_diffusion_iters,  
            beta_schedule='squaredcos_cap_v2',
            # clip_sample=True,
            prediction_type='epsilon'
        )

    # device transfer
    _ = policy.to(device)

    return policy, noise_scheduler

def create_pair_obs_act(**kwargs):

    """
    Create dataset from npz files in dirs
    
    @param: kwargs: max_num_training_demos, percentage_training, percentage_eval, percentage_test, data_dir, device, obs_type
    @return dataset: TensorDataset
    """

    # unpack kwargs
    max_num_training_demos = kwargs.get('max_num_training_demos')
    percentage_training = kwargs.get('percentage_training')
    percentage_eval = kwargs.get('percentage_eval')
    percentage_test = kwargs.get('percentage_test')
    total_num_demos = int(max_num_training_demos / percentage_training)
    data_dir = kwargs.get('data_dir')
    device = kwargs.get('device')
    obs_type = kwargs.get('obs_type')
    en_network_type = kwargs.get('en_network_type')
    num_trajs = kwargs.get('num_trajs')

    # initialize dataset
    obs_data = {}
    traj_data = {}

    # list dirs in dirs
    dirs = [ f.path for f in os.scandir(data_dir) if f.is_dir() ]

    # loop over dirs
    idx = 0
    for dir in dirs: # round-### directories

        # get all the npz files in the dir
        files = [dir + '/' + file for file in os.listdir(dir) if file.endswith('.npz')]
        
        # loop over files
        for file in files:
            
            # load data
            data = np.load(file)
            obs = data['obs'][:-1] # remove the last observation (since it is the latest observation and we don't have the action for it)
            acts = data['acts'] # actions

            # append to the data
            if obs_type == 'last' or en_network_type == 'gnn': # use the last observation 
                
                # one observation and one action
                for i in range(obs.shape[0]):
                    
                    obs_data[idx] = th.from_numpy(obs[i, :, :]).float().to(device) #(num_pairs, 1, obs_dim)
                    traj_data[idx] = th.from_numpy(acts[i, :num_trajs, :]).float().to(device) #(num_pairs, num_trajs, action_dim)
                    idx += 1

            elif obs_type == 'history': # use the history of observation
                
                obs_data[idx] = th.from_numpy(obs).float().to(device) #(num_pairs, obs_horizon, obs_dim)
                traj_data[idx] = th.from_numpy(acts).float().to(device) #(num_pairs, num_trajs, action_dim)
                idx += 1

            if idx >= total_num_demos:
                break
        else:
            continue
        break

    # print out the total data size
    print(f"data size: {len(obs_data.keys())}")

    # split the dataset dictionary into training, eval, and test
    dataset_obs_training, dataset_acts_training, dataset_obs_eval, dataset_acts_eval, dataset_obs_test, dataset_acts_test =  {}, {}, {}, {}, {}, {}
    for idx, key in enumerate(obs_data.keys()):
        if idx < percentage_training * len(obs_data.keys()):
            dataset_obs_training[key] = obs_data[key]
            dataset_acts_training[key] = traj_data[key]
        elif idx < (percentage_training + percentage_eval) * len(obs_data.keys()):
            dataset_obs_eval[key] = obs_data[key]
            dataset_acts_eval[key] = traj_data[key]
        else:
            dataset_obs_test[key] = obs_data[key]
            dataset_acts_test[key] = traj_data[key]

    # dataset_obs_eval[key] should start at 0
    first_key_obs_eval = list(dataset_obs_eval.keys())[0]
    first_key_acts_eval = list(dataset_acts_eval.keys())[0]
    dataset_obs_eval = {key - first_key_obs_eval: dataset_obs_eval[key] for key in dataset_obs_eval.keys()}
    dataset_acts_eval = {key - first_key_acts_eval: dataset_acts_eval[key] for key in dataset_acts_eval.keys()}

    # dataset_obs_test[key] should start at 0
    first_key_obs_test = list(dataset_obs_test.keys())[0]
    first_key_acts_test = list(dataset_acts_test.keys())[0]
    dataset_obs_test = {key - first_key_obs_test: dataset_obs_test[key] for key in dataset_obs_test.keys()}
    dataset_acts_test = {key - first_key_acts_test: dataset_acts_test[key] for key in dataset_acts_test.keys()}

    return dataset_obs_training, dataset_acts_training, dataset_obs_eval, dataset_acts_eval, dataset_obs_test, dataset_acts_test

def create_gnn_dataset_from_obs_and_acts(dataset_obs, dataset_acts):

    """
    This function generates a dataset for GNN
    """

    # initialize dataset
    dataset = []

    # check if the length of dataset_obs and dataset_acts are the same
    assert dataset_obs.shape[0] == dataset_acts.shape[0], "the length of dataset_obs and dataset_acts should be the same"

    # loop over the dataset
    for data_obs, data_acts in zip(dataset_obs, dataset_acts):

        " ********************* GET NODES ********************* "

        # nodes you need for GNN
        # dataset = [f_v, f_a, yaw_dot, f_g,  [f_ctrl_pts_o0], bbox_o0, [f_ctrl_pts_o1], bbox_o1 ,...]
        # dim         3    3      1      3          30            3          30              3 
        # 0. current state 
        # 1. goal state 
        # 2. observation
        # In the current setting f_obs is a realative state from the current state so we pass f_v, f_z, yaw_dot to the current state node

        " ********************* MAKE A DATA OBJECT FOR HETEROGENEUS GRAPH ********************* "

        data = HeteroData()

        # add nodes
        # get num of obst
        num_of_obst = int(len(data_obs[10:])/33)

        feature_vector_for_current_state = data_obs[0:7]
        feature_vector_for_goal = data_obs[7:10]
        feature_vector_for_obs = data_obs[10:]

        dist_current_state_goal = np.linalg.norm(feature_vector_for_goal[:3].to('cpu').numpy())

        dist_current_state_obs = []
        dist_goal_obs = []
        for j in range(num_of_obst):
            dist_current_state_obs.append(np.linalg.norm(feature_vector_for_obs[33*j:33*j+3].to('cpu').numpy()))
            dist_goal_obs.append(np.linalg.norm((feature_vector_for_goal[:3] - feature_vector_for_obs[33*j:33*j+3]).to('cpu').numpy()))

        dist_obst_to_obst = []
        for j in range(num_of_obst):
            for k in range(num_of_obst):
                if j != k:
                    dist_obst_to_obst.append((np.linalg.norm((feature_vector_for_obs[33*j:33*j+3] - feature_vector_for_obs[33*k:33*k+3]).to('cpu').numpy())))


        " ********************* MAKE A DATA OBJECT FOR HETEROGENEUS GRAPH ********************* "

        # add nodes
        data["current_state"].x = feature_vector_for_current_state.unsqueeze(0).float()
        data["goal_state"].x = feature_vector_for_goal.unsqueeze(0).float()
        data["observation"].x = th.stack([feature_vector_for_obs[33*j:33*(j+1)] for j in range(num_of_obst)], dim=0).float()

        # add edges
        if num_of_obst == 2:
            data["current_state", "dist_current_state_to_goal_state", "goal_state"].edge_index = th.tensor([
                                                                                        [0],  # idx of source nodes (current state)
                                                                                        [0],  # idx of target nodes (goal state)
                                                                                        ],dtype=th.int64)
            data["current_state", "dist_current_state_to_observation", "observation"].edge_index = th.tensor([
                                                                                            [0],  # idx of source nodes (current state)
                                                                                            [0],  # idx of target nodes (observation)
                                                                                            ],dtype=th.int64)
            data["observation", "dist_obs_to_goal", "goal_state"].edge_index = th.tensor([
                                                                                            [0, 0],  # idx of source nodes (observation)
                                                                                            [0, 1],  # idx of target nodes (goal state)
                                                                                            ],dtype=th.int64)
            data["observation", "dist_observation_to_current_state", "current_state"].edge_index = th.tensor([
                                                                                            [0, 0],  # idx of source nodes (observation)
                                                                                            [0, 1],  # idx of target nodes (current state)
                                                                                            ],dtype=th.int64)
            data["goal_state", "dist_goal_state_to_current_state", "current_state"].edge_index = th.tensor([
                                                                                            [0],  # idx of source nodes (goal state)
                                                                                            [0],  # idx of target nodes (current state)
                                                                                            ],dtype=th.int64)
            data["goal_state", "dist_to_obs", "observation"].edge_index = th.tensor([
                                                                                            [0, 0],  # idx of source nodes (goal state)
                                                                                            [0, 1],  # idx of target nodes (observation)
                                                                                            ],dtype=th.int64)

        elif num_of_obst == 1:

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
        data.acts = data_acts.unsqueeze(0)
        
        # add observation
        data.obs = data_obs.unsqueeze(0).unsqueeze(0)

        # append data to the dataset
        dataset.append(data)

    " ********************* RETURN ********************* "

    return dataset

def create_dataset(**kwargs):
    
    """
    Create dataset from npz files in dirs
    @param dirs: directory containing npz files
    @param device: device to transfer data to
    @return dataset: DictDataset
    """

    # unpack kwargs
    en_network_type = kwargs.get('en_network_type')

    # get obs and acts
    dataset_obs_training, dataset_acts_training, dataset_obs_eval, dataset_acts_eval, dataset_obs_test, dataset_acts_test = create_pair_obs_act(**kwargs)

    if en_network_type == 'gnn':

        # conver the dataset from dict to tensor (we only support last observation for now)
        dataset_obs_training = th.stack(list(dataset_obs_training.values()), dim=0).squeeze(1)
        dataset_acts_training = th.stack(list(dataset_acts_training.values()), dim=0).squeeze(1)
        dataset_obs_eval = th.stack(list(dataset_obs_eval.values()), dim=0).squeeze(1)
        dataset_acts_eval = th.stack(list(dataset_acts_eval.values()), dim=0).squeeze(1)
        dataset_obs_test = th.stack(list(dataset_obs_test.values()), dim=0).squeeze(1)
        dataset_acts_test = th.stack(list(dataset_acts_test.values()), dim=0).squeeze(1)

    # create dataset
    dataset_training = create_gnn_dataset_from_obs_and_acts(dataset_obs_training, dataset_acts_training) if en_network_type == 'gnn' else DictDataset(dataset_obs_training, dataset_acts_training)
    dataset_eval = create_gnn_dataset_from_obs_and_acts(dataset_obs_eval, dataset_acts_eval) if en_network_type == 'gnn' else DictDataset(dataset_obs_eval, dataset_acts_eval)
    dataset_test = create_gnn_dataset_from_obs_and_acts(dataset_obs_test, dataset_acts_test) if en_network_type == 'gnn' else DictDataset(dataset_obs_test, dataset_acts_test)

    return dataset_training, dataset_eval, dataset_test

def get_nactions(policy, noise_scheduler, dataset, is_visualize=False, **kwargs):

    """
    This function generates a predicted trajectory
    """

    # unpack kwargs
    num_trajs = kwargs.get('num_trajs')
    num_diffusion_iters = kwargs.get('num_diffusion_iters')
    action_dim = kwargs.get('action_dim')
    use_gnn = kwargs.get('en_network_type') == 'gnn'
    device = kwargs.get('device')
    num_eval = kwargs.get('num_eval')

    # set model to evaluation mode
    policy.eval()

    # set batch size to 1
    B = 1

    # num of data to load
    num_data_to_load = min(len(dataset), num_eval)

    # loop over the dataset
    print("start denoising in get_nactions")
    expert_actions, nactions, nobses, times = [], [], [], []

    for dataset_idx in tqdm(range(num_data_to_load), desc="dataset idx in denoising", leave=False):

        # stack the last obs_horizon (2) number of observations
        nobs = dataset[dataset_idx].obs if use_gnn else dataset[dataset_idx]['obs'].unsqueeze(0) # (B, global_cond_dim)
        expert_action = dataset[dataset_idx].acts if use_gnn else dataset[dataset_idx]['acts'].unsqueeze(0) # (B, num_trajs, action_dim)

        # infer action
        with torch.no_grad():

            # reshape observation to (B,obs_horizon*obs_dim)
            obs_cond = nobs.flatten(start_dim=2)
            # initialize action from Guassian noise
            noisy_action = torch.randn(
                (B, num_trajs, action_dim), device=device)
            naction = noisy_action

            # init scheduler
            noise_scheduler.set_timesteps(num_diffusion_iters)

            # start timer
            start_time = time.time()
            # for k in tqdm(noise_scheduler.timesteps, desc="diffusion iter k"):
            for k in noise_scheduler.timesteps:
                # predict noise

                if use_gnn:
                    x_dict = dataset[dataset_idx].x_dict
                    edge_index_dict = dataset[dataset_idx].edge_index_dict
                    noise_pred = policy(
                        sample=naction,
                        timestep=k,
                        global_cond=obs_cond,
                        x_dict=x_dict,
                        edge_index_dict=edge_index_dict
                    )
                else:

                    noise_pred = policy(
                        sample=naction,
                        timestep=k,
                        global_cond=obs_cond
                    )

                # inverse diffusion step (remove noise)
                naction = noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=k,
                    sample=naction
                ).prev_sample

            # end timer
            end_time = time.time()
            times.append(end_time - start_time)

        # append to the list
        expert_actions.append(expert_action.cpu().numpy())
        nactions.append(naction.cpu().numpy())
        nobses.append(nobs.cpu().numpy())

        if is_visualize:
            # print out the computation time
            print("computation time: ", np.mean(times))
            # visualize trajectory
            visualize_trajectory(expert_action.cpu().numpy(), naction.cpu().numpy(), nobs, dataset_idx)

    if not is_visualize: # used get_nactions used in evaluation
        return expert_actions, nactions, nobses, np.mean(times)

def plot_obstacles(start_state, f_obs, ax):

    w_obs_poses = []
    p0 = start_state.w_pos[0] # careful here: we assume the agent pos is at the origin - as the agent moves we expect the obstacles shift accordingly
    
    # extract each obstacle trajs
    num_obst = int(len(f_obs[0][10:])/33)
    for i in range(num_obst):
        
        # get each obstacle's trajectory
        f_obs_each = f_obs[0][10+33*i:10+33*(i+1)]
    
        # get each obstacle's poses in that trajectory in the world frame
        for i in range(int(len(f_obs_each[:-3])/3)):
            w_obs_poses.append((f_obs_each[3*i:3*i+3] - p0.T).tolist())

        # get each obstacle's bbox
        bbox = f_obs_each[-3:]

        # plot the bbox
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
            if idx == 0 and i == 0:
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

def plot_pos(actions, am, ax, label, num_vectors_pos, num_vectors_yaw, start_state):

    # color (expert: green, student: orange)
    color = 'green' if label == 'Expert' else 'orange'

    for action_idx in range(actions.shape[0]):

        action = actions[action_idx, :].reshape(1, -1)
        traj = am.denormalizeTraj(action)
        
        # convert the trajectory to a b-spline
        w_posBS, w_yawBS = am.f_trajAnd_w_State2wBS(traj, start_state)
        time_pos = np.linspace(w_posBS.getT0(), w_posBS.getTf(), num_vectors_pos)
        time_yaw = np.linspace(w_yawBS.getT0(), w_yawBS.getTf(), num_vectors_yaw)

        # plot the predicted trajectory
        if action_idx == 0 and label == 'Expert':
            # plot the start and goal position
            ax.scatter(w_posBS.pos_bs[0](w_posBS.getT0()), w_posBS.pos_bs[1](w_posBS.getT0()), w_posBS.pos_bs[2](w_posBS.getT0()), s=100, c='pink', marker='o', label='Start')
        else:
            label = None
        
        # plot trajectory
        ax.plot(w_posBS.pos_bs[0](time_pos), w_posBS.pos_bs[1](time_pos), w_posBS.pos_bs[2](time_pos), lw=4, alpha=0.7, label=label, c=color)
        # plot yaw direction
        ax.quiver(w_posBS.pos_bs[0](time_yaw), w_posBS.pos_bs[1](time_yaw), w_posBS.pos_bs[2](time_yaw), np.cos(w_yawBS.pos_bs[0](time_yaw)), np.sin(w_yawBS.pos_bs[0](time_yaw)), np.zeros_like(w_yawBS.pos_bs[0](time_yaw)), length=0.5, normalize=True, color='red')

        action_idx += 1
        if action_idx > len(actions):
            break

def plot_yaw(actions, am, ax, label, num_vectors_yaw, start_state):

    # color (expert: green, student: orange)
    color = 'green' if label == 'Expert' else 'orange'

    for action_idx in range(actions.shape[0]):

        action = actions[action_idx, :].reshape(1, -1)
        traj = am.denormalizeTraj(action)

        # convert the trajectory to a b-spline
        _, w_yawBS = am.f_trajAnd_w_State2wBS(traj, start_state)
        time_yaw = np.linspace(w_yawBS.getT0(), w_yawBS.getTf(), num_vectors_yaw)

        if not action_idx == 0:
            label = None
        
        ax.plot(time_yaw, w_yawBS.pos_bs[0](time_yaw), lw=4, alpha=0.7, label=label, c=color)

        action_idx += 1
        if action_idx > len(actions):
            break

def visualize_trajectory(expert_action, action_pred, nobs, image_idx):

    from compression.utils.other import ObservationManager, ActionManager, getZeroState

    # interpolation parameters
    num_vectors_pos = 100
    num_vectors_yaw = 10

    # get start state
    start_state = getZeroState() # TODO (hardcoded)

    # get action and observation manager for normalization and denormalization
    am = ActionManager() # get action manager
    om = ObservationManager() # get observation manager

    # plot 
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(211, projection='3d')

    expert_action = expert_action.squeeze(0)
    action_pred = action_pred.squeeze(0)
    nobs = nobs.squeeze(0)

    # plot pos trajectories
    plot_pos(expert_action, am, ax, 'Expert', num_vectors_pos, num_vectors_yaw, start_state)
    plot_pos(action_pred, am, ax, 'Diffusion', num_vectors_pos, num_vectors_yaw, start_state)

    # plot the goal
    f_obs = om.denormalizeObservation(nobs.to('cpu').numpy())
    ax.scatter(f_obs[0][7], f_obs[0][8], f_obs[0][9], s=100, c='red', marker='*', label='Goal')
    
    # plot the obstacles
    # get w pos of the obstacles
    plot_obstacles(start_state, f_obs, ax)

    ax.grid(True)
    ax.legend(loc='best')
    ax.set_xlim(-2, 15)
    ax.set_ylim(-4, 4)
    ax.set_zlim(-5, 5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_aspect('equal')

    # plot the yaw trajectory
    ax = fig.add_subplot(212)

    # plot yaw trajectories
    plot_yaw(expert_action, am, ax, 'Expert', num_vectors_yaw, start_state)
    plot_yaw(action_pred, am, ax, 'Diffusion', num_vectors_yaw, start_state)

    ax.grid(True)
    ax.legend(loc='best')
    ax.set_xlabel('Time')
    ax.set_ylabel('Yaw')
    # fig.savefig(f'/media/jtorde/T7/gdp/pngs/image_{image_idx}.png')
    plt.show()

def visualize(policy, noise_scheduler, **kwargs):

    """
    This function visualizes the predicted trajectory
    """

    # unpack
    dataset = kwargs.get('datasets_loader')['dataset_test']

    # get expert actions and predicted actions(nactions)
    get_nactions(policy, noise_scheduler, dataset, is_visualize=True, **kwargs)

def calculate_deep_panther_loss(batch, policy, **kwargs):
    """Calculate the supervised learning loss used to train the behavioral clone.

    Args:
        obs: The observations seen by the expert. If this is a Tensor, then
            gradients are detached first before loss is calculated.
        acts: The actions taken by the expert. If this is a Tensor, then its
            gradients are detached first before loss is calculated.

    Returns:
        loss: The supervised learning loss for the behavioral clone to optimize.
        stats_dict: Statistics about the learning process to be logged.

    """

    # unpack
    en_network_type = kwargs.get('en_network_type')
    yaw_loss_weight = kwargs.get('yaw_loss_weight')
    obs_type = kwargs.get('obs_type')

    # get the observation and action
    obs = batch['obs']
    acts = batch['acts']

    # (TODO hardcoded)
    traj_size_pos_ctrl_pts = 15
    traj_size_yaw_ctrl_pts = 6

    # set policy to train mode
    policy.train()

    # get the predicted action
    pred_acts = policy(obs, batch.x_dict, batch.edge_index_dict) if en_network_type == 'gnn' else policy(obs)

    # get size 
    num_of_traj_per_action=list(acts.shape)[1] #acts.shape is [batch size, num_traj_action, size_traj]
    batch_size=list(acts.shape)[0] #acts.shape is [batch size, num_of_traj_per_action, size_traj]

    # initialize the distance matrix
    distance_matrix= th.zeros(batch_size, num_of_traj_per_action, num_of_traj_per_action)
    distance_pos_matrix= th.zeros(batch_size, num_of_traj_per_action, num_of_traj_per_action) 
    distance_yaw_matrix= th.zeros(batch_size, num_of_traj_per_action, num_of_traj_per_action) 
    distance_time_matrix= th.zeros(batch_size, num_of_traj_per_action, num_of_traj_per_action)
    distance_pos_matrix_within_expert= th.zeros(batch_size, num_of_traj_per_action, num_of_traj_per_action)

    #Expert --> i
    #Student --> j
    for i in range(num_of_traj_per_action):
        for j in range(num_of_traj_per_action):

            expert_i=       acts[:,i,:].float(); #All the elements
            student_j=      pred_acts[:,j,:].float() #All the elements

            expert_pos_i=   acts[:,i,0:traj_size_pos_ctrl_pts].float()
            student_pos_j=  pred_acts[:,j,0:traj_size_pos_ctrl_pts].float()

            expert_yaw_i=   acts[:,i,traj_size_pos_ctrl_pts:(traj_size_pos_ctrl_pts+traj_size_yaw_ctrl_pts)].float()
            student_yaw_j=  pred_acts[:,j,traj_size_pos_ctrl_pts:(traj_size_pos_ctrl_pts+traj_size_yaw_ctrl_pts)].float()

            expert_time_i=       acts[:,i,-1:].float(); #Time. Note: Is you use only -1 (instead of -1:), then distance_time_matrix will have required_grad to false
            student_time_j=      pred_acts[:,j,-1:].float() #Time. Note: Is you use only -1 (instead of -1:), then distance_time_matrix will have required_grad to false

            distance_matrix[:,i,j]=th.mean(th.nn.MSELoss(reduction='none')(expert_i, student_j), dim=1)
            distance_pos_matrix[:,i,j]=th.mean(th.nn.MSELoss(reduction='none')(expert_pos_i, student_pos_j), dim=1)
            distance_yaw_matrix[:,i,j]=th.mean(th.nn.MSELoss(reduction='none')(expert_yaw_i, student_yaw_j), dim=1)
            distance_time_matrix[:,i,j]=th.mean(th.nn.MSELoss(reduction='none')(expert_time_i, student_time_j), dim=1)

            #This is simply to delete the trajs from the expert that are repeated
            expert_pos_j=   acts[:,j,0:traj_size_pos_ctrl_pts].float()
            distance_pos_matrix_within_expert[:,i,j]=th.mean(th.nn.MSELoss(reduction='none')(expert_pos_i, expert_pos_j), dim=1)

    is_repeated=th.zeros(batch_size, num_of_traj_per_action, dtype=th.bool)

    for i in range(num_of_traj_per_action):
        for j in range(i+1, num_of_traj_per_action):
            is_repeated[:,j]=th.logical_or(is_repeated[:,j], th.lt(distance_pos_matrix_within_expert[:,i,j], 1e-7))

    assert distance_matrix.requires_grad==True
    assert distance_pos_matrix.requires_grad==True
    assert distance_yaw_matrix.requires_grad==True
    assert distance_time_matrix.requires_grad==True

    #Option 1: Solve assignment problem
    A_matrix=th.zeros(batch_size, num_of_traj_per_action, num_of_traj_per_action)

    for index_batch in range(batch_size):         

        cost_matrix=distance_pos_matrix[index_batch,:,:]
        map2RealRows=np.array(range(num_of_traj_per_action))
        map2RealCols=np.array(range(num_of_traj_per_action))

        rows_to_delete=[]
        for i in range(num_of_traj_per_action): #for each row (expert traj)
            if(is_repeated[index_batch,i]==True): 
                rows_to_delete.append(i) #Delete that row

        cost_matrix=cost_matrix[is_repeated[index_batch,:]==False]   #np.delete(cost_matrix_numpy, rows_to_delete, axis=0)
        cost_matrix_numpy=cost_matrix.cpu().detach().numpy()

        map2RealRows=np.delete(map2RealRows, rows_to_delete, axis=0)
        row_indexes, col_indexes = linear_sum_assignment(cost_matrix_numpy)
        for row_index, col_index in zip(row_indexes, col_indexes):
            A_matrix[index_batch, map2RealRows[row_index], map2RealCols[col_index]]=1
            
    num_nonzero_A=th.count_nonzero(A_matrix); #This is the same as the number of distinct trajectories produced by the expert

    pos_loss=th.sum(A_matrix*distance_pos_matrix)/num_nonzero_A
    yaw_loss=th.sum(A_matrix*distance_yaw_matrix)/num_nonzero_A
    time_loss=th.sum(A_matrix*distance_time_matrix)/num_nonzero_A

    assert (distance_matrix.shape)[0]==batch_size, "Wrong shape!"
    assert (distance_matrix.shape)[1]==num_of_traj_per_action, "Wrong shape!"
    assert pos_loss.requires_grad==True
    assert yaw_loss.requires_grad==True
    assert time_loss.requires_grad==True

    loss = time_loss + pos_loss + yaw_loss_weight*yaw_loss

    return loss