#!/usr/bin/env python3

# diffusion policy import
import numpy as np
import torch as th
import os
import torch.nn as nn
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm
import datetime
# import
import torch as th
import time

# network utils import
from utils import get_nactions, calculate_deep_panther_loss, get_dataloader_training

# wanb
import wandb

# import from py_panther

from torch.autograd import Function

def evaluate_non_diffusion_model(dataset, policy, **kwargs):

    """
    Evaluate noise_pred_net
    @param dataset_eval: evaluation dataset
    @param noise_pred_net: noise prediction network
    """

    # unpack
    num_eval = kwargs['num_eval']
    en_network_type = kwargs['en_network_type']

    # set policy to eval mode
    policy.eval()

    # get cost computer
    from compression.utils.other import CostComputer
    cost_computer = CostComputer()

    # num_eval
    num_eval = min(num_eval, len(dataset))

    avg_cost_expert, avg_obst_avoidance_violation_expert, avg_dyn_lim_violation_expert, avg_augmented_cost_expert = [], [], [], []
    min_cost_expert, min_obst_avoidance_violation_expert, min_dyn_lim_violation_expert, min_augmented_cost_expert = [], [], [], []
    avg_cost_student, avg_obst_avoidance_violation_student, avg_dyn_lim_violation_student, avg_augmented_cost_student = [], [], [], []
    min_cost_student, min_obst_avoidance_violation_student, min_dyn_lim_violation_student, min_augmented_cost_student = [], [], [], []
    computation_times = []

    for dataset_idx in range(num_eval):

        # get expert actions and student actions
        nob = dataset[dataset_idx]['obs']
        expert_action = dataset[dataset_idx]['acts']

        if en_network_type == 'gnn':
            x_dict = dataset[dataset_idx].x_dict
            edge_index_dict = dataset[dataset_idx].edge_index_dict
            start_time = time.time()
            student_action = policy(nob, x_dict, edge_index_dict)
            end_time = time.time()
            nob = nob.squeeze(0) # remove the first dimension
            expert_action = expert_action.squeeze(0) # remove the first dimension
        else:
            nob = nob.unsqueeze(1)
            start_time = time.time()
            student_action = policy(nob)
            end_time = time.time()
            nob = nob.squeeze(1)

        computation_times.append(end_time - start_time)

        # print("student_action.shape", student_action.shape)
        # num_obst = student_action.shape[0]
        student_action = student_action.squeeze(0) # remove the first dimension

        # move to numpy
        nob = nob.detach().cpu().numpy()
        expert_action = expert_action.detach().cpu().numpy()
        student_action = student_action.detach().cpu().numpy()

        # compute cost for expert
        cost_expert, obst_avoidance_violation_expert, dyn_lim_violation_expert, augmented_cost_expert = [], [], [], []
        for j in range(expert_action.shape[0]): # expert_action.shape[1] is num_trajs
            cost, obst_avoidance_violation, dyn_lim_violation, augmented_cost = cost_computer.computeCost_AndObsAvoidViolation_AndDynLimViolation_AndAugmentedCost(nob, expert_action[[j], :])
            cost_expert.append(cost)
            obst_avoidance_violation_expert.append(obst_avoidance_violation)
            dyn_lim_violation_expert.append(dyn_lim_violation)
            augmented_cost_expert.append(augmented_cost)

        # get average and min for each trajectory 
        avg_cost_expert.append(np.mean(cost_expert))
        avg_obst_avoidance_violation_expert.append(np.mean(obst_avoidance_violation_expert))
        avg_dyn_lim_violation_expert.append(np.mean(dyn_lim_violation_expert))
        avg_augmented_cost_expert.append(np.mean(augmented_cost_expert))
        min_cost_expert.append(np.min(cost_expert))
        min_obst_avoidance_violation_expert.append(np.min(obst_avoidance_violation_expert))
        min_dyn_lim_violation_expert.append(np.min(dyn_lim_violation_expert))
        min_augmented_cost_expert.append(np.min(augmented_cost_expert))

        # compute cost for student
        cost_student, obst_avoidance_violation_student, dyn_lim_violation_student, augmented_cost_student = [], [], [], []
        # for i in range(num_obst):
        # for i in range(1):
        for j in range(student_action.shape[0]): # student_action.shape[0] is num_trajs
            # cost, obst_avoidance_violation, dyn_lim_violation, augmented_cost = cost_computer.computeCost_AndObsAvoidViolation_AndDynLimViolation_AndAugmentedCost(nob, student_action[1, [j], :])
            cost, obst_avoidance_violation, dyn_lim_violation, augmented_cost = cost_computer.computeCost_AndObsAvoidViolation_AndDynLimViolation_AndAugmentedCost(nob, student_action[[j], :])
            cost_student.append(cost)
            obst_avoidance_violation_student.append(obst_avoidance_violation)
            dyn_lim_violation_student.append(dyn_lim_violation)
            augmented_cost_student.append(augmented_cost)
        
        # get average and min for each trajectory
        avg_cost_student.append(np.mean(cost_student))
        avg_obst_avoidance_violation_student.append(np.mean(obst_avoidance_violation_student))
        avg_dyn_lim_violation_student.append(np.mean(dyn_lim_violation_student))
        avg_augmented_cost_student.append(np.mean(augmented_cost_student))
        min_cost_student.append(np.min(cost_student))
        min_obst_avoidance_violation_student.append(np.min(obst_avoidance_violation_student))
        min_dyn_lim_violation_student.append(np.min(dyn_lim_violation_student))
        min_augmented_cost_student.append(np.min(augmented_cost_student))

    # return
    return  np.mean(avg_cost_expert), np.mean(avg_obst_avoidance_violation_expert), np.mean(avg_dyn_lim_violation_expert), np.mean(avg_augmented_cost_expert), \
            np.mean(min_cost_expert), np.mean(min_obst_avoidance_violation_expert), np.mean(min_dyn_lim_violation_expert), np.mean(min_augmented_cost_expert), \
            np.mean(avg_cost_student), np.mean(avg_obst_avoidance_violation_student), np.mean(avg_dyn_lim_violation_student), np.mean(avg_augmented_cost_student), \
            np.mean(min_cost_student), np.mean(min_obst_avoidance_violation_student), np.mean(min_dyn_lim_violation_student), np.mean(min_augmented_cost_student), \
            np.mean(np.array(computation_times))

def evaluate_diffusion_model(dataset, policy, noise_scheduler, return_only_student=False, **kwargs):

    """
    Evaluate policy
    @param dataset_eval: evaluation dataset
    @param device: device to transfer data to
    @param policy: noise prediction network
    """

    # unpack
    use_gnn = kwargs['en_network_type'] == 'gnn'

    # get cost computer
    from compression.utils.other import CostComputer
    cost_computer = CostComputer()

    # get expert actions and student actions
    expert_actions, student_actions, nobs, computation_time = get_nactions(policy, noise_scheduler, dataset, is_visualize=False, **kwargs)

    avg_cost_expert, avg_obst_avoidance_violation_expert, avg_dyn_lim_violation_expert, avg_augmented_cost_expert = [], [], [], []
    min_cost_expert, min_obst_avoidance_violation_expert, min_dyn_lim_violation_expert, min_augmented_cost_expert = [], [], [], []
    avg_cost_student, avg_obst_avoidance_violation_student, avg_dyn_lim_violation_student, avg_augmented_cost_student = [], [], [], []
    min_cost_student, min_obst_avoidance_violation_student, min_dyn_lim_violation_student, min_augmented_cost_student = [], [], [], []
    if not return_only_student:

        # evaluation for expert actions
        for nob, expert_action in zip(nobs, expert_actions):
            cost_expert, obst_avoidance_violation_expert, dyn_lim_violation_expert, augmented_cost_expert = [], [], [], []
            for j in range(expert_action.shape[0]): # for num_trajs we loop through

                # compute cost
                cost, obst_avoidance_violation, dyn_lim_violation, augmented_cost = cost_computer.computeCost_AndObsAvoidViolation_AndDynLimViolation_AndAugmentedCost(nob[0, [0], :], expert_action[0, [j], :])
                
                # append
                cost_expert.append(cost)
                obst_avoidance_violation_expert.append(obst_avoidance_violation)
                dyn_lim_violation_expert.append(dyn_lim_violation)
                augmented_cost_expert.append(augmented_cost)

            # get average and min for each trajectory and convert to numpy
            avg_cost_expert.append(np.mean(cost_expert))
            avg_obst_avoidance_violation_expert.append(np.mean(obst_avoidance_violation_expert))
            avg_dyn_lim_violation_expert.append(np.mean(dyn_lim_violation_expert))
            avg_augmented_cost_expert.append(np.mean(augmented_cost_expert))
            min_cost_expert.append(np.min(cost_expert))
            min_obst_avoidance_violation_expert.append(np.min(obst_avoidance_violation_expert))
            min_dyn_lim_violation_expert.append(np.min(dyn_lim_violation_expert))
            min_augmented_cost_expert.append(np.min(augmented_cost_expert))

    else:
        avg_cost_expert, avg_obst_avoidance_violation_expert, avg_dyn_lim_violation_expert, avg_augmented_cost_expert = None, None, None, None
        min_cost_expert, min_obst_avoidance_violation_expert, min_dyn_lim_violation_expert, min_augmented_cost_expert = None, None, None, None

    # evaluation for student actions
    for nob, student_action in zip(nobs, student_actions):
        cost_student, obst_avoidance_violation_student, dyn_lim_violation_student, augmented_cost_student = [], [], [], []
        for j in range(student_action.shape[0]): # for num_trajs we loop through
            
            # compute cost
            cost, obst_avoidance_violation, dyn_lim_violation, augmented_cost = cost_computer.computeCost_AndObsAvoidViolation_AndDynLimViolation_AndAugmentedCost(nob[0, [0], :], student_action[0, [j], :])
            
            # append
            cost_student.append(cost)
            obst_avoidance_violation_student.append(obst_avoidance_violation)
            dyn_lim_violation_student.append(dyn_lim_violation)
            augmented_cost_student.append(augmented_cost)

        # get average and min for each trajectory and convert to numpy
        avg_cost_student.append(np.mean(cost_student))
        avg_obst_avoidance_violation_student.append(np.mean(obst_avoidance_violation_student))
        avg_dyn_lim_violation_student.append(np.mean(dyn_lim_violation_student))
        avg_augmented_cost_student.append(np.mean(augmented_cost_student))
        min_cost_student.append(np.min(cost_student))
        min_obst_avoidance_violation_student.append(np.min(obst_avoidance_violation_student))
        min_dyn_lim_violation_student.append(np.min(dyn_lim_violation_student))
        min_augmented_cost_student.append(np.min(augmented_cost_student))

    # return the average of average and min cost
    return np.mean(avg_cost_expert), np.mean(avg_obst_avoidance_violation_expert), np.mean(avg_dyn_lim_violation_expert), np.mean(avg_augmented_cost_expert), \
            np.mean(min_cost_expert), np.mean(min_obst_avoidance_violation_expert), np.mean(min_dyn_lim_violation_expert), np.mean(min_augmented_cost_expert), \
            np.mean(avg_cost_student), np.mean(avg_obst_avoidance_violation_student), np.mean(avg_dyn_lim_violation_student), np.mean(avg_augmented_cost_student), \
            np.mean(min_cost_student), np.mean(min_obst_avoidance_violation_student), np.mean(min_dyn_lim_violation_student), np.mean(min_augmented_cost_student), \
            computation_time

def get_samples_nbatch(nbatch, policy, noise_scheduler, **kwargs):

    """
    Get samples. This is used for training with RL.
    
    create:
        samples["advantage"] has data size of (nbatch['obs'].shape[0], num_trajs, noise_scheduler.config.num_train_timesteps)
        samples["log_prob"] has data size of (nbatch['obs'].shape[0], num_trajs, noise_scheduler.config.num_train_timesteps)
        samples["prev_log_prob"] has data size of (nbatch['obs'].shape[0], num_trajs, noise_scheduler.config.num_train_timesteps)
    
    @return samples
    """

    # unpack
    num_trajs = kwargs['num_trajs']
    use_gnn = kwargs['en_network_type'] == 'gnn'
    device = kwargs['device']

    # initialize samples with tensors
    samples = dict()
    samples["advantage"] = np.zeros((nbatch['obs'].shape[0], num_trajs, noise_scheduler.config.num_train_timesteps)) # advantage doesn't need to be autograded
    samples["log_prob"] = th.zeros((nbatch['obs'].shape[0], num_trajs, noise_scheduler.config.num_train_timesteps), device=device) # log_prob needs to be autograded
    samples["prev_log_prob"] = th.zeros((nbatch['obs'].shape[0], num_trajs, noise_scheduler.config.num_train_timesteps), device=device)

    # get reward (this is the true reward for the state at t=0)
    reward = nbatch['reward']
    samples["advantage"][:, :, -1] = reward

    # calulate advantage
    avg_reward = np.mean(samples["advantage"])
    std_reward = np.std(samples["advantage"])
    samples["advantage"] = (samples["advantage"] - avg_reward) / std_reward

    # get log_prob

    # data reshape
    nobs = nbatch['obs']
    naction = nbatch['acts']

    x_dict = nbatch.x_dict if use_gnn else None
    edge_index_dict = nbatch.edge_index_dict if use_gnn else None
    B = nobs.shape[0]

    # observation as FiLM conditioning
    # (B, obs_horizon, obs_dim)
    obs_cond = nobs[:, :, :]
    
    # (B, obs_horizon * obs_dim)
    obs_cond = obs_cond.flatten(start_dim=1)

    # naction's num_trajs needs to be a multiple of 2 so let's make it 8 for now
    # naction = naction[:, :8, :] # (B, num_trajs, action_dim)

    # sample noise to add to actions
    noise = th.randn(naction.shape, device=device)

    # sample a diffusion iteration for each data point
    timesteps = th.randint(
        0, noise_scheduler.config.num_train_timesteps,
        (B,), device=device
    ).long()

    # add noise to the clean images according to the noise magnitude at each diffusion iteration
    # (this is the forward diffusion process)
    noisy_actions = noise_scheduler.add_noise(naction, noise, timesteps)
    
    # predict the noise residual
    noise_pred = policy(sample=noisy_actions, timestep=timesteps, global_cond=obs_cond, x_dict=x_dict, edge_index_dict=edge_index_dict)

    # create the first sample (latent)
    latent = th.from_numpy(np.random.normal(size=noise_pred.shape)).float().to(device)

    # scale the initial noise by the standard deviation required by the scheduler (from pipline_flax_stable_diffusion.py)
    init_noise_sigma = np.array(1.0, dtype=np.float32)
    latent = latent * init_noise_sigma

    # get log_prob for each diffusion step
    for t in range(noise_scheduler.config.num_train_timesteps-1, 0, -1):

        # get current log probability
        latent, current_log_prob = noise_scheduler.get_log_prob(model_output=noise_pred, timestep=t, sample=latent, prev_sample=None)

        # store current log probability
        samples["log_prob"][:, :, t] = current_log_prob 
        
    # get prev_log_prob
    # TODO
            
    return samples


def get_samples_dataloader(dataloader, policy, noise_scheduler, **kwargs):

    """
    Get samples. This is used for training with RL.
    
    create:
        samples["advantage"] has data size of (len(dataloader.dataset), num_trajs, noise_scheduler.config.num_train_timesteps)
        samples["log_prob"] has data size of (len(dataloader.dataset), num_trajs, noise_scheduler.config.num_train_timesteps)
        samples["prev_log_prob"] has data size of (len(dataloader.dataset), num_trajs, noise_scheduler.config.num_train_timesteps)
    
    @return samples
    """

    # unpack
    num_trajs = kwargs['num_trajs']
    use_gnn = kwargs['en_network_type'] == 'gnn'
    device = kwargs['device']

    # initialize samples with tensors
    samples = dict()
    samples["advantage"] = np.zeros((len(dataloader.dataset), num_trajs, noise_scheduler.config.num_train_timesteps)) # advantage doesn't need to be autograded
    samples["log_prob"] = th.zeros((len(dataloader.dataset), num_trajs, noise_scheduler.config.num_train_timesteps), device=device) # log_prob needs to be autograded
    samples["prev_log_prob"] = th.zeros((len(dataloader.dataset), num_trajs, noise_scheduler.config.num_train_timesteps), device=device)

    # get advantage
    for nbatch_idx, nbatch in enumerate(dataloader):

        # get reward (this is the true reward for the state at t=0)
        reward = nbatch['reward']

        # assign reward to samples 
        batch_start_idx = nbatch_idx * reward.shape[0]
        batch_end_idx = batch_start_idx + reward.shape[0]
        samples["advantage"][batch_start_idx:batch_end_idx, :, -1] = reward

    # calulate advantage
    avg_reward = np.mean(samples["advantage"])
    std_reward = np.std(samples["advantage"])
    samples["advantage"] = (samples["advantage"] - avg_reward) / std_reward

    # get log_prob
    for nbatch_idx, nbatch in enumerate(dataloader):

        # data reshape
        nobs = nbatch['obs']
        naction = nbatch['acts']

        x_dict = nbatch.x_dict if use_gnn else None
        edge_index_dict = nbatch.edge_index_dict if use_gnn else None
        B = nobs.shape[0]

        # observation as FiLM conditioning
        # (B, obs_horizon, obs_dim)
        obs_cond = nobs[:, :, :]
        
        # (B, obs_horizon * obs_dim)
        obs_cond = obs_cond.flatten(start_dim=1)

        # naction's num_trajs needs to be a multiple of 2 so let's make it 8 for now
        # naction = naction[:, :8, :] # (B, num_trajs, action_dim)

        # sample noise to add to actions
        noise = th.randn(naction.shape, device=device)

        # sample a diffusion iteration for each data point
        timesteps = th.randint(
            0, noise_scheduler.config.num_train_timesteps,
            (B,), device=device
        ).long()

        # add noise to the clean images according to the noise magnitude at each diffusion iteration
        # (this is the forward diffusion process)
        noisy_actions = noise_scheduler.add_noise(naction, noise, timesteps)
        
        # predict the noise residual
        noise_pred = policy(sample=noisy_actions, timestep=timesteps, global_cond=obs_cond, x_dict=x_dict, edge_index_dict=edge_index_dict)

        # create the first sample (latent)
        latent = th.from_numpy(np.random.normal(size=noise_pred.shape)).float().to(device)

        # scale the initial noise by the standard deviation required by the scheduler (from pipline_flax_stable_diffusion.py)
        init_noise_sigma = np.array(1.0, dtype=np.float32)
        latent = latent * init_noise_sigma

        # get log_prob for each diffusion step
        for t in range(noise_scheduler.config.num_train_timesteps-1, 0, -1):

            # get current log probability
            latent, current_log_prob = noise_scheduler.get_log_prob(model_output=noise_pred, timestep=t, sample=latent, prev_sample=None)

            # store current log probability
            batch_start_idx = nbatch_idx * current_log_prob.shape[0]
            batch_end_idx = batch_start_idx + current_log_prob.shape[0]

            samples["log_prob"][batch_start_idx:batch_end_idx, :, t] = current_log_prob 
            
    # get prev_log_prob
    # TODO
            
    return samples

def create_dataloader_with_reward(dataset, policy, noise_scheduler, **kwargs):

    """
    Calculate average and std of the dataset rewards
    """

    # store the original num_eval
    original_num_eval = kwargs['num_eval']
    num_trajs = kwargs['num_trajs']

    # change num_eval to the size of the dataset
    kwargs['num_eval'] = len(dataset)

    # compute reward
    cost = evaluate_diffusion_model(dataset, policy, noise_scheduler, return_only_student=True, **kwargs)[4]
    reward = -cost

    # reshape reward from (num_eval*num_trajs, ) to (num_eval(=len(dataset)), num_trajs)
    reward = reward.reshape((len(dataset), num_trajs))

    # add reward to dataset
    dataset.add_reward(reward)

    # create a new dataloader with reward
    dataloader = get_dataloader_training(dataset, **kwargs)

    # reset num_eval
    kwargs['num_eval'] = original_num_eval

    return dataloader

def train_loop_diffusion_model_with_rl(policy, optimizer, lr_scheduler, noise_scheduler, ema, **kwargs):

    """
    Train noise_pred_net
    @param num_epochs: number of epochs
    @param dataloader_training: dataloader_training
    @param device: device to transfer data to
    @param noise_pred_net: noise prediction network
    @param noise_scheduler: noise scheduler
    @param ema: Exponential Moving Average
    @param optimizer: optimizer
    @param lr_scheduler: learning rate scheduler
    """

    # unpack
    num_epochs = kwargs['num_epochs']
    dataloader_training = kwargs['datasets_loader']['dataloader_training']
    use_gnn = kwargs['en_network_type'] == 'gnn'
    device = kwargs['device']
    save_dir = kwargs['save_dir']
    dataset_eval = kwargs['datasets_loader']['dataset_eval']
    dataset_training = kwargs['datasets_loader']['dataset_training']
    en_network_type = kwargs['en_network_type']
    de_network_type = kwargs['de_network_type']
    policy_save_freq = kwargs['policy_save_freq']
    machine = kwargs['machine']
    use_rl = kwargs['use_rl']
    clip_for_rl = kwargs['clip_for_rl']
    adv_clip_for_rl = kwargs['adv_clip_for_rl']
    use_reinforce_for_rl = kwargs['use_reinforce_for_rl']
    use_importance_sampling_for_rl = kwargs['use_importance_sampling_for_rl']
    
    # set policy to train mode
    policy.train()

    # find average and std of the dataset rewards
    dataloader = create_dataloader_with_reward(dataset_training, policy, noise_scheduler, **kwargs)

    # add to kwargs
    kwargs['datasets_loader']['dataloader_training'] = dataloader


    # training loop
    wandb.init(project='diffusion')
    epoch_counter = 0
    overfitting_counter = 0
    with tqdm(range(num_epochs), desc='Epoch') as tglobal:
        
        # epoch loop
        for epoch_idx in tglobal:
            epoch_loss = list()
        
            # get samples
            samples = get_samples_dataloader(dataloader, policy, noise_scheduler, **kwargs)
            
            # batch loop
            # with tqdm(dataloader_training, desc='Batch', leave=False) as tepoch:
            #     for nbatch in tepoch:
                        
            loss = th.zeros(1, device=device)

            ## Use REINFORCE
            if use_reinforce_for_rl:
                
                # loop over denoising steps
                for j in range(noise_scheduler.config.num_train_timesteps):

                    # accumulating loss could lead to a big loss
                    # create samples
                    # samples = get_samples_nbatch(nbatch, policy, noise_scheduler, **kwargs)

                    # get advantage
                    advantage = th.from_numpy(samples["advantage"][:, :, j]).float().to(device).detach()
                    # clip advantage
                    clipped_advantage = th.clamp(advantage, -adv_clip_for_rl, adv_clip_for_rl)
                    # get current log probability
                    log_prob = samples["log_prob"][:, :, j]
                    # clip log_prob
                    clipped_log_prob = th.clamp(log_prob, -clip_for_rl, clip_for_rl)
                    # get surrogate loss: the negative sign is needed because we want to maximize the surrogate loss. max reward = min cost 
                    loss += -th.mean(advantage * log_prob)
                    # loss += -th.mean(clipped_advantage * clipped_log_prob)


            ## for importance sampling
            elif use_importance_sampling_for_rl: # TODO: you can optimize this with multiple steps (need to change the entire training structure)
                pass

            # TODO: check - i think this needs to here.
            # optimize
            with th.autograd.set_detect_anomaly(True):
                loss.backward(retain_graph=True)
                optimizer.step()
                optimizer.zero_grad()
            # step lr scheduler every batch
            # this is different from standard pytorch behavior
            lr_scheduler.step()

            # update Exponential Moving Average of the model weights
            ema.step(policy.parameters())

            # logging
            loss = loss.item()
            epoch_loss.append(loss)
            # tepoch.set_postfix(loss=loss)

            # wandb logging
            wandb.log({'loss': loss, 'epoch': epoch_idx})

            # save model
            if epoch_counter % policy_save_freq == 0:
                filename = f'{save_dir}/{en_network_type}_{de_network_type}_num_{epoch_counter}.pth'
                th.save(policy.state_dict(), filename)
            epoch_counter += 1
                    
            tglobal.set_postfix(loss=np.mean(epoch_loss))

            # if machine == 'jtorde': # if it's my desktop, then evaluate on the whole dataset
            #     # each epoch, we evaluate the model on evaluation data
            #     costs = evaluate_diffusion_model(dataset_eval, policy, noise_scheduler, **kwargs)
            #     # unpack
            #     avg_cost_expert_eval, avg_obst_avoidance_violation_expert_eval, avg_dyn_lim_violation_expert_eval, avg_augmented_cost_expert_eval, \
            #     min_cost_expert_eval, min_obst_avoidance_violation_expert_eval, min_dyn_lim_violation_expert_eval, min_augmented_cost_expert_eval, \
            #     avg_cost_student_eval, avg_obst_avoidance_violation_student_eval, avg_dyn_lim_violation_student_eval, avg_augmented_cost_student_eval, \
            #     min_cost_student_eval, min_obst_avoidance_violation_student_eval, min_dyn_lim_violation_student_eval, min_augmented_cost_student_eval, \
            #     computation_time = costs

            #     # each epoch we evaluate the model on training data too (to check overfitting)
            #     costs = evaluate_diffusion_model(dataset_training, policy, noise_scheduler, **kwargs)
            #     # unpack
            #     avg_cost_expert_training, avg_obst_avoidance_violation_expert_training, avg_dyn_lim_violation_expert_training, avg_augmented_cost_expert_training, \
            #     min_cost_expert_training, min_obst_avoidance_violation_expert_training, min_dyn_lim_violation_expert_training, min_augmented_cost_expert_training, \
            #     avg_cost_student_training, avg_obst_avoidance_violation_student_training, avg_dyn_lim_violation_student_training, avg_augmented_cost_student_training, \
            #     min_cost_student_training, min_obst_avoidance_violation_student_training, min_dyn_lim_violation_student_training, min_augmented_cost_student_training, \
            #     computation_time = costs

            #     # wandb logging
            #     wandb.log({
            #         'min_cost_expert_eval': min_cost_expert_eval,
            #         'avg_cost_expert_eval': avg_cost_expert_eval,
            #         'min_cost_student_eval': min_cost_student_eval,
            #         'avg_cost_student_eval': avg_cost_student_eval,
            #         'min_cost_expert_training': min_cost_expert_training,
            #         'avg_cost_expert_training': avg_cost_expert_training,
            #         'min_cost_student_training': min_cost_student_training,
            #         'avg_cost_student_training': avg_cost_student_training,
            #         'computation_time': computation_time,
            #         'epoch': epoch_idx
            #     })
                
            #     # terminate conditions
            #     # if epoch_counter is more than 1/5 of the total epoch and augmented cost of expert is less than augmented cost of student 5 times in a row, then stop training
            #     if epoch_counter >= num_epochs and avg_augmented_cost_expert_eval < avg_augmented_cost_student_eval:
            #         overfitting_counter += 1
            #     else:
            #         overfitting_counter = 0
            #     if overfitting_counter >= 5:
            #         break
            
    return policy
    
def train_loop_diffusion_model(policy, optimizer, lr_scheduler, noise_scheduler, ema, **kwargs):

    """
    Train noise_pred_net
    @param num_epochs: number of epochs
    @param dataloader_training: dataloader_training
    @param device: device to transfer data to
    @param noise_pred_net: noise prediction network
    @param noise_scheduler: noise scheduler
    @param ema: Exponential Moving Average
    @param optimizer: optimizer
    @param lr_scheduler: learning rate scheduler
    """

    # unpack
    num_epochs = kwargs['num_epochs']
    dataloader_training = kwargs['datasets_loader']['dataloader_training']
    use_gnn = kwargs['en_network_type'] == 'gnn'
    device = kwargs['device']
    save_dir = kwargs['save_dir']
    dataset_eval = kwargs['datasets_loader']['dataset_eval']
    dataset_training = kwargs['datasets_loader']['dataset_training']
    en_network_type = kwargs['en_network_type']
    de_network_type = kwargs['de_network_type']
    policy_save_freq = kwargs['policy_save_freq']
    machine = kwargs['machine']

    # set policy to train mode
    policy.train()

    # training loop
    wandb.init(project='diffusion')
    epoch_counter = 0
    overfitting_counter = 0
    with tqdm(range(num_epochs), desc='Epoch') as tglobal:
        
        # epoch loop
        for epoch_idx in tglobal:
            epoch_loss = list()
            
            # batch loop
            with tqdm(dataloader_training, desc='Batch', leave=False) as tepoch:
                for nbatch in tepoch:

                    # data reshape
                    nobs = nbatch['obs']
                    naction = nbatch['acts']

                    x_dict = nbatch.x_dict if use_gnn else None
                    edge_index_dict = nbatch.edge_index_dict if use_gnn else None
                    B = nobs.shape[0]

                    # observation as FiLM conditioning
                    # (B, 1, obs_dim) but obs_dim might have multiple obstacles' info (ex. obs_dim could be 43 or 76)
                    obs_cond = nobs[:, :, :]

                    # sample noise to add to actions
                    noise = th.randn(naction.shape, device=device)

                    # sample a diffusion iteration for each data point
                    timesteps = th.randint(
                        0, noise_scheduler.config.num_train_timesteps,
                        (B,), device=device
                    ).long()

                    # add noise to the clean images according to the noise magnitude at each diffusion iteration
                    # (this is the forward diffusion process)
                    noisy_actions = noise_scheduler.add_noise(naction, noise, timesteps)
                    
                    # predict the noise residual
                    noise_pred = policy(sample=noisy_actions, timestep=timesteps, global_cond=obs_cond, x_dict=x_dict, edge_index_dict=edge_index_dict)

                    # L2 loss
                    loss = nn.functional.mse_loss(noise_pred, noise)

                    # optimize
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    # step lr scheduler every batch
                    # this is different from standard pytorch behavior
                    lr_scheduler.step()

                    # update Exponential Moving Average of the model weights
                    ema.step(policy.parameters())

                    # logging
                    loss = loss.item()
                    epoch_loss.append(loss)
                    tepoch.set_postfix(loss=loss)

                    # wandb logging
                    wandb.log({'loss': loss, 'epoch': epoch_idx})

            # save model
            if epoch_counter % policy_save_freq == 0:
                filename = f'{save_dir}/{en_network_type}_{de_network_type}_num_{epoch_counter}.pth'
                th.save(policy.state_dict(), filename)
            epoch_counter += 1
                    
            tglobal.set_postfix(loss=np.mean(epoch_loss))

            # if machine == 'jtorde': # if it's my desktop, then evaluate on the whole dataset
            #     # each epoch, we evaluate the model on evaluation data
            #     costs = evaluate_diffusion_model(dataset_eval, policy, noise_scheduler, **kwargs)
            #     # unpack
            #     avg_cost_expert_eval, avg_obst_avoidance_violation_expert_eval, avg_dyn_lim_violation_expert_eval, avg_augmented_cost_expert_eval, \
            #     min_cost_expert_eval, min_obst_avoidance_violation_expert_eval, min_dyn_lim_violation_expert_eval, min_augmented_cost_expert_eval, \
            #     avg_cost_student_eval, avg_obst_avoidance_violation_student_eval, avg_dyn_lim_violation_student_eval, avg_augmented_cost_student_eval, \
            #     min_cost_student_eval, min_obst_avoidance_violation_student_eval, min_dyn_lim_violation_student_eval, min_augmented_cost_student_eval = costs

            #     # each epoch we evaluate the model on training data too (to check overfitting)
            #     costs = evaluate_diffusion_model(dataset_training, policy, noise_scheduler, **kwargs)
            #     # unpack
            #     avg_cost_expert_training, avg_obst_avoidance_violation_expert_training, avg_dyn_lim_violation_expert_training, avg_augmented_cost_expert_training, \
            #     min_cost_expert_training, min_obst_avoidance_violation_expert_training, min_dyn_lim_violation_expert_training, min_augmented_cost_expert_training, \
            #     avg_cost_student_training, avg_obst_avoidance_violation_student_training, avg_dyn_lim_violation_student_training, avg_augmented_cost_student_training, \
            #     min_cost_student_training, min_obst_avoidance_violation_student_training, min_dyn_lim_violation_student_training, min_augmented_cost_student_training = costs

            #     # wandb logging
            #     wandb.log({
            #         'min_cost_expert_eval': min_cost_expert_eval,
            #         'avg_cost_expert_eval': avg_cost_expert_eval,
            #         'min_cost_student_eval': min_cost_student_eval,
            #         'avg_cost_student_eval': avg_cost_student_eval,
            #         'min_cost_expert_training': min_cost_expert_training,
            #         'avg_cost_expert_training': avg_cost_expert_training,
            #         'min_cost_student_training': min_cost_student_training,
            #         'avg_cost_student_training': avg_cost_student_training,
            #         'epoch': epoch_idx
            #     })
                
            #     # terminate conditions
            #     # if epoch_counter is more than 1/5 of the total epoch and augmented cost of expert is less than augmented cost of student 5 times in a row, then stop training
            #     if epoch_counter >= num_epochs and avg_augmented_cost_expert_eval < avg_augmented_cost_student_eval:
            #         overfitting_counter += 1
            #     else:
            #         overfitting_counter = 0
            #     if overfitting_counter >= 5:
            #         break
            
    return policy

def train_diffusion_model(policy, noise_scheduler, **kwargs):

    """
    Train diffusion model
    """

    # unpack
    dataloader_training = kwargs['datasets_loader']['dataloader_training']
    num_epochs = kwargs['num_epochs']
    save_dir = kwargs['save_dir']
    en_network_type = kwargs['en_network_type']
    de_network_type = kwargs['de_network_type']

    # Exponential Moving Average
    # accelerates training and improves stability
    # holds a copy of the model weights
    ema = EMAModel(
        parameters=policy.parameters(),
        power=0.75)

    # Standard ADAM optimizer
    # Note that EMA parametesr are not optimized
    optimizer = th.optim.AdamW(
        params=policy.parameters(),
        lr=1e-4, weight_decay=1e-6)

    # Cosine LR schedule with linear warmup
    lr_scheduler = get_scheduler(
        name='cosine',
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=len(dataloader_training) * num_epochs
    )

    # training loop
    policy = train_loop_diffusion_model(policy, optimizer, lr_scheduler, noise_scheduler, ema, **kwargs) 
    # policy = train_loop_diffusion_model_with_rl(policy, optimizer, lr_scheduler, noise_scheduler, ema, **kwargs) 

    # Weights of the EMA model
    # is used for inference
    ema.copy_to(policy.parameters())

    # save model
    filename = f'{save_dir}/{en_network_type}_{de_network_type}_final.pth'
    th.save(policy.state_dict(), filename)

    return policy

def train_loop_non_diffusion_model(policy, optimizer, lr_scheduler, **kwargs):
    """
    Train MLP/LSTM/Transformer
    """

    # unpack
    num_epochs = kwargs['num_epochs']
    dataloader_training = kwargs['datasets_loader']['dataloader_training']
    dataset_training = kwargs['datasets_loader']['dataset_training']
    dataset_eval = kwargs['datasets_loader']['dataset_eval']
    save_dir = kwargs['save_dir']
    en_network_type = kwargs['en_network_type']
    de_network_type = kwargs['de_network_type']
    policy_save_freq = kwargs['policy_save_freq']
    machine = kwargs['machine']

    # training loop
    wandb.init(project=en_network_type)
    epoch_counter = 0
    overfitting_counter = 0
    with tqdm(range(num_epochs), desc='Epoch') as tglobal:
        
        # epoch loop
        for epoch_idx in tglobal:
            epoch_loss = list()
            
            # batch loop
            with tqdm(dataloader_training, desc='Batch', leave=False) as tepoch:
                for nbatch in tepoch:

                    loss = calculate_deep_panther_loss(nbatch, policy, **kwargs) # calculate loss
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    lr_scheduler.step()

                    # logging
                    loss = loss.item()
                    epoch_loss.append(loss)
                    tepoch.set_postfix(loss=loss)

                    # wandb logging
                    wandb.log({'loss': loss, 'epoch': epoch_idx})

            # save model
            if epoch_counter % policy_save_freq == 0:
                filename = f'{save_dir}/{en_network_type}_{de_network_type}_num_{epoch_counter}.pth'
                th.save(policy.state_dict(), filename)
            epoch_counter += 1
                    
            tglobal.set_postfix(loss=np.mean(epoch_loss))

            # each epoch, we evaluate the model on evaluation data
            # if machine == 'jtorde': # if it's my desktop, then evaluate on the whole dataset
            #     # each epoch, we evaluate the model on evaluation data
            #     costs = evaluate_non_diffusion_model(dataset_eval, policy, **kwargs)
            #     # unpack
            #     avg_cost_expert_eval, avg_obst_avoidance_violation_expert_eval, avg_dyn_lim_violation_expert_eval, avg_augmented_cost_expert_eval, \
            #     min_cost_expert_eval, min_obst_avoidance_violation_expert_eval, min_dyn_lim_violation_expert_eval, min_augmented_cost_expert_eval, \
            #     avg_cost_student_eval, avg_obst_avoidance_violation_student_eval, avg_dyn_lim_violation_student_eval, avg_augmented_cost_student_eval, \
            #     min_cost_student_eval, min_obst_avoidance_violation_student_eval, min_dyn_lim_violation_student_eval, min_augmented_cost_student_eval = costs

            #     # each epoch we evaluate the model on training data too (to check overfitting)
            #     costs = evaluate_non_diffusion_model(dataset_training, policy, **kwargs)
            #     # unpack
            #     avg_cost_expert_training, avg_obst_avoidance_violation_expert_training, avg_dyn_lim_violation_expert_training, avg_augmented_cost_expert_training, \
            #     min_cost_expert_training, min_obst_avoidance_violation_expert_training, min_dyn_lim_violation_expert_training, min_augmented_cost_expert_training, \
            #     avg_cost_student_training, avg_obst_avoidance_violation_student_training, avg_dyn_lim_violation_student_training, avg_augmented_cost_student_training, \
            #     min_cost_student_training, min_obst_avoidance_violation_student_training, min_dyn_lim_violation_student_training, min_augmented_cost_student_training = costs

            #     # wandb logging
            #     wandb.log({
            #         'min_cost_expert_eval': min_cost_expert_eval,
            #         'avg_cost_expert_eval': avg_cost_expert_eval,
            #         'min_cost_student_eval': min_cost_student_eval,
            #         'avg_cost_student_eval': avg_cost_student_eval,
            #         'min_cost_expert_training': min_cost_expert_training,
            #         'avg_cost_expert_training': avg_cost_expert_training,
            #         'min_cost_student_training': min_cost_student_training,
            #         'avg_cost_student_training': avg_cost_student_training,
            #         'epoch': epoch_idx
            #     })
                
            #     # terminate conditions
            #     # if epoch_counter is more than 1/5 of the total epoch and augmented cost of expert is less than augmented cost of student 5 times in a row, then stop training
            #     if epoch_counter >= num_epochs and avg_augmented_cost_expert_eval < avg_augmented_cost_student_eval:
            #         overfitting_counter += 1
            #     else:
            #         overfitting_counter = 0
            #     if overfitting_counter >= 5:
            #         break
            
    return policy

def train_non_diffusion_model(policy, **kwargs):

    """
    Train MLP
    """

    # unpack
    num_epochs = kwargs['num_epochs']
    dataloader_training = kwargs['datasets_loader']['dataloader_training']
    save_dir = kwargs['save_dir']
    en_network_type = kwargs['en_network_type']
    de_network_type = kwargs['de_network_type']

    # Standard ADAM optimizer
    # Note that EMA parametesr are not optimized
    optimizer = th.optim.AdamW(
        params=policy.parameters(),
        lr=1e-4, weight_decay=1e-6)

    # Cosine LR schedule with linear warmup
    lr_scheduler = get_scheduler(
        name='cosine',
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=len(dataloader_training) * num_epochs
    )

    # training loop
    policy = train_loop_non_diffusion_model(policy, optimizer, lr_scheduler, **kwargs)

    # save model
    filename = f'{save_dir}/{en_network_type}_{de_network_type}_final.pth'
    th.save(policy.state_dict(), filename)

    return policy

def test_net(policy, dataset, noise_scheduler=None, **kwargs):

    """
    Test policy after training
    """

    # unpack
    en_network_type = kwargs['en_network_type']
    de_network_type = kwargs['de_network_type']
    train_model = kwargs['train_model']
    save_dir = kwargs['save_dir']
    model_path = kwargs['model_path']

    # evaluate on test data
    if de_network_type == 'diffusion':
        costs = evaluate_diffusion_model(dataset, policy, noise_scheduler, **kwargs)
    elif de_network_type == 'mlp' or de_network_type == 'lstm' or de_network_type == 'transformer':
        costs = evaluate_non_diffusion_model(dataset, policy, **kwargs)

    # unpack
    avg_cost_expert_test, avg_obst_avoidance_violation_expert_test, avg_dyn_lim_violation_expert_test, avg_augmented_cost_expert_test, \
    min_cost_expert_test, min_obst_avoidance_violation_expert_test, min_dyn_lim_violation_expert_test, min_augmented_cost_expert_test, \
    avg_cost_student_test, avg_obst_avoidance_violation_student_test, avg_dyn_lim_violation_student_test, avg_augmented_cost_student_test, \
    min_cost_student_test, min_obst_avoidance_violation_student_test, min_dyn_lim_violation_student_test, min_augmented_cost_student_test, \
    computation_time = costs

    # print
    print("en_network_type:                 ", en_network_type)
    print("de_network_type:                 ", de_network_type)
    print("min expert test:                 ", min_cost_expert_test)
    print("min student test:                ", min_cost_student_test)
    print("avg expert test:                 ", avg_cost_expert_test)
    print("avg student test:                ", avg_cost_student_test)
    print("avg augmented cost expert test:  ", avg_augmented_cost_expert_test)
    print("avg augmented cost student test: ", avg_augmented_cost_student_test)
    print("computation time:                ", computation_time)
    
    # save results in file
    path = save_dir if train_model else model_path
    with open(f'/media/jtorde/T7/gdp/benchmark_results.txt', 'a') as f:
        f.write(f'date:                        {datetime.datetime.now()}\n')
        f.write(f'model_path:                  {path}\n')
        f.write(f'en_network_type:             {en_network_type}\n')
        f.write(f'de_network_type:             {de_network_type}\n')
        f.write(f'min cost expert test:        {min_cost_expert_test}\n')
        f.write(f'min cost student test:       {min_cost_student_test}\n')
        f.write(f'avg cost expert test:        {avg_cost_expert_test}\n')
        f.write(f'avg cost student test:       {avg_cost_student_test}\n')
        f.write(f'augmented cost expert test:  {avg_augmented_cost_expert_test}\n')
        f.write(f'augmented cost student test: {avg_augmented_cost_student_test}\n')
        f.write(f'computation time:            {computation_time}\n')
        f.write(f'\n')