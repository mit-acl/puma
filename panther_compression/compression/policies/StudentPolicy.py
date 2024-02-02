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

# network utils import
from network_utils import ConditionalUnet1D

class StudentPolicy(BasePolicy):
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
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        if optimizer_kwargs is None:
            optimizer_kwargs = {}
            # Small values to avoid NaN in Adam optimizer
            if optimizer_class == th.optim.Adam:
                optimizer_kwargs["eps"] = 1e-5

        super(StudentPolicy, self).__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor = features_extractor_class(observation_space),
            features_extractor_kwargs = features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=False,
        )

        # Save arguments to re-create object at loading
        self.name=Style.BRIGHT+Fore.WHITE+"  [Stu]"+Style.RESET_ALL
        self.om=ObservationManager()
        self.am=ActionManager()
        self.obsm=ObstaclesManager()
        self.features_extractor_class = features_extractor_class

        # get U-Nets for the agent
        self.unet = ConditionalUnet1D(**kwargs)

        # set dtype to float
        self = self.float()

    def forward(self, 
                sample: th.Tensor,
                timestep: Union[th.Tensor, float, int],
                global_cond=None, 
                deterministic: bool = True) -> th.Tensor:
        
        return self.unet(sample, timestep, global_cond)

    def _get_data(self) -> Dict[str, Any]:
        data = super()._get_data()
        data.update(
            dict(
                net_arch=self.net_arch,
                features_dim=self.features_dim,
                activation_fn=self.activation_fn,
                features_extractor=self.features_extractor,
            )
        )
        return data
    
    def printwithName(self,data):
        print(self.name+data)

    def _predict(self, obs_n: th.Tensor, deterministic: bool = True) -> th.Tensor:

        action = self.forward(obs_n, self.num_obs, self.num_oa, deterministic)
        self.am.assertActionIsNormalized(action.cpu().numpy().reshape(self.am.getActionShape()), self.name)
        return action

    def predictSeveral(self, obs_n, deterministic: bool = True):

        self.features_extractor = self.features_extractor_class(self.observation_space)
        acts=[]
        for i in range(len(obs_n)):
            self.i_index = i
            acts.append(self.predict( obs_n[i,:], deterministic=deterministic)[0].reshape(self.am.getActionShape()))
        acts=np.stack(acts, axis=0)
        return acts
