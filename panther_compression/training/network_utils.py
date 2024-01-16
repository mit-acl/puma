#!/usr/bin/env python3

# diffusion policy import
import torch
import torch.nn as nn
import math
from typing import Union

# torch import
import torch as th

# gnn import
from torch_geometric.nn import HGTConv
from torch_geometric.nn import Linear as gnn_Linear

def reshape_input_for_rnn(x: th.Tensor, obst_obs_dim: int) -> th.Tensor:
    """
    Reshape input tensor for RNN (LSTM and Transformer)
    """

    # reshape input to (batch, num_obst, input_dim=33)
    num_obst = x.shape[-1] // obst_obs_dim
    output = th.zeros((x.shape[0], num_obst, obst_obs_dim))
    for i in range(num_obst):
        # get i-th obstacle observation
        output[:, [i], :] = x[:, :, i*obst_obs_dim: (i+1)*obst_obs_dim]
    return output

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Conv1dBlock(nn.Module):
    '''
        Conv1d --> GroupNorm --> Mish
    '''

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
    def __init__(self,
            in_channels,
            out_channels,
            cond_dim,
            kernel_size=3,
            n_groups=8):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
        ])

        # FiLM modulation https://arxiv.org/abs/1709.07871
        # predicts per-channel scale and bias
        cond_channels = out_channels * 2
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
            nn.Unflatten(-1, (-1, 1))
        )

        # make sure dimensions compatible
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond):
        '''
            x : [ batch_size x in_channels x horizon ]
            cond : [ batch_size x cond_dim]

            returns:
            out : [ batch_size x out_channels x horizon ]
        '''
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)

        embed = embed.reshape(
            embed.shape[0], 2, self.out_channels, 1)
        scale = embed[:,0,...]
        bias = embed[:,1,...]
        out = scale * out + bias

        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out

class ConditionalUnet1D(nn.Module):
    def __init__(self, **kwargs):
        """
        input_dim: Dim of actions.
        global_cond_dim: Dim of global conditioning applied with FiLM
          in addition to diffusion step embedding. This is usually obs_horizon * obs_dim
        diffusion_step_embed_dim: Size of positional encoding for diffusion iteration k
        down_dims: Channel size for each UNet level.
          The length of this array determines numebr of levels.
        kernel_size: Conv kernel size
        n_groups: Number of groups for GroupNorm
        """
        super().__init__()

        
        # unpack kwargs
        input_dim = kwargs["action_dim"]
        obs_dim = kwargs["obs_dim"]
        diffusion_step_embed_dim = kwargs["diffusion_step_embed_dim"]
        down_dims = kwargs["diffusion_down_dims"]
        kernel_size = kwargs["diffusion_kernel_size"]
        use_gnn = kwargs["en_network_type"] == "gnn"
        lstm_hidden_size = kwargs["lstm_hidden_size"]
        transformer_dim_feedforward = kwargs["transformer_dim_feedforward"]
        transformer_dropout = kwargs["transformer_dropout"]
        gnn_hidden_channels = kwargs["gnn_hidden_channels"]
        gnn_num_layers = kwargs["gnn_num_layers"]
        gnn_num_heads = kwargs["gnn_num_heads"]
        group = kwargs["gnn_group"]
        gnn_data = kwargs["gnn_data"]
        en_network_type = kwargs["en_network_type"]
        agent_obs_hidden_sizes = kwargs["agent_obs_hidden_sizes"]
        mlp_hidden_sizes = kwargs["mlp_hidden_sizes"]
        mlp_activation = kwargs["mlp_activation"]
        linear_layer_output_dim = kwargs["linear_layer_output_dim"]
        output_dim_for_agent_obs = kwargs["output_dim_for_agent_obs"]
        transformer_nhead = kwargs["transformer_nhead"]

        self.use_gnn = use_gnn
        self.gnn_data = gnn_data
        self.gnn_hidden_channels = gnn_hidden_channels
        self.gnn_num_layers = gnn_num_layers
        self.gnn_num_heads = gnn_num_heads
        self.group = group
        self.en_network_type = en_network_type
        self.agent_obs_hidden_sizes = agent_obs_hidden_sizes
        self.mlp_hidden_sizes = mlp_hidden_sizes
        self.mlp_activation = mlp_activation
        self.obs_dim = obs_dim

        # Use MLP for agent observation encoder
        # layers = []
        # input_dim_for_agent_obs = self.agent_obs_dim
        # for next_dim in agent_obs_hidden_sizes:
        #     layers.append(nn.Linear(input_dim_for_agent_obs, next_dim))
        #     layers.append(mlp_activation)
        #     input_dim_for_agent_obs = next_dim
        # layers.append(nn.Linear(input_dim_for_agent_obs, output_dim_for_agent_obs))
        # self.agent_obs_layers = nn.Sequential(*layers)

        # Use MLP for encoder
        if self.en_network_type == "mlp":
            linear_layer_input_dim = obs_dim

        # Use LSTM for encoder
        if self.en_network_type == "lstm":
            self.lstm = nn.LSTM(obs_dim, lstm_hidden_size, batch_first=True)
            linear_layer_input_dim = lstm_hidden_size

        # Use Transformer for encoder
        if self.en_network_type == "transformer":
            # TransformerEncoderLayer won't be able to output the fixed tensor size when input's num of obst is not fixed
            # self.transformer = nn.TransformerEncoderLayer(d_model=obs_dim, nhead=transformer_nhead, dim_feedforward=transformer_dim_feedforward, dropout=transformer_dropout, batch_first=True)
            self.transformer = nn.Transformer(d_model=obs_dim, nhead=transformer_nhead, num_encoder_layers=4, num_decoder_layers=4, dim_feedforward=transformer_dim_feedforward, dropout=transformer_dropout, batch_first=True)
            linear_layer_input_dim = obs_dim

        # Use GNN for encoder
        if self.en_network_type == "gnn":
            self.lin_dict = th.nn.ModuleDict()
            for node_type in self.gnn_data.node_types:
                self.lin_dict[node_type] = gnn_Linear(-1, self.gnn_hidden_channels)
            # HGTConv Layers
            self.convs = th.nn.ModuleList()
            for _ in range(self.gnn_num_layers):
                conv = HGTConv(self.gnn_hidden_channels, self.gnn_hidden_channels, self.gnn_data.metadata(), self.gnn_num_heads, group=self.group)
                self.convs.append(conv)
            linear_layer_input_dim = self.gnn_hidden_channels

        # linear layers after obst encoder
        layers = []
        for next_dim in mlp_hidden_sizes:
            layers.append(nn.Linear(linear_layer_input_dim, next_dim))
            layers.append(mlp_activation)
            linear_layer_input_dim = next_dim
        layers.append(nn.Linear(linear_layer_input_dim, linear_layer_output_dim))
        self.layers = nn.Sequential(*layers)

        # tanh activation
        self.tanh = nn.Tanh()

        #  diffusion step encoder
        dsed = diffusion_step_embed_dim
        diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )

        # 1D UNet
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]
        cond_dim = dsed + linear_layer_output_dim
        in_out = list(zip(all_dims[:-1], all_dims[1:]))

        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(dim_in, dim_out, cond_dim),
                ConditionalResidualBlock1D(dim_out, dim_out, cond_dim),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim),
            ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim),
        ])

        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(dim_out*2, dim_in, cond_dim),
                ConditionalResidualBlock1D(dim_in, dim_in, cond_dim),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))

        final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

        self.diffusion_step_encoder = diffusion_step_encoder
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv

        # print("number of parameters: {:e}".format(
        #     sum(p.numel() for p in self.parameters()))
        # )

        # make it float
        self = self.float()

    def forward(self,
            sample: torch.Tensor,
            timestep: Union[torch.Tensor, float, int],
            global_cond=None,
            x_dict=None,
            edge_index_dict=None
            ):
        """
        x: (B,T,input_dim)
        timestep: (B,) or int, diffusion step
        global_cond: (B,global_cond_dim)
        output: (B,T,input_dim)
        """
        # (B,T,C)
        sample = sample.moveaxis(-1, -2)
        # (B,C,T)

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        
        # Encoder layers
        # first separate the agent observation and the obstacle observation
        # agent_obs = global_cond[:, [0], :10] # agent's observation won't change across obstacles in the same scene
        obst_obs = global_cond[:, :, :]

        # agent_obs goes to a separate linear layer
        # agent_obs = self.agent_obs_layers(agent_obs)

        # obst_obs goes to a separate encoder
        if self.en_network_type == "mlp":
            pass
        elif self.en_network_type == "lstm":
            
            # reshape input to (num_obst, input_dim=33)
            obst_obs = reshape_input_for_rnn(obst_obs, self.obs_dim)
            obst_obs, (h_n, c_n) = self.lstm(obst_obs) # get the obst_obs
            obst_obs = obst_obs[:, [-1], :]
        
        elif self.en_network_type == "transformer":

            # reshape input to (batch_size, num_obst, input_dim=43)
            obst_obs = reshape_input_for_rnn(obst_obs, self.obs_dim)

            obst_obs = self.transformer(src=obst_obs, tgt=obst_obs)
            obst_obs = obst_obs[:, [-1], :] # get the last obst_obs (because we want the obst_obs dim fixed)

        # Use GNN for encoder
        if self.en_network_type == "gnn" and x_dict is not None and edge_index_dict is not None:
            for node_type, x_gnn in x_dict.items():
                x_dict[node_type] = self.lin_dict[node_type](x_gnn).relu_()
            for conv in self.convs:
                x_dict = conv(x_dict, edge_index_dict)
            obst_obs = x_dict["current_state"] # extract the latent vector
            obst_obs = obst_obs.unsqueeze(1)

        # agent_obs and obst_obs are concatenated
        # encoder_output = torch.cat((agent_obs, obst_obs), dim=-1)
        encoder_output = self.layers(obst_obs).squeeze(1)

        # tanh activation
        encoder_output = self.tanh(encoder_output)

        # global conditioning
        timesteps = timesteps.expand(sample.shape[0])
        global_feature = self.diffusion_step_encoder(timesteps)
        global_feature = torch.cat([
            global_feature, encoder_output
        ], axis=-1)

        x = sample
        h = []

        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):

            # print out devices
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)

        # (B,C,T)
        x = x.moveaxis(-1,-2)
        # (B,T,C)
        return x

class MLP(nn.Module):
    """
    Multi-layer perceptron (MLP)
    """
    def __init__(self, **kwargs):
        
        super().__init__()

        # unpack kwargs
        self.action_dim = kwargs["action_dim"]
        self.num_trajs = kwargs["num_trajs"]
        obs_dim = kwargs["obs_dim"]
        use_gnn = kwargs["en_network_type"] == "gnn"
        lstm_hidden_size = kwargs["lstm_hidden_size"]
        transformer_dim_feedforward = kwargs["transformer_dim_feedforward"]
        transformer_dropout = kwargs["transformer_dropout"]
        gnn_hidden_channels = kwargs["gnn_hidden_channels"]
        gnn_num_layers = kwargs["gnn_num_layers"]
        gnn_num_heads = kwargs["gnn_num_heads"]
        group = kwargs["gnn_group"]
        gnn_data = kwargs["gnn_data"]
        en_network_type = kwargs["en_network_type"]
        agent_obs_hidden_sizes = kwargs["agent_obs_hidden_sizes"]
        mlp_hidden_sizes = kwargs["mlp_hidden_sizes"]
        mlp_activation = kwargs["mlp_activation"]
        output_dim_for_agent_obs = kwargs["output_dim_for_agent_obs"]
        transformer_nhead = kwargs["transformer_nhead"]

        self.use_gnn = use_gnn
        self.gnn_data = gnn_data
        self.gnn_hidden_channels = gnn_hidden_channels
        self.gnn_num_layers = gnn_num_layers
        self.gnn_num_heads = gnn_num_heads
        self.group = group
        self.en_network_type = en_network_type
        self.agent_obs_hidden_sizes = agent_obs_hidden_sizes
        self.mlp_hidden_sizes = mlp_hidden_sizes
        self.mlp_activation = mlp_activation
        self.obs_dim = obs_dim

        # Use MLP for encoder
        if self.en_network_type == "mlp":
            linear_layer_input_dim = obs_dim

        # Use LSTM for encoder
        if self.en_network_type == "lstm":
            self.lstm = nn.LSTM(obs_dim, lstm_hidden_size, batch_first=True)
            linear_layer_input_dim = lstm_hidden_size

        # Use Transformer for encoder
        if self.en_network_type == "transformer":
            # TransformerEncoderLayer won't be able to output the fixed tensor size when input's num of obst is not fixed
            # self.transformer = nn.TransformerEncoderLayer(d_model=obs_dim, nhead=transformer_nhead, dim_feedforward=transformer_dim_feedforward, dropout=transformer_dropout, batch_first=True)
            self.transformer = nn.Transformer(d_model=obs_dim, nhead=transformer_nhead, num_encoder_layers=4, num_decoder_layers=4, dim_feedforward=transformer_dim_feedforward, dropout=transformer_dropout, batch_first=True)
            linear_layer_input_dim = obs_dim

        # Use GNN for encoder
        if self.en_network_type == "gnn":
            self.lin_dict = th.nn.ModuleDict()
            for node_type in self.gnn_data.node_types:
                self.lin_dict[node_type] = gnn_Linear(-1, self.gnn_hidden_channels)
            # HGTConv Layers
            self.convs = th.nn.ModuleList()
            for _ in range(self.gnn_num_layers):
                conv = HGTConv(self.gnn_hidden_channels, self.gnn_hidden_channels, self.gnn_data.metadata(), self.gnn_num_heads, group=self.group)
                self.convs.append(conv)
            linear_layer_input_dim = self.gnn_hidden_channels

        # linear layers after obst encoder
        layers = []
        for next_dim in mlp_hidden_sizes:
            layers.append(nn.Linear(linear_layer_input_dim, next_dim))
            layers.append(mlp_activation)
            linear_layer_input_dim = next_dim
        layers.append(nn.Linear(linear_layer_input_dim, self.num_trajs*self.action_dim))
        self.layers = nn.Sequential(*layers)

        # tanh activation
        self.tanh = nn.Tanh()

    def forward(self, x: th.Tensor, x_dict=None, edge_index_dict=None) -> th.Tensor:

        # Encoder layers
        
        if self.en_network_type == "mlp":
            
            output = self.layers(x)

        elif self.en_network_type == "lstm":
            
            # reshape input to (num_obst, input_dim=33)
            output = reshape_input_for_rnn(x, self.obs_dim)
            output, (h_n, c_n) = self.lstm(output) # get the output
            output = output[:, [-1], :]
            output = self.layers(output) # pass it through the layers
        
        elif self.en_network_type == "transformer":

            # reshape input to (batch_size, num_obst, input_dim=43)
            output = reshape_input_for_rnn(x, self.obs_dim)

            output = self.transformer(src=output, tgt=output)
            output = output[:, [-1], :] # get the last output (because we want the output dim fixed)
            output = self.layers(output) # pass it through the layers

        # Use GNN for encoder
        if self.en_network_type == "gnn" and x_dict is not None and edge_index_dict is not None:
            for node_type, x_gnn in x_dict.items():
                x_dict[node_type] = self.lin_dict[node_type](x_gnn).relu_()
            for conv in self.convs:
                x_dict = conv(x_dict, edge_index_dict)
            output = x_dict["current_state"] # extract the latent vector
            output = self.layers(output)
            output = output.unsqueeze(1)

        # tanh activation
        output = self.tanh(output)

        # (B, num_traj, action_dim)
        output = output.reshape(output.shape[0], self.num_trajs, self.action_dim)

        return output