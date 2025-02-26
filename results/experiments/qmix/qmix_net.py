import torch.nn as nn
import torch
import torch.nn.functional as F

class QMixNet(nn.Module):
    def __init__(self, args):
        super(QMixNet, self).__init__()
        self.args = args

        # Initialize the hypernetwork for hyper_w1 and hyper_w2 with two layers or not
        if args.two_hyper_layers:
            self.hyper_w1 = nn.Sequential(
                nn.Linear(args.state_shape, args.hyper_hidden_dim),
                nn.ReLU(),
                nn.Linear(args.hyper_hidden_dim, args.n_agents * args.qmix_hidden_dim)
            )
            self.hyper_w2 = nn.Sequential(
                nn.Linear(args.state_shape, args.hyper_hidden_dim),
                nn.ReLU(),
                nn.Linear(args.hyper_hidden_dim, args.qmix_hidden_dim)
            )
        else:
            self.hyper_w1 = nn.Linear(args.state_shape, args.n_agents * args.qmix_hidden_dim)
            self.hyper_w2 = nn.Linear(args.state_shape, args.qmix_hidden_dim)

        self.hyper_b1 = nn.Linear(args.state_shape, args.qmix_hidden_dim)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(args.state_shape, args.qmix_hidden_dim),
            nn.ReLU(),
            nn.Linear(args.qmix_hidden_dim, 1)
        )

    def forward(self, q_values, states):
        """
        q_values: shape (episode_num, max_episode_len, n_agents)
        states: shape (episode_num, max_episode_len, state_shape)
        """
        episode_num = q_values.size(0)
        # Reshape tensors to match the expected dimensions
        q_values = q_values.view(-1, 1, self.args.n_agents)  # (episode_num * max_episode_len, 1, n_agents)
        states = states.reshape(-1, self.args.state_shape)     # (episode_num * max_episode_len, state_shape)

        # Compute the first layer of the hypernetwork
        w1 = torch.abs(self.hyper_w1(states))  # (batch, n_agents * qmix_hidden_dim)
        b1 = self.hyper_b1(states)             # (batch, qmix_hidden_dim)

        w1 = w1.view(-1, self.args.n_agents, self.args.qmix_hidden_dim)  # (batch, n_agents, qmix_hidden_dim)
        b1 = b1.view(-1, 1, self.args.qmix_hidden_dim)                   # (batch, 1, qmix_hidden_dim)

        # Compute pre-activation
        pre_activation = torch.bmm(q_values, w1) + b1   # (batch, 1, qmix_hidden_dim)
        # Ensure non-negative values and apply the activation function log(1+alpha*x)
        activated = pre_activation.clamp(min=0)
        alpha = 2.5  # Alpha parameter, adjustable as needed
        hidden = torch.log(1 + alpha * activated)

        # Second layer
        w2 = torch.abs(self.hyper_w2(states))  # (batch, qmix_hidden_dim)
        b2 = self.hyper_b2(states)             # (batch, 1)

        w2 = w2.view(-1, self.args.qmix_hidden_dim, 1)  # (batch, qmix_hidden_dim, 1)
        b2 = b2.view(-1, 1, 1)                          # (batch, 1, 1)

        q_total = torch.bmm(hidden, w2) + b2  # (batch, 1, 1)
        q_total = q_total.view(episode_num, -1, 1)  # (episode_num, max_episode_len, 1)
        return q_total
