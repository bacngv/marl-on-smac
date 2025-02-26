import numpy as np
import torch
from torch.distributions import Categorical


# Agent without communication
class Agents:
    def __init__(self, args):
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        if args.alg == 'vdn':
            from policy.vdn import VDN
            self.policy = VDN(args)
        elif args.alg == 'iql':
            from policy.iql import IQL
            self.policy = IQL(args)
        elif args.alg == 'qmix':
            from policy.qmix import QMIX
            self.policy = QMIX(args)
        elif args.alg == 'coma':
            from policy.coma import COMA
            self.policy = COMA(args)
        elif args.alg == 'qtran_alt':
            from policy.qtran_alt import QtranAlt
            self.policy = QtranAlt(args)
        elif args.alg == 'qtran_base':
            from policy.qtran_base import QtranBase
            self.policy = QtranBase(args)
        elif args.alg == 'maven':
            from policy.maven import MAVEN
            self.policy = MAVEN(args)
        elif args.alg == 'central_v':
            from policy.central_v import CentralV
            self.policy = CentralV(args)
        elif args.alg == 'reinforce':
            from policy.reinforce import Reinforce
            self.policy = Reinforce(args)
        else:
            raise Exception("No such algorithm")
        self.args = args

    def choose_action(self, obs, last_action, agent_num, avail_actions, epsilon, maven_z=None):
        inputs = obs.copy()
        avail_actions_ind = np.nonzero(avail_actions)[0]  # indices of actions that can be chosen

        # Convert agent number to one-hot vector
        agent_id = np.zeros(self.n_agents)
        agent_id[agent_num] = 1.0

        if self.args.last_action:
            inputs = np.hstack((inputs, last_action))
        if self.args.reuse_network:
            inputs = np.hstack((inputs, agent_id))
        hidden_state = self.policy.eval_hidden[:, agent_num, :]

        # Reshape inputs from (42,) to (1, 42)
        inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
        avail_actions = torch.tensor(avail_actions, dtype=torch.float32).unsqueeze(0)
        if self.args.cuda:
            inputs = inputs.cuda()
            hidden_state = hidden_state.cuda()

        # Get Q value
        if self.args.alg == 'maven':
            maven_z = torch.tensor(maven_z, dtype=torch.float32).unsqueeze(0)
            if self.args.cuda:
                maven_z = maven_z.cuda()
            q_value, self.policy.eval_hidden[:, agent_num, :] = self.policy.eval_rnn(inputs, hidden_state, maven_z)
        else:
            q_value, self.policy.eval_hidden[:, agent_num, :] = self.policy.eval_rnn(inputs, hidden_state)

        # Choose action based on Q value
        if self.args.alg in ['coma', 'central_v', 'reinforce']:
            action = self._choose_action_from_softmax(q_value.cpu(), avail_actions, epsilon)
        else:
            q_value[avail_actions == 0.0] = -float("inf")
            if np.random.uniform() < epsilon:
                action = np.random.choice(avail_actions_ind)  # action is an integer
            else:
                action = torch.argmax(q_value)
        return action

    def _choose_action_from_softmax(self, inputs, avail_actions, epsilon):
        """
        :param inputs: Q values for all actions
        """
        action_num = avail_actions.sum(dim=1, keepdim=True).float().repeat(1, avail_actions.shape[-1])  # number of available actions
        # First, convert the actor network's output into a probability distribution via softmax
        prob = torch.nn.functional.softmax(inputs, dim=-1)
        # Add epsilon noise to the probability distribution
        prob = ((1 - epsilon) * prob + torch.ones_like(prob) * epsilon / action_num)
        prob[avail_actions == 0] = 0.0  # Set probability of unavailable actions to 0

        """
        After setting the probability of unavailable actions to 0, the sum of probabilities in 'prob'
        is not equal to 1. This is acceptable because torch.distributions.Categorical will normalize it.
        Note that during training, Categorical is not used, so the probability corresponding to the executed action
        needs to be renormalized.
        """
        
        action = Categorical(prob).sample().long()
        return action

    def _get_max_episode_len(self, batch):
        terminated = batch['terminated']
        episode_num = terminated.shape[0]
        max_episode_len = 0
        for episode_idx in range(episode_num):
            for transition_idx in range(self.args.episode_limit):
                if terminated[episode_idx, transition_idx, 0] == 1:
                    if transition_idx + 1 >= max_episode_len:
                        max_episode_len = transition_idx + 1
                    break
        if max_episode_len == 0:  # In case none of the episodes have ended, so no 1 exists in 'terminated'
            max_episode_len = self.args.episode_limit
        return max_episode_len

    def train(self, batch, train_step, epsilon=None):  # COMA requires epsilon during training
        # Since episodes have different lengths, determine the maximum episode length in the batch
        max_episode_len = self._get_max_episode_len(batch)
        for key in batch.keys():
            if key != 'z':
                batch[key] = batch[key][:, :max_episode_len]
        loss = self.policy.learn(batch, max_episode_len, train_step, epsilon)
        if train_step > 0 and train_step % self.args.save_cycle == 0:
            self.policy.save_model(train_step)
        return loss


# Agent with communication
class CommAgents:
    def __init__(self, args):
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        alg = args.alg
        if alg.find('reinforce') > -1:
            from policy.reinforce import Reinforce
            self.policy = Reinforce(args)
        elif alg.find('coma') > -1:
            from policy.coma import COMA
            self.policy = COMA(args)
        elif alg.find('central_v') > -1:
            from policy.central_v import CentralV
            self.policy = CentralV(args)
        else:
            raise Exception("No such algorithm")
        self.args = args
        print('Init CommAgents')

    # Determine probabilities from weights and select an action based on epsilon
    def choose_action(self, weights, avail_actions, epsilon):
        weights = weights.unsqueeze(0)
        avail_actions = torch.tensor(avail_actions, dtype=torch.float32).unsqueeze(0)
        action_num = avail_actions.sum(dim=1, keepdim=True).float().repeat(1, avail_actions.shape[-1])  # number of available actions
        # Convert the actor network's output into a probability distribution via softmax
        prob = torch.nn.functional.softmax(weights, dim=-1)
        # Add noise to the probability distribution during training
        prob = ((1 - epsilon) * prob + torch.ones_like(prob) * epsilon / action_num)
        prob[avail_actions == 0] = 0.0  # Set probability of unavailable actions to 0

        """
        After setting the probability of unavailable actions to 0, the sum of probabilities in 'prob'
        is not equal to 1. This is acceptable because torch.distributions.Categorical will normalize it.
        Note that during training, Categorical is not used, so the probability corresponding to the executed action
        needs to be renormalized.
        """

        action = Categorical(prob).sample().long()
        return action

    def get_action_weights(self, obs, last_action):
        obs = torch.tensor(obs, dtype=torch.float32)
        last_action = torch.tensor(last_action, dtype=torch.float32)
        inputs = list()
        inputs.append(obs)
        # Append the last action and agent identity to the observation
        if self.args.last_action:
            inputs.append(last_action)
        if self.args.reuse_network:
            inputs.append(torch.eye(self.args.n_agents))
        inputs = torch.cat(inputs, dim=1)
        if self.args.cuda:
            inputs = inputs.cuda()
            self.policy.eval_hidden = self.policy.eval_hidden.cuda()
        weights, self.policy.eval_hidden = self.policy.eval_rnn(inputs, self.policy.eval_hidden)
        weights = weights.reshape(self.args.n_agents, self.args.n_actions)
        return weights.cpu()

    def _get_max_episode_len(self, batch):
        terminated = batch['terminated']
        episode_num = terminated.shape[0]
        max_episode_len = 0
        for episode_idx in range(episode_num):
            for transition_idx in range(self.args.episode_limit):
                if terminated[episode_idx, transition_idx, 0] == 1:
                    if transition_idx + 1 >= max_episode_len:
                        max_episode_len = transition_idx + 1
                    break
        if max_episode_len == 0:  # In case none of the episodes have ended, so no 1 exists in 'terminated'
            max_episode_len = self.args.episode_limit
        return max_episode_len

    def train(self, batch, train_step, epsilon=None):  # COMA also requires epsilon for computing action probabilities during training
        # Since each episode can have a different length, use the longest episode in the batch for training
        max_episode_len = self._get_max_episode_len(batch)
        for key in batch.keys():
            batch[key] = batch[key][:, :max_episode_len]
        loss = self.policy.learn(batch, max_episode_len, train_step, epsilon)
        if train_step > 0 and train_step % self.args.save_cycle == 0:
            self.policy.save_model(train_step)
        return loss
