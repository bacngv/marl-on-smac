import numpy as np
import os
import pandas as pd
from common.rollout import RolloutWorker, CommRolloutWorker
from agent.agent import Agents, CommAgents
from common.replay_buffer import ReplayBuffer
import matplotlib.pyplot as plt

class Runner:
    def __init__(self, env, args):
        self.env = env

        if args.alg.find('commnet') > -1 or args.alg.find('g2anet') > -1:  # communication agent
            self.agents = CommAgents(args)
            self.rolloutWorker = CommRolloutWorker(env, self.agents, args)
        else:  # no communication agent
            self.agents = Agents(args)
            self.rolloutWorker = RolloutWorker(env, self.agents, args)
        if not args.evaluate and args.alg.find('coma') == -1 and args.alg.find('central_v') == -1 and args.alg.find('reinforce') == -1:
            self.buffer = ReplayBuffer(args)
        self.args = args
        self.win_rates = []
        self.episode_rewards = []
        self.loss_history = []      # Store the loss for each training step (training_step, loss)
        self.loss_avg_history = []  # Store the average loss every 100 training steps (training_step, avg_loss)
        self.eval_timesteps = []    # Store evaluation timesteps

        # Path to save results
        self.save_path = os.path.join(self.args.result_dir, args.alg, args.map)
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def run(self, num):
        time_steps, train_steps, evaluate_steps = 0, 0, -1
        while time_steps < self.args.n_steps:
            print('Run {}, time_steps {}'.format(num, time_steps))
            if time_steps // self.args.evaluate_cycle > evaluate_steps:
                win_rate, episode_reward = self.evaluate()
                self.win_rates.append(win_rate)
                self.episode_rewards.append(episode_reward)
                self.eval_timesteps.append(time_steps)
                self.plt(num)
                evaluate_steps += 1
            episodes = []
            # Collect self.args.n_episodes episodes
            for episode_idx in range(self.args.n_episodes):
                episode, _, _, steps = self.rolloutWorker.generate_episode(episode_idx)
                episodes.append(episode)
                time_steps += steps
            # Merge episodes together
            episode_batch = episodes[0]
            episodes.pop(0)
            for episode in episodes:
                for key in episode_batch.keys():
                    episode_batch[key] = np.concatenate((episode_batch[key], episode[key]), axis=0)
            # Train agent
            if self.args.alg.find('coma') > -1 or self.args.alg.find('central_v') > -1 or self.args.alg.find('reinforce') > -1:
                loss = self.agents.train(episode_batch, train_steps, self.rolloutWorker.epsilon)
                self.loss_history.append((train_steps, loss.item()))
                train_steps += 1
                # After every 100 training steps, calculate the average loss of the last 100 steps and plot the graph
                if train_steps % 100 == 0:
                    last_100 = [l for s, l in self.loss_history[-100:]]
                    avg_loss = np.mean(last_100)
                    print("Average loss at training step {}: {:.6f}".format(train_steps, avg_loss))
                    self.loss_avg_history.append((train_steps, avg_loss))
                    self.plot_loss(num)
            else:
                self.buffer.store_episode(episode_batch)
                for _ in range(self.args.train_steps):
                    mini_batch = self.buffer.sample(min(self.buffer.current_size, self.args.batch_size))
                    loss = self.agents.train(mini_batch, train_steps)
                    self.loss_history.append((train_steps, loss.item()))
                    train_steps += 1
                    if train_steps % 100 == 0:
                        last_100 = [l for s, l in self.loss_history[-100:]]
                        avg_loss = np.mean(last_100)
                        print("Average loss at training step {}: {:.6f}".format(train_steps, avg_loss))
                        self.loss_avg_history.append((train_steps, avg_loss))
                        self.plot_loss(num)
            # If you do not want to plot loss after each batch, comment out the line below
            # self.plot_loss(num)
        win_rate, episode_reward = self.evaluate()
        print('win_rate is ', win_rate)
        self.win_rates.append(win_rate)
        self.episode_rewards.append(episode_reward)
        self.eval_timesteps.append(time_steps)
        self.plt(num)
        self.plot_loss(num)

    def evaluate(self):
        win_number = 0
        episode_rewards = 0
        for epoch in range(self.args.evaluate_epoch):
            _, episode_reward, win_tag, _ = self.rolloutWorker.generate_episode(epoch, evaluate=True)
            episode_rewards += episode_reward
            if win_tag:
                win_number += 1
        return win_number / self.args.evaluate_epoch, episode_rewards / self.args.evaluate_epoch

    def plt(self, num):
        # Plot win_rates and episode_rewards over evaluation timesteps
        x = self.eval_timesteps

        plt.figure(figsize=(12, 8))

        # --- Subplot 1: Win Rate ---
        plt.subplot(2, 1, 1)
        plt.plot(x, self.win_rates, label='Win Rate')
        plt.xlabel('Timesteps')
        plt.ylabel('Win Rate')
        plt.ylim(0, 1.0)
        plt.legend()

        # --- Subplot 2: Episode Rewards ---
        plt.subplot(2, 1, 2)
        plt.plot(x, self.episode_rewards, label='Episode Rewards')
        plt.xlabel('Timesteps')
        plt.ylabel('Episode Rewards')
        plt.legend()

        plt.tight_layout()

        # Save figure
        plt.savefig(os.path.join(self.save_path, f'plt_{num}.png'), format='png')
        # Save evaluation data
        data = pd.DataFrame({
            'time_steps': x,
            'win_rate': self.win_rates,
            'episode_rewards': self.episode_rewards
        })
        data.to_csv(os.path.join(self.save_path, f'results_{num}.csv'), index=False)
        plt.close()

    def plot_loss(self, num):
        # Plot the average loss (every 100 training steps)
        if len(self.loss_avg_history) == 0:
            print("No loss history to plot.")
            return

        steps = [s for s, avg in self.loss_avg_history]
        avg_losses = [avg for s, avg in self.loss_avg_history]

        plt.figure(figsize=(12, 8))
        plt.plot(steps, avg_losses, label='Average Loss per 100 steps', marker='o')
        plt.xlabel("Training Steps")
        plt.ylabel("Average Loss")
        plt.title("Training Loss History (averaged every 100 steps)")
        plt.legend()
        plt.tight_layout()

        loss_plot_path = os.path.join(self.save_path, f'loss_plot_{num}.png')
        plt.savefig(loss_plot_path, format='png')
        plt.close()

        # Save the averaged loss history to a CSV file
        df = pd.DataFrame({"training_step": steps, "avg_loss": avg_losses})
        csv_path = os.path.join(self.save_path, f'loss_history_{num}.csv')
        df.to_csv(csv_path, index=False)
