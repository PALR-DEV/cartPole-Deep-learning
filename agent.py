import gymnasium
import flappy_bird_gymnasium 
import numpy as np
import random
import torch
from torch import nn
import yaml
from experience_replay import ReplayMemory
from dqn import DQN
import os
import itertools
from datetime import datetime, timedelta
import argparse
import matplotlib
import matplotlib.pyplot as plt

# Directory for saving run info
RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)

matplotlib.use('Agg')

# Deep Q-Learning Agent
class Agent():
    def __init__(self, hyperparameter_set):
        with open('hyperparameters.yaml', 'r') as file:
            all_hyperparameter_sets = yaml.safe_load(file)
            hyperparameters = all_hyperparameter_sets[hyperparameter_set]
            # print(hyperparameters)

        self.hyperparameter_set = hyperparameter_set

        # Hyperparameters (adjustable)
        self.env_id             = hyperparameters['env_id']
        self.learning_rate_a    = hyperparameters['learning_rate_a']        # learning rate (alpha)
        self.discount_factor_g  = hyperparameters['discount_factor_g']      # discount rate (gamma)
        self.network_sync_rate  = hyperparameters['network_sync_rate']      # number of steps the agent takes before syncing the policy and target network
        self.replay_memory_size = hyperparameters['replay_memory_size']     # size of replay memory
        self.mini_batch_size    = hyperparameters['mini_batch_size']        # size of the training data set sampled from the replay memory
        self.epsilon_init       = hyperparameters['epsilon_init']           # 1 = 100% random actions
        self.epsilon_decay      = hyperparameters['epsilon_decay']          # epsilon decay rate
        self.epsilon_min        = hyperparameters['epsilon_min']            # minimum epsilon value
        self.stop_on_reward     = hyperparameters.get('stop_on_reward', float('inf')) # stop training after reaching this number of rewards
        self.fc1_nodes          = hyperparameters['fc1_nodes']
        self.env_make_params    = hyperparameters.get('env_make_params',{}) # Get optional environment-specific parameters, default to empty dict
        self.enable_double_dqn  = hyperparameters.get('enable_double_dqn', False)      # double dqn on/off flag
        self.enable_dueling_dqn = hyperparameters.get('enable_dueling_dqn', False)     # dueling dqn on/off flag

        # Neural Network
        self.loss_fn = nn.MSELoss()          # NN Loss function. MSE=Mean Squared Error can be swapped to something else.
        self.optimizer = None                # NN Optimizer. Initialize later.

        # Path to Run info
        self.LOG_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.log')
        self.MODEL_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.pt')
        self.GRAPH_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.png')

        self.gamma = hyperparameters.get('gamma', self.discount_factor_g) # fallback to discount_factor_g if gamma is missing


    def run(self, is_training = True, render  = False):
        env = gymnasium.make(self.env_id, render_mode='human' if render else None, **self.env_make_params)

        num_states = env.observation_space.shape[0] # type: ignore
        num_actions = env.action_space.n # type: ignore
        reward_per_episode = []
        epsilon_history = []

        policy_dqn = DQN(num_states, num_actions, self.fc1_nodes) 
        # Load pre-trained model if available
        if not is_training and os.path.exists(self.MODEL_FILE):
            print(f"Loading pre-trained model from {self.MODEL_FILE}")
            policy_dqn.load_state_dict(torch.load(self.MODEL_FILE))

        if is_training:
            #create replay memory
            replay_memory = ReplayMemory(self.replay_memory_size)

            epsilon = self.epsilon_init

            # Create target DQN
            target_dqn = DQN(num_states, num_actions, self.fc1_nodes)

            #copy the weights of the policy DQN to the target DQN
            target_dqn.load_state_dict(policy_dqn.state_dict())

            #track number of steps taken, used for scyncing policy => tareget network
            step_count = 0

            self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a)




        
        try:
            for episode in itertools.count():
                state, _ = env.reset()
                state = torch.tensor(state, dtype=torch.float)  # Add batch dimension

                terminated = False
                truncated = False
                episode_reward = 0.0
                while not (terminated or truncated):
                    if is_training and random.random() < epsilon:
                        action = env.action_space.sample()
                        action = torch.tensor(action, dtype=torch.int64)

                    else:
                        with torch.no_grad():
                            #tensor ([1,2,3, ...]) -> tensor ([[1,2,3, ...]])
                            action  = policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax()

                    new_state, reward, terminated, truncated, info = env.step(action.item()) 
                    episode_reward += float(reward)

                    #convert new state and reward to tensors on device
                    new_state = torch.tensor(new_state, dtype=torch.float)
                    reward = torch.tensor(reward, dtype=torch.float)

                    if is_training:
                        replay_memory.append((state, action, reward, new_state, terminated or truncated))
                        

                        step_count += 1
                        #sync policy and target network every 1000 steps
                        if step_count % self.network_sync_rate == 0:
                            target_dqn.load_state_dict(policy_dqn.state_dict())

                    #move to new state
                    state = new_state

                reward_per_episode.append(episode_reward)
                print(f"Episode {episode + 1}: Reward = {episode_reward}")

                if is_training:
                    epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
                    epsilon_history.append(epsilon)

                #if enough experiences in replay memory has been collected
                if is_training and len(replay_memory)>self.mini_batch_size:
                    #sample from memory 
                    mini_batch = replay_memory.sample(self.mini_batch_size)
                    self.optimize(mini_batch, policy_dqn, target_dqn)

                    if step_count > self.network_sync_rate:
                        #sync policy and target network
                        target_dqn.load_state_dict(policy_dqn.state_dict())


        except KeyboardInterrupt:
            print("\nTraining interrupted by user. Saving model and graphs...")
        finally:
            print(f"Saving model to {self.MODEL_FILE} and graphs to {self.GRAPH_FILE}")
            torch.save(policy_dqn.state_dict(), self.MODEL_FILE)
            if self.optimizer is not None:
                torch.save(self.optimizer.state_dict(), self.MODEL_FILE.replace('.pt', '_optimizer.pt'))
            self.save_graph(reward_per_episode, epsilon_history)
            print(f"Model and graphs saved!")

    def optimize(self, mini_batch, policy_dqn, target_dqn):
        """
        Optimize the policy DQN using a mini-batch sampled from the replay memory.
        :param mini_batch: A list of transitions sampled from the replay memory.
        :param policy_dqn: The policy DQN to be optimized.
        :param target_dqn: The target DQN used for calculating the target Q-values.
        """
        if self.optimizer is None:
            raise RuntimeError("Optimizer has not been initialized. Make sure to call optimize only after initializing the optimizer in training mode.")

        states, actions, rewards, next_states, terminations = zip(*mini_batch)

        states = torch.stack(states)
        actions = torch.stack(actions)
        rewards = torch.stack(rewards)
        next_states = torch.stack(next_states)
        terminations = torch.tensor(terminations, dtype=torch.float)

        # Compute Q-values for current states
        q_values = policy_dqn(states).gather(1, actions.unsqueeze(1)).squeeze()

        # Compute target Q-values
        with torch.no_grad():
            next_q_values = target_dqn(next_states).max(1)[0]
            target_q_values = rewards + self.gamma * (1 - terminations) * next_q_values

        # Compute loss
        loss = (q_values - target_q_values).pow(2).mean()

        # Update policy DQN
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save_graph(self, reward_per_episode, epsilon_history):
        """
        Save the training progress graph (rewards and epsilon over episodes).
        :param reward_per_episode: List of total rewards per episode.
        :param epsilon_history: List of epsilon values over episodes.
        """
        import matplotlib.pyplot as plt

        # Plot rewards
        plt.figure(figsize=(10, 5))
        plt.plot(reward_per_episode, label='Reward per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('Training Progress - Rewards')
        plt.legend()
        plt.grid()
        plt.savefig(self.GRAPH_FILE.replace('.png', '_rewards.png'))
        plt.close()

        # Plot epsilon
        plt.figure(figsize=(10, 5))
        plt.plot(epsilon_history, label='Epsilon (Exploration Rate)', color='orange')
        plt.xlabel('Episode')
        plt.ylabel('Epsilon')
        plt.title('Training Progress - Epsilon')
        plt.legend()
        plt.grid()
        plt.savefig(self.GRAPH_FILE.replace('.png', '_epsilon.png'))
        plt.close()

        # Combined plot
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(reward_per_episode, label='Reward per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('Training Progress - Rewards')
        plt.legend()
        plt.grid()

        plt.subplot(1, 2, 2)
        plt.plot(epsilon_history, label='Epsilon (Exploration Rate)', color='orange')
        plt.xlabel('Episode')
        plt.ylabel('Epsilon')
        plt.title('Training Progress - Epsilon')
        plt.legend()
        plt.grid()

        plt.tight_layout()
        plt.savefig(self.GRAPH_FILE)
        plt.close()

        print(f"Graphs saved: {self.GRAPH_FILE.replace('.png', '_rewards.png')}, {self.GRAPH_FILE.replace('.png', '_epsilon.png')}, {self.GRAPH_FILE}")



if __name__ == "__main__":
    agent = Agent('cartpole1')
    agent.run(is_training=True, render=True)