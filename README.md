# CartPole Reinforcement Learning (RL) Project

## Project Overview
This project demonstrates how to use Reinforcement Learning (RL) to train an agent to balance a pole on a cart using the CartPole environment provided by Gymnasium. The goal is to have an AI agent learn to control the cart in order to keep the pole balanced for as long as possible, maximizing its total reward.

---

## What is Reinforcement Learning?
Reinforcement Learning is a type of machine learning where an agent learns to make decisions through trial and error by interacting with an environment. The agent receives observations (information about the current state), takes actions (such as moving the cart left or right), and receives rewards based on its actions. Over time, the agent learns to maximize its total reward.

Key components include:
- **Environment:** In this project, CartPole-v1 is used.
- **Agent:** The algorithm that learns to balance the pole.
- **Observation Space:** The state of the environment, e.g., cart position, cart velocity, pole angle, and pole velocity.
- **Action Space:** The possible actions available to the agent (e.g., move cart left or right).
- **Reward:** The feedback signal for the agent. Typically, the agent receives a positive reward for every time step it keeps the pole balanced, and the episode ends (or a negative reward is given) when the pole falls.

---

## Project Structure

- **agent.py**: Contains the core RL agent logic, training loop, interaction with the environment, and saving/loading models and training graphs.
- **dqn.py**: Defines the Deep Q-Network (DQN) neural network. The network receives an environment state and outputs the estimated value (Q-value) for each possible action.
- **experience_replay.py**: Implements the replay memory that stores past transitions. This allows the agent to learn more efficiently by reusing past experiences.
- **hyperparameters.yaml**: Contains all the hyperparameters (settings) used for training. These include learning rate, batch size, exploration rate parameters, and network architecture settings.
- **RL.md**: Contains additional notes and ideas about Reinforcement Learning concepts used in the project.

---

## How the Code Works (Step by Step)

### 1. Environment Setup
- The project uses the **CartPole-v1** environment from Gymnasium.
- The environment provides a continuous state (four-dimensional vector) and a discrete action space (0 for moving left, 1 for moving right).

### 2. Agent Initialization (`agent.py`)
- The `Agent` class loads hyperparameters from `hyperparameters.yaml`.
- It sets up the replay memory, initializes the policy and target DQNs, and configures the optimizer.
- If a pre-trained model exists and the agent is running in evaluation mode, that model is loaded.

### 3. Training Loop
- For each episode, the environment is reset and the agent begins a new round.
- In each time step within an episode:
  - The agent selects an action using an epsilon-greedy policy—either exploring by choosing a random action or exploiting the learned value estimates from the DQN.
  - The selected action is applied, and the environment returns the next state, reward, and a flag indicating whether the episode is done.
  - The transition (current state, action, reward, new state, and done flag) is stored in the replay memory.
  - Once there are enough experiences in memory, a mini-batch is sampled for training the policy network.
  - Periodically, the target network is synchronized with the policy network to stabilize learning.
- The agent tracks the total reward per episode and the exploration rate (epsilon).

### 4. Deep Q-Network (`dqn.py`)
- The DQN is designed with one or more fully connected layers. In this version, a single hidden layer is used (with a configurable number of nodes via `fc1_nodes`).
- The network outputs Q-values for both possible actions. The action with the highest Q-value is selected during exploitation.

### 5. Experience Replay (`experience_replay.py`)
- The replay memory stores transitions in a deque.
- This mechanism helps in breaking the correlation between consecutive experiences, which improves the stability and efficiency of the learning process.

### 6. Hyperparameters (`hyperparameters.yaml`)
Here is an example configuration for the CartPole-v1 environment:
```yaml
cartpole1:
  env_id: CartPole-v1
  replay_memory_size: 100000
  mini_batch_size: 64
  epsilon_init: 1
  epsilon_decay: 0.9995
  epsilon_min: 0.01
  network_sync_rate: 100
  learning_rate_a: 0.001
  discount_factor_g: 0.99
  stop_on_reward: 100000
  fc1_nodes: 128
  enable_double_dqn: False
  enable_dueling_dqn: True
```
This file makes it easy to adjust settings like the initial exploration rate (epsilon), epsilon decay rate, network architecture, and more.

---

## How to Run the Code

1. **Install Dependencies:**
   Ensure you have Python and the required packages installed:
   ```bash
   pip install torch gymnasium pyyaml matplotlib
   ```

2. **Start Training:**
   Run the main agent script to start training the CartPole agent:
   ```bash
   python agent.py
   ```

3. **Monitor Training:**
   - The training loop will print out the reward for each episode.
   - Graphs plotting the rewards and epsilon history will be saved to a designated `runs` directory.
   - The agent automatically saves the model checkpoints for future sessions.

4. **Evaluate:**
   To evaluate the trained agent:
   ```bash
   python agent.py --eval
   ```
   (Adjust the code as needed to use a command-line flag or a parameter to set evaluation mode.)

---

## Key RL Concepts Used

- **Epsilon-Greedy Policy:**  
  Balances exploration (random action selection) with exploitation (choosing the best-known action). Epsilon decays over time to reduce exploration.

- **Replay Memory:**  
  Stores past experiences so the agent can learn from them in a less correlated manner.

- **Target Network:**  
  A copy of the DQN that is updated less frequently. This improves stability during training by providing a more consistent target for Q-value updates.

- **Discount Factor (gamma):**  
  Represents how much future rewards are taken into account. A gamma close to 1 means future rewards are nearly as important as immediate rewards.

---

## Tips for Better Performance

- **Reward Shaping:**  
  The default CartPole rewards are straightforward, but if needed you can add small bonuses (for example, for keeping the pole balanced for additional time steps) to encourage longer survival.

- **Hyperparameter Tuning:**  
  Experiment with different values for the learning rate, epsilon decay, batch size, and network architecture to find what works best.

- **Visualization:**  
  Training graphs are generated after each training session to help you understand the agent's learning progress over time.

---

## Questions?

If you have questions or need help understanding the code, please refer to additional RL tutorials online, or feel free to reach out for guidance. The project is well-commented to help beginners understand each component of the RL framework.

---

Enjoy training your CartPole agent, and happy coding!