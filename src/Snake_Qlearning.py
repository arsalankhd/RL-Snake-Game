import numpy as np
import random
import pickle
from SnakeEnvironment import SnakeEnv

# Define Q-learning parameters
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 1.0  # Initial exploration rate
epsilon_decay = 0.995  # Decay factor per episode
epsilon_min = 0.01  # Minimum exploration rate
num_episodes = 5000  # Total training episodes

# Initialize environment
env = SnakeEnv(grid_size=10)
state_space_size = (env.grid_size, env.grid_size, 4)  # (snake_x, snake_y, direction)
action_space_size = env.action_space.n  # 4 possible actions

# Initialize Q-table
Q_table = np.zeros(state_space_size + (action_space_size,))

def get_discrete_state(state, direction):
    snake_head = state.snake[0]
    return (snake_head[0], snake_head[1], direction)

# Training loop
for episode in range(num_episodes):
    state = env.reset()
    discrete_state = get_discrete_state(env, env.snake_dir)
    done = False
    total_reward = 0
    
    while not done:
        # Epsilon-greedy action selection
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q_table[discrete_state])
        
        next_state, reward, done, _ = env.step(action)
        next_discrete_state = get_discrete_state(env, env.snake_dir)
        
        # Q-learning update rule
        Q_table[discrete_state + (action,)] = (1 - alpha) * Q_table[discrete_state + (action,)] + \
            alpha * (reward + gamma * np.max(Q_table[next_discrete_state]))
        
        discrete_state = next_discrete_state
        total_reward += reward
    
    # Decay epsilon
    epsilon = max(epsilon * epsilon_decay, epsilon_min)
    
    if episode % 500 == 0:
        print(f"Episode {episode}: Total Reward: {total_reward}")

# Save the trained Q-table
with open("q_table.pkl", "wb") as f:
    pickle.dump(Q_table, f)

print("Training completed!")

def test_agent(episodes=10):
    with open("q_table.pkl", "rb") as f:
        Q_table = pickle.load(f)
    
    for episode in range(episodes):
        state = env.reset()
        discrete_state = get_discrete_state(env, env.snake_dir)
        done = False
        total_reward = 0
        
        while not done:
            action = np.argmax(Q_table[discrete_state])
            next_state, reward, done, _ = env.step(action)
            discrete_state = get_discrete_state(env, env.snake_dir)
            total_reward += reward
            env.render()
        
        print(f"Test Episode {episode + 1}: Total Score: {env.score}")
    
    env.close()

test_agent()