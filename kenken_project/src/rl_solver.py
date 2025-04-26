import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from typing import List, Tuple
from puzzle import Puzzle

class KenKenEnv:
    def __init__(self, puzzle: Puzzle):
        self.puzzle = puzzle
        self.size = puzzle.size
        self.state = self._get_state()
        self.done = False

    def _get_state(self) -> np.ndarray:
        # Represent the puzzle grid as a flattened array
        grid = self.puzzle.get_grid_copy()
        state = np.array(grid).flatten()
        return state

    def reset(self):
        # Reset puzzle to initial empty state
        self.puzzle.reset_grid()
        self.state = self._get_state()
        self.done = False
        return self.state

    def step(self, action: Tuple[int, int, int]) -> Tuple[np.ndarray, float, bool]:
        # Action is (row, col, value)
        row, col, value = action
        if self.done:
            return self.state, 0.0, True

        # Check if cell is empty
        if self.puzzle.get_cell_value(row, col) != 0:
            # Invalid move, negative reward
            reward = -1.0
            return self.state, reward, self.done

        # Check if value is valid in domain
        domain = self._get_domain(row, col)
        if value not in domain:
            reward = -1.0
            return self.state, reward, self.done

        # Place the value
        self.puzzle.set_cell_value(row, col, value)

        # Check if cage constraints are satisfied so far
        cage = self.puzzle.get_cage(row, col)
        if cage and not cage.check(self.puzzle.get_cage_values(cage)):
            # Invalid cage constraint, revert move
            self.puzzle.set_cell_value(row, col, 0)
            reward = -1.0
            return self.state, reward, self.done

        # Update state
        self.state = self._get_state()

        # Check if puzzle is solved
        if self._is_solved():
            reward = 10.0  # Large positive reward for solving
            self.done = True
        else:
            reward = 0.1  # Small positive reward for valid move

        return self.state, reward, self.done

    def _get_domain(self, row: int, col: int) -> List[int]:
        # Return possible values for cell based on row/col constraints
        possible = set(range(1, self.size + 1))
        row_values = {self.puzzle.get_cell_value(row, i) for i in range(self.size) if i != col and self.puzzle.get_cell_value(row, i) != 0}
        col_values = {self.puzzle.get_cell_value(i, col) for i in range(self.size) if i != row and self.puzzle.get_cell_value(i, col) != 0}
        possible -= row_values | col_values
        return list(possible)

    def _is_solved(self) -> bool:
        # Check if all cells are filled and all cages satisfy constraints
        for r in range(self.size):
            for c in range(self.size):
                if self.puzzle.get_cell_value(r, c) == 0:
                    return False
        for cage in self.puzzle.cages:
            if not cage.check(self.puzzle.get_cage_values(cage)):
                return False
        return True

    def _get_state(self) -> np.ndarray:
        # Represent the puzzle grid as a flattened array
        grid = self.puzzle.get_grid_copy()
        state = np.array(grid).flatten()
        return state

    def reset(self):
        # Reset puzzle to initial empty state
        for r in range(self.size):
            for c in range(self.size):
                self.puzzle.set_cell_value(r, c, 0)
        self.state = self._get_state()
        self.done = False
        return self.state

    def step(self, action: Tuple[int, int, int]) -> Tuple[np.ndarray, float, bool]:
        # Action is (row, col, value)
        row, col, value = action
        if self.done:
            return self.state, 0.0, True

        # Check if cell is empty
        if self.puzzle.get_cell_value(row, col) != 0:
            # Invalid move, negative reward
            reward = -1.0
            return self.state, reward, self.done

        # Check if value is valid in domain
        domain = self._get_domain(row, col)
        if value not in domain:
            reward = -1.0
            return self.state, reward, self.done

        # Place the value
        self.puzzle.set_cell_value(row, col, value)

        # Check if cage constraints are satisfied so far
        cage = self.puzzle.get_cage(row, col)
        if cage and not cage.check(self.puzzle.get_cage_values(cage)):
            # Invalid cage constraint, revert move
            self.puzzle.set_cell_value(row, col, 0)
            reward = -1.0
            return self.state, reward, self.done

        # Update state
        self.state = self._get_state()

        # Check if puzzle is solved
        if self._is_solved():
            reward = 10.0  # Large positive reward for solving
            self.done = True
        else:
            reward = 0.1  # Small positive reward for valid move

        return self.state, reward, self.done

    def _get_domain(self, row: int, col: int) -> List[int]:
        # Return possible values for cell based on row/col constraints
        possible = set(range(1, self.size + 1))
        row_values = {self.puzzle.get_cell_value(row, i) for i in range(self.size) if i != col and self.puzzle.get_cell_value(row, i) != 0}
        col_values = {self.puzzle.get_cell_value(i, col) for i in range(self.size) if i != row and self.puzzle.get_cell_value(i, col) != 0}
        possible -= row_values | col_values
        return list(possible)

    def _is_solved(self) -> bool:
        # Check if all cells are filled and all cages satisfy constraints
        for r in range(self.size):
            for c in range(self.size):
                if self.puzzle.get_cell_value(r, c) == 0:
                    return False
        for cage in self.puzzle.cages:
            if not cage.check(self.puzzle.get_cage_values(cage)):
                return False
        return True

class DQN(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, state_size: int, action_size: int, lr=1e-3, gamma=0.99, epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.995, batch_size=64, memory_size=10000):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQN(state_size, action_size).to(self.device)
        self.target_net = DQN(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.update_target_network()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            act_values = self.policy_net(state)
        return torch.argmax(act_values).item()

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        q_values = self.policy_net(states).gather(1, actions)
        next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

        loss = nn.MSELoss()(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def encode_action(row: int, col: int, value: int, size: int) -> int:
    # Encode (row, col, value) into a single integer action
    return row * size * size + col * size + (value - 1)

def decode_action(action: int, size: int) -> Tuple[int, int, int]:
    # Decode integer action back to (row, col, value)
    row = action // (size * size)
    rem = action % (size * size)
    col = rem // size
    value = (rem % size) + 1
    return row, col, value

class RLSolver:
    def __init__(self, puzzle: Puzzle):
        self.puzzle = puzzle
        self.env = KenKenEnv(puzzle)
        self.size = puzzle.size
        self.state_size = self.size * self.size
        self.action_size = self.size * self.size * self.size  # row * col * value
        self.agent = DQNAgent(self.state_size, self.action_size)

    def train(self, episodes: int = 1000, max_steps: int = 1000):
        for e in range(episodes):
            state = self.env.reset()
            total_reward = 0
            for step in range(max_steps):
                action_idx = self.agent.act(state)
                action = decode_action(action_idx, self.size)
                next_state, reward, done = self.env.step(action)
                self.agent.remember(state, action_idx, reward, next_state, done)
                self.agent.replay()
                state = next_state
                total_reward += reward
                if done:
                    print(f"Episode {e+1}/{episodes} finished after {step+1} steps with reward {total_reward:.2f}")
                    break
            self.agent.update_target_network()

    def solve(self):
        state = self.env.reset()
        done = False
        while not done:
            action_idx = self.agent.act(state)
            action = decode_action(action_idx, self.size)
            state, reward, done = self.env.step(action)
        return self.puzzle.get_grid_copy()
