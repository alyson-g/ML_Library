# Q Learning algorithm
import matplotlib.pyplot as plt
import numpy as np

from crash_type import CrashType
from utils.rl import bernoulli_trial


class QLearning:
    def __init__(self, agent, gamma, eta, epsilon, decay=0.999):
        self.agent = agent
        self.gamma = gamma
        self.eta = eta
        self.epsilon = epsilon
        self.decay = decay

        height = self.agent.track.shape[0]
        width = self.agent.track.shape[1]

        # Enumerate all potential actions
        self.actions = []

        values = [-1, 0, 1]
        for i in values:
            for j in values:
                self.actions.append((i, j))

        # Enumerate all potential states
        self.states = []

        values = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
        for i in range(height):
            for j in range(width):
                for k in values:
                    for m in values:
                        self.states.append((i, j, k, m))

        # Initialize q
        self.q = {}

        for state in self.states:
            x = state[0]
            y = state[1]

            terminal_state = self.agent.track[x][y] == 'F'
            invalid_state = self.agent.track[x][y] == '#'

            for action in self.actions:
                if terminal_state:
                    self.q[(state, action)] = 0
                elif invalid_state:
                    self.q[(state, action)] = -10
                else:
                    self.q[(state, action)] = np.random.uniform(-1, 1, 1)[0]

    def fit(self, starting_positions, num_episodes=1000, verbose=False):
        """
        Fit the model
        :param starting_positions: The set of potential starting positions of the agent
        :param num_episodes: The number of episodes to complete
        :param verbose: Whether or not to print outputs
        :return: None
        """
        total_iterations = []

        episodes = 1
        while episodes <= num_episodes:
            # Randomly choose starting position
            index = np.random.randint(0, len(starting_positions))
            starting_position = starting_positions[index]

            x = starting_position[0]
            y = starting_position[1]
            state = (x, y, 0, 0)

            won = False
            iterations = 1

            if self.agent.crash_type == CrashType.RESET:
                max_iterations = 15000

            # Continue to perform actions until the agent has won
            while not won:
                # Pick next action
                action = self.epsilon_greedy(state)

                success = bernoulli_trial(0.8)
                if not success:
                    action = (0, 0)

                pos_x, pos_y, vel_x, vel_y = self.agent.perform_action(state[0], state[1], state[2], state[3],
                                                                       action[0], action[1])
                new_state = (pos_x, pos_y, vel_x, vel_y)

                # Calculate reward
                reward = -1

                if self.agent.track[pos_x][pos_y] == 'F':
                    reward = 0
                if self.agent.track[pos_x][pos_y] == '#':
                    # Invalid state
                    reward = -1000

                # Find next best action
                next_action = self.best_action(new_state)

                old_q = self.q[(state, action)]
                self.q[(state, action)] = old_q + self.eta * (reward + self.gamma * self.q[(new_state, next_action)] - old_q)

                # Check for win
                won = self.check_win(state, new_state)
                state = new_state

                # print(f"Iterations: {iterations}")
                iterations = iterations + 1

                if self.agent.crash_type == CrashType.RESET:
                    if iterations >= max_iterations:
                        won = True

            self.epsilon = self.epsilon * self.decay
            self.eta = self.eta * self.decay

            if verbose:
                print(f"Episodes: {episodes}, Iterations: {iterations}")

            episodes = episodes + 1
            total_iterations.append(iterations)

        return total_iterations

    def check_win(self, prev_state, curr_state):
        """
        Check if the agent has won
        :param prev_state: The agent's previous state
        :param curr_state: The agent's current state
        :return: Whether or not the agent has won
        """
        if prev_state is None:
            x = curr_state[0]
            y = curr_state[1]

            return self.agent.track[x][y] == 'F'
        else:
            x = curr_state[0]
            y = curr_state[1]

            return self.agent.track[x][y] == 'F'

    def epsilon_greedy(self, state):
        """
        Use the epsilon greedy search to choose the next action
        :param state: The agent's current state
        :return: The action that should be taken
        """
        # Determine whether to explore or exploit
        exploit = bernoulli_trial(self.epsilon)

        if exploit:
            # Choose best action
            return self.best_action(state)
        else:
            # Choose randomly
            index = np.random.randint(0, len(self.actions), 1)[0]
            return self.actions[index]

    def best_action(self, state):
        """
        Determine the best action
        :param state: The current state
        :return: The best action
        """
        # Choose best action
        best_action = self.actions[0]
        best_value = self.q[(state, self.actions[0])]

        for action in self.actions:
            value = self.q[(state, action)]
            if value > best_value:
                best_action = action

        return best_action

    def race(self, starting_position, max_iterations, verbose=False):
        """
        Simulate a race
        :param starting_position: The starting position of the agent
        :param max_iterations: The maximum number of iterations allowed
        :param verbose: Whether or not to print outputs
        :return: None
        """
        self.agent.set_initial_position(starting_position)

        x = starting_position[0]
        y = starting_position[1]

        # Initial state
        state = (x, y, 0, 0)

        if verbose:
            print(f"\nInitial State: {state}")

        won = False
        iterations = 0

        while not won and iterations < max_iterations:
            action = self.best_action(state)

            success = bernoulli_trial(0.8)
            if not success:
                action = (0, 0)

            if verbose:
                print(f"Action: {action}")

            pos_x, pos_y, vel_x, vel_y = self.agent.perform_action(state[0], state[1], state[2], state[3],
                                                                   action[0], action[1])
            new_state = (pos_x, pos_y, vel_x, vel_y)

            won = self.check_win(state, new_state)
            state = new_state

            if verbose:
                print(f"State: {state}")

            iterations = iterations + 1

        if verbose:
            print(f"Iterations: {iterations}")

        return iterations

    def generate_learning_curve(self, starting_position, track_name, num_episodes=10000):
        """
        Generate a learning curve
        :param starting_position: The starting position of the agent
        :param track_name: The name of the track the agent is racing on
        :param num_episodes: The total number of episodes to train over
        :return: None
        """
        iterations = self.fit(starting_position, num_episodes=num_episodes)

        x = list(range(num_episodes))

        # Generate learning curve
        plt.plot(x, iterations)
        plt.title(f"Q-Learning {track_name}")
        plt.xlabel('Episode #')
        plt.ylabel('Iterations')

        # Save graph to image file
        mod_track_name = track_name.lower()
        plt.savefig(f"q_learning_{mod_track_name}_learning_curve.png", bbox_inches='tight')

        # Show image
        plt.show()
