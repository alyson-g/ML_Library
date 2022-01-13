# Value iteration algorithm
import numpy as np

from utils.rl import bernoulli_trial


class ValueIteration:
    def __init__(self, gamma, agent, theta, uncertainty=0.2):
        self.gamma = gamma
        self.agent = agent
        self.theta = theta

        # Uncertainty in actions leading to the desired state
        self.uncertainty = uncertainty

        # Variable to hold the policy
        self.policy = None

    def fit(self):
        """
        Fit the model
        :return: None
        """
        height = self.agent.track.shape[0]
        width = self.agent.track.shape[1]

        # Enumerate all potential actions
        actions = []

        values = [-1, 0, 1]
        for i in values:
            for j in values:
                actions.append([i, j])

        # Enumerate all potential states
        states = {}

        values = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
        for i in range(height):
            for j in range(width):
                terminal_state = self.agent.track[i][j] == 'F'

                for k in values:
                    for m in values:
                        if terminal_state:
                            states[(i, j, k, m)] = 0
                        else:
                            states[(i, j, k, m)] = np.random.uniform(-1, 1, 1)[0]

        convergence_reached = False
        iterations = 1

        # Iterate until convergence is reached
        while not convergence_reached:
            delta = 0

            # Iterate over all states
            for state in states.keys():
                value = states[state]

                q = np.empty(len(actions))

                # Iterate over all actions
                for j in range(len(actions)):
                    action = actions[j]
                    q[j] = self.calculate_value(states, state, action)

                states[state] = np.max(q)

                # Check if convergence was reached
                delta = np.max([delta, np.abs(states[state] - value)])
                if delta < self.theta:
                    convergence_reached = True

                # print(f"Iteration {iterations} completed")
                iterations = iterations + 1

        # Create policy
        self.policy = {}

        for state in states.keys():
            q = np.empty(len(actions))

            # Iterate over all actions
            for j in range(len(actions)):
                action = actions[j]
                q[j] = self.calculate_value(states, state, action)

            best_action = np.argmax(q)
            self.policy[state] = actions[best_action]

    def calculate_value(self, states, state, action):
        """
        Calculate the new value, given a state and an action
        :param states: The set of all states
        :param state: The starting state
        :param action: The action to perform
        :return: The value
        """
        value = states[state]

        height = self.agent.track.shape[0]
        width = self.agent.track.shape[1]

        pos_x, pos_y, vel_x, vel_y = self.agent.perform_action(state[0], state[1], state[2], state[3],
                                                               action[0], action[1])
        # Calculate updated value
        if pos_x < 0 or pos_x >= height or pos_y < 0 or pos_y >= width:
            # Unreachable state
            return -1000
        elif self.agent.track[pos_x][pos_y] == '#':
            # Unreachable state
            return -1000
        elif vel_x == 0 and vel_y == 0:
            # Higher penalty for crashing
            return -10
        else:
            space_value = -1

            if self.agent.track[pos_x, pos_y] == 'F':
                space_value = 0

            next_state = (pos_x, pos_y, vel_x, vel_y)
            return ((1 - self.uncertainty) * (space_value + states[next_state] * self.gamma)) + \
                   (self.uncertainty * (value * self.gamma))

    def race(self, starting_position, max_iterations, verbose=False):
        """
        Run a race
        :param starting_position: The starting position of the agent
        :param max_iterations: The maximum number of iterations allowed
        :param verbose: Whether or not to print outputs
        :return: None
        """
        self.agent.set_initial_position(starting_position)
        state = (starting_position[0], starting_position[1], 0, 0)

        if verbose:
            print(f"\nInitial State: {state}")

        iterations = 0
        won = False
        while not won and iterations < max_iterations:
            # Choose next best action
            action = self.policy[state]

            success = bernoulli_trial(0.8)
            if not success:
                action = (0, 0)

            if verbose:
                print(f"Action: {action}")

            # Perform action
            pos_x, pos_y, vel_x, vel_y = self.agent.perform_action(state[0], state[1], state[2], state[3], action[0],
                                                                   action[1])
            state = (pos_x, pos_y, vel_x, vel_y)

            if verbose:
                print(f"State: {state}")

            # Check if won
            won = self.check_win(state)
            iterations = iterations + 1

        if verbose:
            print(f"Iterations: {iterations}")

        return iterations

    def check_win(self, state):
        """
        Check if the agent has won
        :param state: The state of the agent
        :return: Whether or not the agent won
        """
        pos_x = state[0]
        pos_y = state[1]

        return self.agent.track[pos_x][pos_y] == 'F'
