# Defines the agent class
from crash_type import CrashType


class Agent:
    def __init__(self, track, crash_type, initial_position=None):
        self.track = track
        self.crash_type = crash_type
        self.initial_position = initial_position

    def set_initial_position(self, position):
        """
        Set the agent's initial position
        :param position: The position to set the agent to
        :return: None
        """
        self.initial_position = position

    def perform_action(self, pos_x, pos_y, vel_x, vel_y, ac_x, ac_y):
        """
        Perform an action
        :param pos_x: The x position
        :param pos_y: The y position
        :param vel_x: The current velocity of x
        :param vel_y: The current velocity of y
        :param ac_x: The acceleration to be applied to x
        :param ac_y: The acceleration to be applied to y
        :return: The result of the action
        """
        next_vel_x = 0
        next_vel_y = 0
        next_pos_x = 0
        next_pos_y = 0

        # Update x acceleration and position
        if not(vel_x + ac_x < -5 or vel_x + ac_x > 5):
            next_vel_x = vel_x + ac_x
            next_pos_x = pos_x + next_vel_x

        # Update y acceleration and position
        if not (vel_y + ac_y < -5 or vel_y + ac_y > 5):
            next_vel_y = vel_y + ac_y
            next_pos_y = pos_y + next_vel_y

        # Check for out of bounds position and correct
        next_pos_x, next_pos_y, corrected = self.check_and_correct(pos_x, pos_y, next_pos_x, next_pos_y)

        if corrected:
            next_vel_x = 0
            next_vel_y = 0

        return next_pos_x, next_pos_y, next_vel_x, next_vel_y

    def check_and_correct(self, pos_x, pos_y, next_pos_x, next_pos_y):
        """
        Check and correct the agent's current position
        :param pos_x: The agent's previous x position
        :param pos_y: The agent's previous y position
        :param next_pos_x: The agent's next x position
        :param next_pos_y: The agent's next y position
        :return: The corrected position
        """
        height = len(self.track)
        width = len(self.track[0])

        # Check position
        error_flag = False

        error_flag = error_flag or next_pos_x < 0 or next_pos_x > height - 1
        error_flag = error_flag or next_pos_y < 0 or next_pos_y > width - 1
        error_flag = error_flag or self.track[next_pos_x][next_pos_y] == '#'

        # Check and correct position
        corrected_x, corrected_y = self.calculate_bresenham_line([pos_x, pos_y], [next_pos_x, next_pos_y])
        error_flag = error_flag or not (corrected_x == next_pos_x) or not (corrected_y == next_pos_y)

        # If a crash has been detected, correct position
        if error_flag:
            if self.crash_type == CrashType.RESET:
                return self.initial_position[0], self.initial_position[1], True
            else:
                return corrected_x, corrected_y, True
        else:
            return next_pos_x, next_pos_y, False

    def calculate_bresenham_line(self, prev_position, next_position):
        """
        Calculate the position closest to the crash site

        Algorithm adapted from https://www.geeksforgeeks.org/bresenhams-line-generation-algorithm/
        :param prev_position: The starting position of the agent before the crash
        :param next_position: The ending position of the agent after the crash
        :return: The agent's new position
        """
        reverse_flag = False

        if next_position[0] <= prev_position[0]:
            x1 = next_position[0]
            y1 = next_position[1]

            x2 = prev_position[0]
            y2 = prev_position[1]
        else:
            reverse_flag = True

            x1 = prev_position[0]
            y1 = prev_position[1]

            x2 = next_position[0]
            y2 = next_position[1]

        indices = list(range(x1, x2 + 1))

        if reverse_flag:
            indices = reversed(indices)

            m_new = 2 * (y2 - y1)
            slope_error_new = m_new + (x2 - x1)

            y = y2
        else:
            m_new = 2 * (y2 - y1)
            slope_error_new = m_new - (x2 - x1)

            y = y1

        best_x = None
        best_y = None

        if m_new == 0:
            y = y2
            for x in indices:
                if x >= len(self.track):
                    continue
                if not (self.track[x][y] == '#'):
                    if best_x is None and best_y is None:
                        best_x = x
                        best_y = y
                else:
                    best_x = None
                    best_y = None
        else:
            for x in indices:
                if not (x < 0 or x >= len(self.track) or y < 0 or y >= len(self.track[0])):
                    if not (self.track[x][y] == '#'):
                        if best_x is None and best_y is None:
                            best_x = x
                            best_y = y
                    else:
                        best_x = None
                        best_y = None

                # Add slope to increment angle formed
                slope_error_new = slope_error_new + m_new

                # Slope error reached limit, time to
                # increment y and update slope error.
                if slope_error_new >= 0:
                    if reverse_flag:
                        y = y - 1
                        slope_error_new = slope_error_new + 2 * (x2 - x1)
                    else:
                        y = y + 1
                        slope_error_new = slope_error_new - 2 * (x2 - x1)

        if best_x is None or best_y is None:
            return prev_position[0], prev_position[1]
        else:
            return best_x, best_y
