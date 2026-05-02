import numpy as np
from gymnasium.spaces import Box
from gymnasium.spaces import Discrete
from gymnasium.spaces import MultiDiscrete
from gymnasium.spaces import Tuple


class Base_Action(object):
    """ """

    def __init__(self, config):
        return

    def __call__(self, *args, **kargs):
        return self.get_action(*args, **kargs)

    def get_action(self, action):
        """

        :param action:

        """
        return action
