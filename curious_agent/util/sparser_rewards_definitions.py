from typeguard import typechecked


class MakeRewardSparserCounter(object):
    @typechecked
    def __init__(self, reset_steps):
        self.reset_steps = reset_steps
        self.numberOfHitsUntilReward = self.reset_steps

    @typechecked
    def __call__(self, reward: float):
        """Returns a reward every set-number of times this function is called with a positive reward

        :param reward: a float corresponding to the reward
        :return: a float corresponding to the sparser reward
        """
        if reward > 0:  # if the default reward is at all positive
            self.numberOfHitsUntilReward -= 1  # decrement the counter
            if self.numberOfHitsUntilReward <= 0:  # if the counter reaches 0 or becomes negative
                self.numberOfHitsUntilReward = self.reset_steps  # rest the counter
                return 1.0  # and return a positive reward
            else:
                return 0.0  # else return 0 reward


class MakeRewardSparserThresholdBased(object):
    @typechecked
    def __init__(self, threshold: float, binary_reward=False):
        self.threshold = threshold
        self.binary_reward = binary_reward

    @typechecked
    def __call__(self, reward: float):
        """Returns a reward whenever the reward is larger than a threshold when the binary flag is set

        :param reward: a float corresponding to the reward
        :return: a float corresponding to the sparser reward
        """
        if reward >= self.threshold:
            if self.binary_reward:
                return 1.0
            else:
                return reward
        else:
            return 0.0


"""
makeRewardSparserCounter = MakeRewardSparserCounter(20)  # define the functor
"""

