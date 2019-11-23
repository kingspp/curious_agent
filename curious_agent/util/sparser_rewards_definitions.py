class MakeRewardSparserCounter(object):
    def __init__(self, reset_steps):
        self.reset_steps = reset_steps
        self.numberOfHitsUntilReward = self.reset_steps

    def __call__(self, reward: float):
        """Returns a reward every set-number of times this function is called with a positive reward

        :param reward: a float corresponding to the reward
        :return: an integer corresponding to the sparser reward
        """
        if reward > 0:  # if the default reward is at all positive
            self.numberOfHitsUntilReward -= 1  # decrement the counter
            if self.numberOfHitsUntilReward <= 0:  # if the counter reaches 0 or becomes negative
                self.numberOfHitsUntilReward = self.reset_steps  # rest the counter
                return 1  # and return a positive reward
            else:
                return 0  # else return 0 reward


makeRewardSparserCounter = MakeRewardSparserCounter(20)  # define the functor

