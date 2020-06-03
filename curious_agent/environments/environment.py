class Environment(object):
    def __init__(self):
        pass
        
    def seed(self, seed):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def step(self,action):
        raise NotImplementedError

    def get_action_space(self):
        raise NotImplementedError

    def get_observation_space(self):
        raise NotImplementedError

    def get_random_action(self):
        raise NotImplementedError
