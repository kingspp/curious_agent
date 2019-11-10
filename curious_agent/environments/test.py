from atari_environments import EnvironmentWrapper

# test
if __name__ == '__main__':
    env_name = 'BreakoutNoFrameskip-v4'
    env = EnvironmentWrapper(env_name, atari_wrapper=True)
