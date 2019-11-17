from curious_agent.pipelines import Pipeline
from curious_agent.agents.open_ai_agents.dqn_agent import DQNAgent
from curious_agent.environments.open_ai.atari_environments.open_ai_environment import AtariEnvironment

env_name ='BreakoutNoFrameskip-v4'

pipeline = Pipeline(
    train_agent=DQNAgent,
    environent=AtariEnvironment(env_name, args, atari_wrapper=True))


pipeline.execute()