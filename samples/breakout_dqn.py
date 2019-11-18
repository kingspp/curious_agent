from curious_agent.pipelines import Pipeline
from curious_agent.agents.open_ai_agents.dqn_agent import DQNAgent
from curious_agent.environments.open_ai.atari_environments.open_ai_environment import AtariEnvironment
from curious_agent.util import pipeline_config_loader
import sys

config = pipeline_config_loader(sys.argv[1])

env = AtariEnvironment(config['env_config'], atari_wrapper=True)
agent = DQNAgent(env, config['agent_config'])



pipeline = Pipeline(train_agent=agent, environment=env, config=config)



pipeline.execute()
