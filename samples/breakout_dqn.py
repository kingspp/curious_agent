from curious_agent.pipelines import Pipeline
from curious_agent.agents.open_ai_agents.dqn_agent import DQNAgent
from curious_agent.environments.open_ai.atari_environments.open_ai_environment import AtariEnvironment
from curious_agent.util import pipeline_config_loader
import sys

# Load Configuration
config = pipeline_config_loader(sys.argv[1])

# Create Environment
env = AtariEnvironment(config['env_config'], atari_wrapper=True)
# Create Agent
agent = DQNAgent(env, config['agent_config'])

# Load the environment and agent in the pipeline
pipeline = Pipeline(train_agent=agent, environment=env, config=config)

# Run the training loop
pipeline.execute()


# pipeline.resume()
