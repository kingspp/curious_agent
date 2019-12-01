import sys
import os

# Add the project root to the PYTHONPATH env variable
sys.path.insert(0, os.getcwd())
# OSX depdency conflict resolution for ffmpeg and OMP
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from curious_agent.pipelines import Pipeline
from curious_agent.agents.open_ai_agents.a2c_agent import A2CAgent
from curious_agent.environments.open_ai.atari.atari_environment import AtariEnvironment
from curious_agent.util import pipeline_config_loader

# Load Configuration
config = pipeline_config_loader(sys.argv[1])

# Create Environment
env = AtariEnvironment(config['env_config'], atari_wrapper=True)
# Create Agent
agent = A2CAgent(env, config['agent_config'])

# Load the environment and agent into the pipeline
pipeline = Pipeline(train_agent=agent, environment=env, config=config, test=True)

# Run the training loop
pipeline.execute()
# pipeline._load_pipeline()
# pipeline.resume()
