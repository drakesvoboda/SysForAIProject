import os

os.environ["TUNE_RESULT_DIR"] = "/media/drake/BlackPassport/ray_results/"

import ray
import ray.rllib.agents.ppo as ppo
from ray.tune.logger import pretty_print

ray.init()

config = ppo.DEFAULT_CONFIG.copy()
config["num_gpus"] = 0
config["num_workers"] = 3
config["num_envs_per_worker"] = 4
trainer = ppo.PPOTrainer(config=config, env="BreakoutNoFrameskip-v4")

# Can optionally call trainer.restore(path) to load a checkpoint.

for i in range(1000):
   # Perform one iteration of training the policy with PPO
   result = trainer.train()
   print(pretty_print(result))