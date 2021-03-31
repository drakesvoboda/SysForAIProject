import os

# os.environ["TUNE_RESULT_DIR"] = "/media/drake/BlackPassport/ray_results/"

import ray
import ray.rllib.agents.a3c as a3c
from ray.tune.logger import pretty_print

ray.init()

config = a3c.DEFAULT_CONFIG.copy()
config["num_gpus"] = 0
config["num_workers"] = 5
config["num_envs_per_worker"] = 5
trainer = a3c.A3CTrainer(config=config, env="LunarLander-v2")

# Can optionally call trainer.restore(path) to load a checkpoint.

for i in range(1000):
   # Perform one iteration of training the policy with PPO
   result = trainer.train()
   print(pretty_print(result))