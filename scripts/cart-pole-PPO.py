import os

# os.environ["TUNE_RESULT_DIR"] = "/media/drake/BlackPassport/ray_results/"

import time
import ray
import ray.rllib.agents.ppo as ppo
from ray.tune.logger import pretty_print

ray.init()

config = ppo.DEFAULT_CONFIG.copy()
config["num_gpus"] = 0
config["num_workers"] = 5
config["num_envs_per_worker"] = 5
trainer = ppo.PPOTrainer(config=config, env="QbertNoFrameskip-v4")

# Can optionally call trainer.restore(path) to load a checkpoint.

start = int(round(time.time()))
while True:
   # Perform one iteration of training the policy with PPO
   elapsed = int(round(time.time())) - start
   if elapsed > 3600:
      break
   result = trainer.train()
   print(pretty_print(result))