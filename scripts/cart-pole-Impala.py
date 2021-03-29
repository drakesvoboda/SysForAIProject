import os

os.environ["TUNE_RESULT_DIR"] = "/media/drake/BlackPassport/ray_results/"

import ray
import ray.rllib.agents.impala as impala
from ray.tune.logger import pretty_print

ray.init()

config = impala.DEFAULT_CONFIG.copy()
config["num_gpus"] = 0
config["num_workers"] = 5
config["num_envs_per_worker"] = 5
trainer = impala.ImpalaTrainer(config=config, env="QbertNoFrameskip-v4")

# Can optionally call trainer.restore(path) to load a checkpoint.

for i in range(1000):
   # Perform one iteration of training the policy with PPO
   result = trainer.train()
   print(pretty_print(result))