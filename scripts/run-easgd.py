import os

# os.environ["TUNE_RESULT_DIR"] = "/media/drake/BlackPassport/ray_results/"

import ray
import easgd
from ray.tune.logger import pretty_print

ray.init()

config = easgd.DEFAULT_CONFIG.copy()

config["num_gpus"] = 0
config["num_workers"] = 5
config["num_envs_per_worker"] = 5
config["lr_schedule"] = [[0, 0.0007],[20000000, 0.000000000001],]
config["moving_rate"] = 0.9
config["update_frequency"] = 5

trainer = easgd.EASGDTrainer(config=config, env="LunarLander-v2")

for i in range(1000):
   result = trainer.train()
   print(pretty_print(result))